"""Zero-shot LLM baseline: ask Claude which metro a user (or tweet) is from.

No training data is used: the model sees only the tweets and the list of 41
metros, so every labelled user can be evaluated. The answer is constrained to
the label set with a JSON schema (structured outputs), so there is no
free-text parsing.

Responses are cached under .cache/llm/ so re-running the report costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .labels import METROS

DEFAULT_MODEL = "claude-opus-5-5"
CACHE_DIR = Path(".cache/llm")
MAX_TWEETS_PER_USER = 300  # most recent tweets; ~6-10k tokens per user

SYSTEM = (
    "You are a geolocation analyst. Given tweets written by one Twitter user (or a single tweet), "
    "infer which Canadian census metropolitan area the author lives in. Use every clue: places, "
    "teams, transit, weather, local news, slang, language (French suggests Quebec), events. "
    "Always choose exactly one metro from the allowed list, even when unsure."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences naming the strongest clues."},
        "metro": {"type": "string", "enum": METROS},
    },
    "required": ["reasoning", "metro"],
    "additionalProperties": False,
}


def _client():
    import anthropic

    return anthropic.Anthropic(max_retries=6)


def _ask(client, model: str, text: str) -> dict:
    key = hashlib.sha256(f"{model}\n{SYSTEM}\n{text}".encode()).hexdigest()[:24]
    path = CACHE_DIR / model / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    response = client.beta.messages.create(
        model=model,
        max_tokens=2000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Allowed metros: {', '.join(METROS)}\n\nTweets:\n{text}"}],
    )
    if response.stop_reason == "refusal":
        result = {"metro": None, "reasoning": "refused"}
    else:
        body = next(b.text for b in response.content if b.type == "text")
        result = json.loads(body)
    result["usage"] = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False))
    return result


def predict(texts: list[str], model: str = DEFAULT_MODEL, workers: int = 8) -> tuple[np.ndarray, dict]:
    client = _client()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda t: _ask(client, model, t), texts))
    usage = {
        "input_tokens": sum(r["usage"]["input"] for r in results),
        "output_tokens": sum(r["usage"]["output"] for r in results),
        "refusals": sum(r["metro"] is None for r in results),
    }
    return np.array([r["metro"] for r in results], dtype=object), usage


def _recent_tweets(text: str, n: int = MAX_TWEETS_PER_USER) -> str:
    return "\n".join(text.split("\n")[-n:])


def run(level: str = "user", limit: int | None = None, model: str | None = None) -> None:
    """Evaluate the zero-shot baseline and merge it into the results files."""
    from .data import load_geotagged_tweets, load_users
    from .experiments import RESULTS_DIR
    from .metrics import score
    from .social import metro_to_province

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("Note: no ANTHROPIC_API_KEY set; relying on another credential source.")
    model = model or DEFAULT_MODEL

    if level == "user":
        df = load_users()
        texts = [_recent_tweets(t) for t in df["text"]]
        tasks = [("user_text_metro", "metro", df["metro"].to_numpy())]
    else:
        df = load_geotagged_tweets().sample(n=limit or 2000, random_state=42)
        texts = df["text"].tolist()
        tasks = [("tweet_text_metro", "metro", df["metro"].to_numpy())]
    if limit and level == "user":
        texts, df = texts[:limit], df.iloc[:limit]
        tasks = [(t, lvl, y[:limit]) for t, lvl, y in tasks]

    pred, usage = predict(texts, model)
    pred = np.where(pred == None, "Toronto", pred)  # noqa: E711  refusals count as the majority guess

    for task, lvl, y in tasks:
        _merge(
            RESULTS_DIR / f"{task}.json",
            f"llm_{model}",
            {
                "description": f"2026: zero-shot {model}, structured output, no training",
                "cv": score(y, pred, lvl).as_dict(),
                "usage": usage,
            },
        )
    if level == "user":
        y_prov = df["metro"].map(metro_to_province).to_numpy()
        p_prov = np.array([metro_to_province(p) for p in pred])
        _merge(
            RESULTS_DIR / "user_text_province.json",
            f"llm_{model}",
            {
                "description": f"2026: zero-shot {model} (metro answer mapped to province)",
                "cv": score(y_prov, p_prov, "province").as_dict(),
                "usage": usage,
            },
        )
    print(json.dumps(usage))


def _merge(path: Path, name: str, entry: dict) -> None:
    results = json.loads(path.read_text()) if path.exists() else {}
    results[name] = entry
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    cv = entry["cv"]
    print(
        f"  {path.stem:22s} {name:24s} acc={cv['accuracy']:.3f} acc161={cv['acc161']:.3f} "
        f"median={cv['median_km']:.1f}km (n={cv['n']})"
    )

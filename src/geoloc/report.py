"""Turn results/*.json into results/summary.md and the README figure."""

from __future__ import annotations

import json
from pathlib import Path

from .experiments import RESULTS_DIR

TASK_TITLES = {
    "user_text_metro": "User level, from tweets: 41 metros (5-fold CV, 420 users)",
    "user_text_province": "User level, from tweets: 10 provinces (5-fold CV, 420 users)",
    "user_friends_metro": "User level, from friends' locations: 41 metros",
    "user_followers_metro": "User level, from followers' locations: 41 metros",
    "user_graph_metro": "User level, from friends + followers: 41 metros",
    "tweet_text_metro": "Single tweet: 41 metros (5-fold CV grouped by user, 42k tweets)",
}

# Categorical slots in fixed order (validated: scripts in the dataviz guide).
FAMILY_COLORS = {
    "Baseline": "#a9a8a2",
    "2019 text model": "#2a78d6",
    "2026 text model": "#eb6834",
    "Social graph": "#1baf7a",
    "Zero-shot LLM": "#eda100",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _family(task: str, name: str) -> str:
    if name.startswith("llm_"):
        return "Zero-shot LLM"
    if task.startswith(("user_friends", "user_followers", "user_graph")):
        return "Social graph"
    if name in ("majority", "uniform"):
        return "Baseline"
    if name.endswith("_2019"):
        return "2019 text model"
    return "2026 text model"


def _load() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))}


def _table(task: str, results: dict) -> str:
    rows = [
        "| Model | Accuracy | Acc@161 km | Median error | Mean error | 2019 split Acc@161 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    entries = [(k, v) for k, v in results.items() if not k.startswith("_")]
    entries.sort(key=lambda kv: -kv[1]["cv"]["acc161"])
    for _name, e in entries:
        cv, old = e["cv"], e.get("split_2019")
        old_s = f"{old['acc161']:.1%}" if old else "–"
        rows.append(
            f"| {e['description']} | {cv['accuracy']:.1%} | **{cv['acc161']:.1%}** | "
            f"{cv['median_km']:,.0f} km | {cv['mean_km']:,.0f} km | {old_s} |"
        )
    return f"### {TASK_TITLES.get(task, task)}\n\n" + "\n".join(rows) + "\n"


def figure(all_results: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [
        ("user_text_metro", "uniform", "Random guess"),
        ("user_text_metro", "majority", "Always 'Toronto'"),
        ("user_text_metro", "mnb_2019", "Naive Bayes (2019)"),
        ("user_text_metro", "mlp_2019", "MLP (2019)"),
        ("user_text_metro", "svm_2019", "Linear SVM (2019)"),
        ("user_text_metro", "embed_2026", "Sentence embeddings (2026)"),
        ("user_text_metro", "logreg_2026", "TF-IDF + hashtags (2026)"),
        ("user_text_metro", "llm_claude-opus-5-5", "Claude, zero-shot (2026)"),
        ("user_friends_metro", "vote", "Friends' locations (vote)"),
        ("user_followers_metro", "vote", "Followers' locations (vote)"),
        ("user_graph_metro", "vote", "Friends + followers (vote)"),
    ]
    rows = [
        (label, all_results[t][m]["cv"]["acc161"], _family(t, m))
        for t, m, label in picks
        if t in all_results and m in all_results[t]
    ]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(rows))[::-1]
    for yi, (_label, val, fam) in zip(y, rows, strict=True):
        ax.barh(yi, val, height=0.62, color=FAMILY_COLORS[fam], edgecolor=SURFACE, linewidth=2)
        ax.text(val + 0.01, yi, f"{val:.0%}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(list(y), [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", colors=INK_2, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Share of users placed within 161 km of their home metro (Acc@161)",
        loc="left",
        fontsize=10.5,
        color=INK,
        pad=24,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY_COLORS.items() if f in {r[2] for r in rows}]
    labels = [f for f in FAMILY_COLORS if f in {r[2] for r in rows}]
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_2,
        handlelength=1,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def build() -> None:
    results = _load()
    order = [t for t in TASK_TITLES if t in results]
    md = [
        "# Results\n",
        "Generated by `geoloc report` from `results/*.json`. "
        "Accuracy = exact metro; Acc@161 = within 161 km (100 miles). "
        "'2019 split' = the thesis protocol (one 80/20 split, random_state=42).\n",
    ]
    md += [_table(t, results[t]) for t in order]
    (RESULTS_DIR / "summary.md").write_text("\n".join(md))
    figure(results, RESULTS_DIR / "acc161.png")
    print(f"Wrote {RESULTS_DIR / 'summary.md'} and {RESULTS_DIR / 'acc161.png'}")

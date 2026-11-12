"""Web demo: paste a few tweets, see which Canadian metro they sound like.

Run locally:   python app/app.py
On Hugging Face Spaces this file is the entry point (see app/README.md).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))  # use the package when run from the repo
try:
    from geoloc.labels import METRO_COORDS
except ImportError:  # deployed alone: coordinates ship next to the app
    import json

    METRO_COORDS = {k: tuple(v) for k, v in json.loads((HERE / "metros.json").read_text()).items()}

bundle = joblib.load(HERE / "model.joblib")
MODEL, CV = bundle["model"], bundle["cv"]
VECT, CLF = MODEL.named_steps["vect"], MODEL.named_steps["clf"]
VOCAB = np.array(VECT.get_feature_names_out())

EXAMPLES = [
    "Stuck on the 401 again 🙄\nLeafs blew a 3-goal lead, classic\nTTC delays on Line 1, walking to Union",
    "Il neige encore sur le Plateau ❄️\nGo Habs Go! 🔵⚪🔴\nPoutine à La Banquise après le show",
    "Chinook today, +12 in January 😎\nFlames game tonight at the Saddledome\nStampede tickets booked!",
    "Rain again. Shocking.\nSeawall bike ride before it pours\nCanucks lose, sushi makes it better",
    "Skating the canal before work ⛸️\nParliament Hill is packed for Canada Day\nBeaver tails in the ByWard Market",
    "#yyc traffic on Deerfoot is wild today\nGo @NHLFlames! #cofred",
]

INK, MUTED, ACCENT, SURFACE = "#0b0b0b", "#a9a8a2", "#eb6834", "#fcfcfb"

# The training tweets were normalised in 2019 with ekphrasis: mentions -> <user>,
# links -> <url>, numbers -> <number>, #tag -> <hashtag> tag <hashtag>, and SHOUTED
# words wrapped in <allcaps>. Pasted tweets get the same treatment so the model
# sees the tokens it learned ("hashtag yyc", "allcaps ttc").
_RULES = [
    (re.compile(r"https?://\S+|www\.\S+"), " <url> "),
    (re.compile(r"@\w+"), " <user> "),
    (re.compile(r"#(\w+)"), lambda m: f" <hashtag> {m.group(1)} <hashtag> "),
    (re.compile(r"\b([A-Z]{2,})\b"), lambda m: f" <allcaps> {m.group(1)} <allcaps> "),
    (re.compile(r"\b\d+(?:[.,]\d+)?\b"), " <number> "),
]


def normalise(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).lower().strip()


def _map(probs: dict[str, float]) -> go.Figure:
    names = list(METRO_COORDS)
    p = np.array([probs.get(n, 0.0) for n in names])
    top = names[int(p.argmax())]
    fig = go.Figure(
        go.Scattergeo(
            lat=[METRO_COORDS[n][0] for n in names],
            lon=[METRO_COORDS[n][1] for n in names],
            text=[f"{n}: {probs.get(n, 0):.0%}" for n in names],
            hoverinfo="text",
            mode="markers",
            marker=dict(
                size=5 + 140 * p,
                color=[ACCENT if n == top else MUTED for n in names],
                opacity=0.85,
                line=dict(width=2, color=SURFACE),
            ),
        )
    )
    fig.update_geos(
        projection_type="conic conformal",
        projection_rotation=dict(lon=-96),
        center=dict(lat=52, lon=-96),
        lataxis_range=[41, 60],
        lonaxis_range=[-130, -52],
        showland=True,
        landcolor="#f0efec",
        showcountries=True,
        countrycolor="#d4d3cd",
        showsubunits=True,
        subunitcolor="#e4e3df",
        showlakes=True,
        lakecolor=SURFACE,
        bgcolor=SURFACE,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380, paper_bgcolor=SURFACE)
    return fig


def _evidence(text: str, label: str, k: int = 8) -> str:
    """Words that pushed the prediction toward `label` (tf-idf weight x coefficient)."""
    x = VECT.transform([normalise(text)])
    cls = list(CLF.classes_).index(label)
    idx = x.indices
    contrib = x.data * CLF.coef_[cls, idx]
    order = np.argsort(-contrib)
    words = [f"`{VOCAB[idx[i]]}`" for i in order[:k] if contrib[i] > 0]
    return ", ".join(words) if words else "_no words the model knows; it fell back to its priors_"


def predict(text: str):
    text = (text or "").strip()
    if not text:
        return {}, _map({}), "Paste a few tweets to start."
    proba = CLF.predict_proba(VECT.transform([normalise(text)]))[0]
    probs = dict(zip(CLF.classes_, proba.astype(float), strict=True))
    top = max(probs, key=probs.get)
    why = f"**Top guess: {top}.** Words that pointed there: {_evidence(text, top)}"
    return probs, _map(probs), why


with gr.Blocks(title="Where do these tweets come from?") as demo:
    gr.Markdown(
        "# Where do these tweets come from?\n"
        "Paste a few tweets, one per line. A model trained on 420 Canadian Twitter users (my University of Ottawa "
        "thesis data) guesses which of 41 Canadian metros the author lives in.\n\n"
        f"In cross-validation it puts **{CV['acc161']:.0%}** of users within 161 km of home from text alone; "
        "a user's friends and followers are far stronger evidence (88%), but that needs the social graph. "
        "It has seen ~2,400 tweets per user, so a few lines give it much less to go on: expect it to be "
        "wrong often, especially between Ottawa and Montreal, whose vocabularies overlap. "
        "[Code and write-up](https://github.com/mertmetin/Geolocation-Prediction)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            box = gr.Textbox(lines=8, label="Tweets", placeholder="One tweet per line…")
            btn = gr.Button("Guess the city", variant="primary")
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=1):
            label = gr.Label(num_top_classes=5, label="Most likely metros")
            why = gr.Markdown()
    geo = gr.Plot(label="Probability by metro")
    btn.click(predict, box, [label, geo, why])
    box.submit(predict, box, [label, geo, why])

if __name__ == "__main__":
    demo.launch()

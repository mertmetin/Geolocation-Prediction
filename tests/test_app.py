import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")
APP_DIR = Path(__file__).parents[1] / "app"
if not (APP_DIR / "model.joblib").exists():
    pytest.skip("run `geoloc train-demo` first", allow_module_level=True)
sys.path.insert(0, str(APP_DIR))

import app  # noqa: E402


def test_normalise_matches_2019_preprocessing():
    out = app.normalise("Go @NHLFlames! #yyc TTC delays 3 times https://t.co/x")
    assert out == "go <user> ! <hashtag> yyc <hashtag> <allcaps> ttc <allcaps> delays <number> times <url>"


def test_predict_returns_probabilities_for_every_metro():
    probs, fig, why = app.predict("#yyc traffic on Deerfoot\nStampede tickets booked")
    assert len(probs) == 41
    assert abs(sum(probs.values()) - 1) < 1e-6
    assert max(probs, key=probs.get) == "Calgary"
    assert "Calgary" in why


def test_empty_input():
    probs, _, why = app.predict("   ")
    assert probs == {} and "Paste" in why

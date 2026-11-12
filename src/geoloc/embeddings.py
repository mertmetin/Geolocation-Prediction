"""Sentence-embedding classifier.

Each tweet is embedded with a small multilingual sentence-transformer
(English and French share one vector space), a user is the mean of their
tweet vectors, and a logistic regression is trained on top. Embeddings are
computed once and cached under .cache/embeddings/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = Path(".cache/embeddings")
TWEETS_PER_USER = 300


def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str], cache_name: str, batch_size: int = 256) -> np.ndarray:
    path = CACHE_DIR / f"{cache_name}.npy"
    if path.exists():
        return np.load(path)
    vecs = _encoder().encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def user_vectors(users: pd.DataFrame) -> np.ndarray:
    """Mean of each user's most recent tweet embeddings."""
    tweets, owner = [], []
    for i, text in enumerate(users["text"]):
        recent = [t for t in text.split("\n")[-TWEETS_PER_USER:] if t.strip()]
        tweets += recent
        owner += [i] * len(recent)
    vecs = embed(tweets, f"users_{TWEETS_PER_USER}")
    owner = np.array(owner)
    out = np.zeros((len(users), vecs.shape[1]), dtype=np.float32)
    for i in range(len(users)):
        out[i] = vecs[owner == i].mean(axis=0)
    return out


def classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    # Class weighting pushed predictions toward metros with two users and fell
    # below the majority baseline; plain L2-normalised vectors do best in CV.
    return make_pipeline(Normalizer(), LogisticRegression(C=10, max_iter=3000))


def run(level: str) -> None:
    from .data import load_geotagged_tweets, load_users
    from .experiments import run_task
    from .social import metro_to_province

    model = {"embed_2026": ("2026: multilingual MiniLM sentence embeddings + logistic regression", classifier)}
    if level == "user":
        users = load_users()
        X = user_vectors(users)
        run_task("user_text_metro", "metro", X, users["metro"].to_numpy(), model)
        run_task("user_text_province", "province", X, users["metro"].map(metro_to_province).to_numpy(), model)
    else:
        tweets = load_geotagged_tweets()
        X = embed(tweets["text"].tolist(), "geotagged_tweets")
        run_task(
            "tweet_text_metro", "metro", X, tweets["metro"].to_numpy(), model, groups=tweets["screen_name"].to_numpy()
        )

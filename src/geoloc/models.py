"""Classifiers: the 2019 thesis configurations (bugs fixed) and 2026 baselines.

Every model is a factory returning an object with `fit(X, y)` and
`predict(X)`, so the experiment runner can treat them all the same way.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

Factory = Callable[[], object]

# English + French stop words: about a quarter of users tweet mostly in French,
# and the 2019 pipelines removed English stop words only.
_FR_STOP = (
    "au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me même mes "
    "moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un "
    "une vos votre vous c d j l à m n s t y été être avoir fait est sont ai as a ça cette plus tout très"
).split()


# --------------------------------------------------------------------------- text, 2019 configs


def legacy_majority() -> object:
    return DummyClassifier(strategy="most_frequent")


def legacy_uniform() -> object:
    """The thesis's random baseline (DummyClassifier(strategy='uniform'))."""
    return DummyClassifier(strategy="uniform", random_state=1)


def legacy_mnb() -> object:
    return Pipeline(
        [
            ("vect", CountVectorizer(stop_words="english")),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )


def legacy_svm() -> object:
    return Pipeline(
        [
            ("vect", TfidfVectorizer(stop_words="english")),
            ("clf", LinearSVC(C=1, loss="hinge", fit_intercept=False, max_iter=1000, random_state=1)),
        ]
    )


def legacy_mlp() -> object:
    """The configuration left active in `User Level/svm_and_mb_metro.py`.

    Two changes so it trains in minutes, not hours: 2^18 hashed features
    (was 2^20) and at most 200 epochs (was 1000).
    """
    return Pipeline(
        [
            ("vect", HashingVectorizer(stop_words="english", n_features=2**18)),
            ("tfidf", TfidfTransformer()),
            ("clf", MLPClassifier(solver="sgd", alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=200)),
        ]
    )


# --------------------------------------------------------------------------- text, 2026 baselines


# Words and #hashtags. @handles and bare numbers are dropped: handles tie a
# model to individual accounts, and both overfit on 420 users.
_TWEET_TOKENS = r"(?u)#?\b[^\W\d_][\w']+\b"


def tfidf_logreg() -> object:
    """Word uni+bigram TF-IDF with a class-balanced logistic regression.

    Hashtags are kept as tokens (#yyc, #habs); English and French stop words are
    removed. A term must occur in at least 5 users' tweets and the vocabulary is
    capped at 30k. That regularises better than a larger vocabulary (CV acc161
    0.61 vs 0.53 with min_df=2 / 300k terms) and keeps the model small and free
    of anything specific to one person, so the same model powers the web demo.
    """
    return Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    lowercase=True,
                    token_pattern=_TWEET_TOKENS,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.8,
                    sublinear_tf=True,
                    max_features=30_000,
                    stop_words=_FR_STOP + list(TfidfVectorizer(stop_words="english").get_stop_words()),
                    dtype=np.float32,
                ),
            ),
            ("clf", LogisticRegression(C=20, max_iter=3000, class_weight="balanced")),
        ]
    )


def tfidf_cnb() -> object:
    """Complement NB: a strong, fast baseline for skewed text classes."""
    return Pipeline(
        [
            ("vect", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", ComplementNB(alpha=0.3)),
        ]
    )


# --------------------------------------------------------------------------- social graph


class NeighbourVote:
    """Predict the metro where most friends/followers live (no training needed).

    Falls back to the training majority when no neighbour location parses.
    """

    def fit(self, X: pd.DataFrame, y):
        self.fallback_ = pd.Series(y).mode()[0]
        return self

    def predict(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        has = X.sum(axis=1) > 0
        return np.where(has, X.idxmax(axis=1), self.fallback_)


def social_logreg() -> object:
    """Logistic regression on log-scaled neighbour location counts."""
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )


TEXT_MODELS: dict[str, tuple[str, Factory]] = {
    "majority": ("2019 baseline: always predict the most common class", legacy_majority),
    "uniform": ("2019 baseline: random class", legacy_uniform),
    "mnb_2019": ("2019: TF-IDF + Multinomial NB", legacy_mnb),
    "svm_2019": ("2019: TF-IDF + linear SVM", legacy_svm),
    "mlp_2019": ("2019: hashed TF-IDF + MLP", legacy_mlp),
    "cnb_2026": ("2026: word-bigram TF-IDF + Complement NB", tfidf_cnb),
    "logreg_2026": ("2026: word + char TF-IDF + logistic regression", tfidf_logreg),
}

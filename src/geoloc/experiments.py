"""Experiment runner: fixed evaluation protocols, one results file per task.

Protocols
---------
User level (420 users): 5-fold cross-validation, so every user is predicted
exactly once by a model that never saw them. 84 test users (the thesis's
single 80/20 split) is too few to tell models apart, and several metros have
only two users. The thesis split is also reported, as `split_2019`, so the
numbers can be compared directly.

Tweet level (42k geotagged tweets): the thesis split tweets at random, so the
same person's tweets landed in both train and test and the model could
memorise people rather than places. The main protocol here holds out whole
users (GroupKFold on screen_name); the random split is kept as `split_2019`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from .metrics import score

RESULTS_DIR = Path("results")
SEED = 42


def _fit_predict(factory: Callable[[], object], X_tr, y_tr, X_te) -> np.ndarray:
    model = factory()
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te))


def _take(X, idx):
    if isinstance(X, pd.DataFrame | pd.Series):
        return X.iloc[idx]
    if isinstance(X, np.ndarray):
        return X[idx]
    return [X[i] for i in idx]


def cross_val_predict(factory, X, y: np.ndarray, groups=None, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions for every row."""
    splitter = GroupKFold(n_splits) if groups is not None else KFold(n_splits, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), dtype=object)
    for tr, te in splitter.split(np.zeros(len(y)), y, groups):
        pred[te] = _fit_predict(factory, _take(X, tr), y[tr], _take(X, te))
    return pred


def split_2019(factory, X, y: np.ndarray):
    """The thesis protocol: one 80/20 split, random_state=42, rows in database order."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    return y[te], _fit_predict(factory, _take(X, tr), y[tr], _take(X, te))


def run_task(
    task: str,
    level: str,
    X,
    y: np.ndarray,
    models: dict[str, tuple[str, Callable[[], object]]],
    groups=None,
    with_2019_split: bool = True,
) -> dict:
    """Evaluate every model on one task and write results/<task>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{task}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("_meta", {}).update({"task": task, "level": level, "n": int(len(y))})

    for name, (desc, factory) in models.items():
        t0 = time.time()
        entry = {"description": desc}
        pred = cross_val_predict(factory, X, y, groups)
        entry["cv"] = score(y, pred, level).as_dict()
        if with_2019_split:
            yt, yp = split_2019(factory, X, y)
            entry["split_2019"] = score(yt, yp, level).as_dict()
        entry["seconds"] = round(time.time() - t0, 1)
        results[name] = entry
        cv = entry["cv"]
        print(
            f"  {task:22s} {name:16s} acc={cv['accuracy']:.3f} acc161={cv['acc161']:.3f} "
            f"median={cv['median_km']:7.1f}km mean={cv['mean_km']:7.1f}km  ({entry['seconds']}s)"
        )
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results

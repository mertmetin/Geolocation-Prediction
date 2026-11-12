"""Train the model behind the web demo: `logreg_2026` fitted on all 420 users.

Its vocabulary has no @handles, only terms used by at least 5 different
users, and at most 30k terms (~5 MB on disk), so it is safe to publish.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .models import tfidf_logreg as build

DEMO_MODEL_PATH = Path("app/model.joblib")


def train(out: Path = DEMO_MODEL_PATH) -> None:
    from .data import load_users
    from .experiments import cross_val_predict
    from .metrics import score

    users = load_users()
    X, y = users["text"].tolist(), users["metro"].to_numpy()
    cv = score(y, cross_val_predict(build, X, y), "metro")
    print(f"demo model 5-fold CV: acc={cv.accuracy:.3f} acc161={cv.acc161:.3f} median={cv.median_km:.0f}km")

    model = build().fit(X, y)
    clf = model.named_steps["clf"]
    clf.coef_ = clf.coef_.astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "cv": cv.as_dict()}, out, compress=3)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

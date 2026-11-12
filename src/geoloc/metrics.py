"""Geolocation metrics used in the thesis and the wider literature.

* mean / median error distance (km) between the predicted and true reference points
* Acc@161: share of predictions within 161 km (100 miles) of the truth
* exact accuracy: predicted label == true label
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .labels import coords_for

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))


@dataclass
class Scores:
    n: int
    accuracy: float
    acc161: float
    mean_km: float
    median_km: float

    def as_dict(self) -> dict:
        return asdict(self)


def error_distances(y_true: Sequence[str], y_pred: Sequence[str], level: str) -> np.ndarray:
    return np.array(
        [haversine_km(coords_for(t, level), coords_for(p, level)) for t, p in zip(y_true, y_pred, strict=True)]
    )


def score(y_true: Sequence[str], y_pred: Sequence[str], level: str) -> Scores:
    d = error_distances(y_true, y_pred, level)
    exact = np.mean([t == p for t, p in zip(y_true, y_pred, strict=True)])
    return Scores(
        n=len(d),
        accuracy=float(exact),
        acc161=float(np.mean(d <= 161)),
        mean_km=float(d.mean()),
        median_km=float(np.median(d)),
    )

"""Social-graph features: where do a user's friends and followers say they live?

The 2019 scripts built one count dictionary per user with `cpy = D`, which
copies the *reference*, not the dictionary. Every user's feature vector was
the same shared dictionary, accumulating counts across all users, so every
row of X was identical and the classifier could only learn the class prior.
Here each user gets a fresh Counter.

Declared locations are free text ("Toronto, ON", "the 6ix", "YYC",
"Montréal 🇨🇦"). The 2019 code used GeoText, a world-wide city list that
missed many Canadian spellings and nicknames; `parse_location` matches
against the CMA names, their member municipalities and common nicknames.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

from .labels import CITY_TO_METRO, METROS, PROVINCES, canonical_metro

_NICKNAMES = {
    "the 6ix": "Toronto",
    "the six": "Toronto",
    "6ix": "Toronto",
    "the 6": "Toronto",
    "gta": "Toronto",
    "yyz": "Toronto",
    "t.o": "Toronto",
    "toronto": "Toronto",
    "mtl": "Montreal",
    "yul": "Montreal",
    "yvr": "Vancouver",
    "van city": "Vancouver",
    "vancity": "Vancouver",
    "yyc": "Calgary",
    "yeg": "Edmonton",
    "yow": "Ottawa-Gatineau",
    "ywg": "Winnipeg",
    "yhz": "Halifax",
    "kw": "Kitchener-Cambridge-Waterloo",
    "yqr": "Regina",
    "yxe": "Saskatoon",
    "the hammer": "Hamilton",
    "ldnont": "London",
    "london, on": "London",
    "london ontario": "London",
    "london, ontario": "London",
    "london on": "London",
    "london, ont": "London",
    "yyj": "Victoria",
    "yqg": "Windsor",
}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c) and (c.isalnum() or c in " ,.'-/"))
    return re.sub(r"\s+", " ", s.replace("–", "-").lower()).strip()


def _build_patterns() -> list[tuple[re.Pattern, str]]:
    names: dict[str, str] = {}
    for m in METROS:
        names[_fold(m)] = m
        for part in re.split(r"\s*-\s*|–", m):  # "Kitchener-Cambridge-Waterloo" -> parts
            if len(part) > 3:
                names.setdefault(_fold(part), m)
    for city, metro in CITY_TO_METRO.items():
        names[_fold(city)] = metro
    names.update({_fold(k): v for k, v in _NICKNAMES.items()})
    # "London" alone is ambiguous (UK). Only accept it with an Ontario hint.
    names.pop("london", None)
    # Longest names first so "St. Catharines" wins over "Catharines".
    ordered = sorted(names.items(), key=lambda kv: -len(kv[0]))
    return [(re.compile(rf"(?<![a-z]){re.escape(k)}(?![a-z])"), v) for k, v in ordered]


_PATTERNS = _build_patterns()


def parse_location(text: str | None) -> str | None:
    """Map a free-text declared location to a canonical metro, or None."""
    if not text:
        return None
    direct = canonical_metro(text)
    if direct:
        return direct
    s = _fold(text)
    if not s:
        return None
    for pat, metro in _PATTERNS:
        if pat.search(s):
            return metro
    return None


def location_counts(graph: pd.DataFrame, users: pd.Series) -> pd.DataFrame:
    """Rows = users, columns = metros, values = how many neighbours live there."""
    parsed = graph.assign(metro=graph["user_declared_location"].map(parse_location)).dropna(subset=["metro"])
    by_user = {name: Counter(g["metro"]) for name, g in parsed.groupby("of")}
    rows = [{m: by_user.get(u, Counter()).get(m, 0) for m in METROS} for u in users]
    return pd.DataFrame(rows, index=users.values, columns=METROS)


def metro_to_province(metro: str) -> str:
    return _METRO_PROVINCE[metro]


_METRO_PROVINCE = {
    "Toronto": "Ontario",
    "Montreal": "Quebec",
    "Vancouver": "British Columbia",
    "Calgary": "Alberta",
    "Ottawa-Gatineau": "Ontario/Quebec",
    "Edmonton": "Alberta",
    "Quebec City": "Quebec",
    "Winnipeg": "Manitoba",
    "Hamilton": "Ontario",
    "Kitchener-Cambridge-Waterloo": "Ontario",
    "London": "Ontario",
    "St. Catharines - Niagara": "Ontario",
    "Halifax": "Nova Scotia",
    "Oshawa": "Ontario",
    "Victoria": "British Columbia",
    "Windsor": "Ontario",
    "Saskatoon": "Saskatchewan",
    "Regina": "Saskatchewan",
    "Sherbrooke": "Quebec",
    "St. John's": "Newfoundland and Labrador",
    "Barrie": "Ontario",
    "Kelowna": "British Columbia",
    "Abbotsford–Mission": "British Columbia",
    "Greater Sudbury": "Ontario",
    "Kingston": "Ontario",
    "Saguenay": "Quebec",
    "Trois-Rivieres": "Quebec",
    "Guelph": "Ontario",
    "Moncton": "New Brunswick",
    "Brantford": "Ontario",
    "Saint John": "New Brunswick",
    "Peterborough": "Ontario",
    "Thunder Bay": "Ontario",
    "Lethbridge": "Alberta",
    "Nanaimo": "British Columbia",
    "Kamloops": "British Columbia",
    "Belleville": "Ontario",
    "Chatham-Kent": "Ontario",
    "Fredericton": "New Brunswick",
    "Chilliwack": "British Columbia",
    "Red Deer": "Alberta",
}
assert set(_METRO_PROVINCE) == set(METROS) and set(_METRO_PROVINCE.values()) <= set(PROVINCES)

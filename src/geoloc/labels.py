"""Canonical label sets, name aliases and reference coordinates.

The 2019 scripts spelled some labels differently in different files
("Quebec City", "Québec City", "Québec metro"), so users from that metro
silently fell out of some experiments. Everything now goes through
`canonical_metro`, which maps every known variant to one name.

Coordinates are (lat, lon) with western longitudes negative. The 2019 JSON
files stored longitudes as positive numbers; distances were still correct
(mirroring every point across the prime meridian preserves distances), but
the points could not be plotted on a map.
"""

from __future__ import annotations

import unicodedata

# 41 census metropolitan areas / agglomerations with population > 100k.
# Reference points are the thesis's own (legacy/*/coordinates_metro.json), longitudes negated.
METRO_COORDS: dict[str, tuple[float, float]] = {
    "Toronto": (44.0384, -79.2),
    "Montreal": (45.6066, -73.7124),
    "Vancouver": (49.19854, -123.224215),
    "Calgary": (51.05011, -114.08529),
    "Ottawa-Gatineau": (45.4522, -75.6292),
    "Edmonton": (53.6865, -113.5208),
    "Quebec City": (46.829853, -71.254028),
    "Winnipeg": (49.8951, -97.1384),
    "Hamilton": (43.2557, -79.8711),
    "Kitchener-Cambridge-Waterloo": (43.4516, -80.4925),
    "London": (42.9849, -81.2453),
    "St. Catharines - Niagara": (43.0896, -79.0849),
    "Halifax": (44.6488, -63.5752),
    "Oshawa": (43.8971, -78.8658),
    "Victoria": (48.4284, -123.3656),
    "Windsor": (42.3149, -83.0364),
    "Saskatoon": (52.1332, -106.67),
    "Regina": (50.4452, -104.6189),
    "Sherbrooke": (45.4042, -71.8929),
    "St. John's": (47.5615, -52.7126),
    "Barrie": (44.3894, -79.6903),
    "Kelowna": (49.888, -119.496),
    "Abbotsford–Mission": (49.0504, -122.3045),
    "Greater Sudbury": (46.4917, -80.993),
    "Kingston": (44.2312, -76.486),
    "Saguenay": (48.4281, -71.0685),
    "Trois-Rivieres": (46.343, -72.5421),
    "Guelph": (43.5448, -80.2482),
    "Moncton": (46.0878, -64.7782),
    "Brantford": (43.1394, -80.2644),
    "Saint John": (45.2733, -66.0633),
    "Peterborough": (44.3091, -78.3197),
    "Thunder Bay": (48.3809, -89.2477),
    "Lethbridge": (49.6935, -112.8418),
    "Nanaimo": (49.1659, -123.9401),
    "Kamloops": (50.6745, -120.3273),
    "Belleville": (44.1628, -77.3832),
    "Chatham-Kent": (42.4048, -82.191),
    "Fredericton": (45.9636, -66.6431),
    "Chilliwack": (49.15794, -121.951469),
    "Red Deer": (52.2681, -113.8112),
}

PROVINCE_COORDS: dict[str, tuple[float, float]] = {
    "Alberta": (53.9333, -116.5765),
    "British Columbia": (53.7267, -127.6476),
    "Manitoba": (53.7609, -98.8139),
    "Newfoundland and Labrador": (53.1355, -57.6604),
    "New Brunswick": (46.5653, -66.4619),
    "Nova Scotia": (44.6820, -63.7443),
    "Ontario": (51.2538, -85.3232),
    # Ottawa-Gatineau straddles the border; the thesis treated it as its own class.
    "Ontario/Quebec": (50.78681, -86.08282),
    "Quebec": (52.9399, -73.5491),
    "Saskatchewan": (52.9399, -106.4509),
}

METROS: list[str] = list(METRO_COORDS)
PROVINCES: list[str] = list(PROVINCE_COORDS)

# Spelling variants seen in the data and the 2019 code.
_METRO_ALIASES = {
    "quebec metro": "Quebec City",
    "quebec": "Quebec City",
    "quebec city": "Quebec City",
    "abbotsford-mission": "Abbotsford–Mission",
    "abbotsford mission": "Abbotsford–Mission",
    "trois-rivieres": "Trois-Rivieres",
    "st. catharines-niagara": "St. Catharines - Niagara",
}

# Municipalities that belong to a larger CMA (from the 2019 `check_metro`).
CITY_TO_METRO = {
    "Mississauga": "Toronto",
    "Brampton": "Toronto",
    "Markham": "Toronto",
    "Vaughan": "Toronto",
    "Laval": "Montreal",
    "Longueuil": "Montreal",
    "Surrey": "Vancouver",
    "Burnaby": "Vancouver",
    "Ottawa": "Ottawa-Gatineau",
    "Gatineau": "Ottawa-Gatineau",
    "Levis": "Quebec City",
    "Lévis": "Quebec City",
    "Québec": "Quebec City",
    "Burlington": "Hamilton",
    "Niagara Falls": "St. Catharines - Niagara",
    "Welland": "St. Catharines - Niagara",
    "St. Catharines": "St. Catharines - Niagara",
    "Whitby": "Oshawa",
    "Clarington": "Oshawa",
    "Magog": "Sherbrooke",
    "Conception Bay South": "St. John's",
    "Mount Pearl": "St. John's",
    "Paradise": "St. John's",
    "Innisfil": "Barrie",
    "Kelowna West": "Kelowna",
    "South Frontenac": "Kingston",
    "Loyalist": "Kingston",
    "Dieppe": "Moncton",
    "Riverview": "Moncton",
    "Quispamsis": "Saint John",
    "Selwyn": "Peterborough",
    "Quinte West": "Belleville",
    "Kitchener": "Kitchener-Cambridge-Waterloo",
    "Waterloo": "Kitchener-Cambridge-Waterloo",
    "Cambridge": "Kitchener-Cambridge-Waterloo",
    "Abbotsford": "Abbotsford–Mission",
    "Mission": "Abbotsford–Mission",
    "Sudbury": "Greater Sudbury",
    "Trois-Rivières": "Trois-Rivieres",
    "Chatham": "Chatham-Kent",
}


def _fold(s: str) -> str:
    """Lowercase and strip accents and dash variants for fuzzy matching."""
    s = unicodedata.normalize("NFKD", s.strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("–", "-").replace("—", "-").lower()


_FOLDED_METROS = {_fold(m): m for m in METROS}
_FOLDED_ALIASES = {_fold(k): v for k, v in _METRO_ALIASES.items()}


def canonical_metro(name: str | None) -> str | None:
    """Return the canonical metro name for any known spelling, or None."""
    if not name:
        return None
    key = _fold(name)
    return _FOLDED_METROS.get(key) or _FOLDED_ALIASES.get(key)


def canonical_province(name: str | None) -> str | None:
    if not name:
        return None
    key = _fold(name)
    for p in PROVINCES:
        if _fold(p) == key:
            return p
    return None


def coords_for(label: str, level: str) -> tuple[float, float]:
    table = METRO_COORDS if level == "metro" else PROVINCE_COORDS
    return table[label]

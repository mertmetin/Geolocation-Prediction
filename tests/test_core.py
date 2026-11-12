import numpy as np
import pandas as pd
import pytest

from geoloc.data import _parse_tuples, read_dump
from geoloc.labels import METROS, PROVINCES, canonical_metro
from geoloc.metrics import haversine_km, score
from geoloc.models import NeighbourVote
from geoloc.social import location_counts, metro_to_province, parse_location

# --------------------------------------------------------------------------- SQL dump parser


def test_parse_tuples_handles_escapes_commas_and_nulls():
    body = r"""
	(1,'it\'s, (complicated)','line\nbreak',NULL,3.5),
	(2,'O''Brien','',0,-1);"""
    rows = list(_parse_tuples(body))
    assert rows == [
        [1, "it's, (complicated)", "line\nbreak", None, 3.5],
        [2, "O'Brien", "", 0, -1],
    ]


def test_read_dump_reads_only_requested_tables(tmp_path):
    sql = tmp_path / "x.sql"
    sql.write_text(
        "INSERT INTO `Users` (`id`, `screen_name`, `metro`)\nVALUES\n"
        "\t(1,'a','Toronto'),\n\t(2,'b','Québec metro');\n"
        "INSERT INTO `Other` (`id`)\nVALUES\n\t(9);\n",
        encoding="utf-8",
    )
    out = read_dump(sql, ["Users"])
    assert list(out) == ["Users"]
    assert out["Users"].to_dict("records") == [
        {"id": 1, "screen_name": "a", "metro": "Toronto"},
        {"id": 2, "screen_name": "b", "metro": "Québec metro"},
    ]


# --------------------------------------------------------------------------- labels


@pytest.mark.parametrize("raw", ["Quebec City", "Québec City", "Québec metro", "quebec metro"])
def test_quebec_spellings_collapse_to_one_label(raw):
    # 2019 bug: three spellings across scripts silently dropped Quebec City users.
    assert canonical_metro(raw) == "Quebec City"


def test_every_metro_maps_to_a_province():
    assert {metro_to_province(m) for m in METROS} <= set(PROVINCES)
    assert len(METROS) == 41


# --------------------------------------------------------------------------- social graph


@pytest.mark.parametrize(
    "text, metro",
    [
        ("Toronto, Ontario", "Toronto"),
        ("the 6ix 🍁", "Toronto"),
        ("Montréal, QC", "Montreal"),
        ("YYC", "Calgary"),
        ("Mississauga", "Toronto"),
        ("London, Ontario", "London"),
        ("London, England", None),  # not the Ontario one
        ("Los Angeles, CA", None),
        ("Canada", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_location(text, metro):
    assert parse_location(text) == metro


def test_location_counts_are_per_user():
    # 2019 bug: `cpy = D` shared one dict between all users, so every row was identical.
    graph = pd.DataFrame(
        {
            "of": ["alice", "alice", "bob"],
            "screen_name": ["f1", "f2", "f3"],
            "user_declared_location": ["Toronto", "Toronto, ON", "Vancouver, BC"],
        }
    )
    X = location_counts(graph, pd.Series(["alice", "bob", "carol"]))
    assert X.loc["alice", "Toronto"] == 2 and X.loc["alice", "Vancouver"] == 0
    assert X.loc["bob", "Vancouver"] == 1 and X.loc["bob", "Toronto"] == 0
    assert X.loc["carol"].sum() == 0


def test_neighbour_vote_falls_back_to_majority():
    X = pd.DataFrame({"Toronto": [3, 0], "Calgary": [1, 0]})
    model = NeighbourVote().fit(X, ["Calgary", "Calgary", "Toronto"])
    assert list(model.predict(X)) == ["Toronto", "Calgary"]


# --------------------------------------------------------------------------- metrics


def test_haversine_toronto_montreal():
    assert haversine_km((43.6532, -79.3832), (45.5017, -73.5673)) == pytest.approx(504, abs=5)


def test_score():
    s = score(["Toronto", "Toronto", "Calgary"], ["Toronto", "Oshawa", "Vancouver"], "metro")
    assert s.accuracy == pytest.approx(1 / 3)
    assert s.acc161 == pytest.approx(2 / 3)  # Oshawa is within 161 km of Toronto
    assert s.median_km < 161
    assert np.isfinite(s.mean_km)

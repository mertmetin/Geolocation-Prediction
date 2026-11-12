"""Load the Twitter-CAD datasets.

The datasets are published as MySQL dumps. The 2019 code needed a running
MySQL server with a root password; here a small streaming parser turns the
dumps into Parquet files once (`geoloc prepare`), and everything after that
reads Parquet with pandas.
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .labels import canonical_metro, canonical_province

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"

DROPBOX_URL = "https://www.dropbox.com/sh/a58ac7638cjrnv2/AABmbU-UUPM34kYo2YSjRIe4a?dl=1"
DUMPS = {
    "canada_users_and_tweets.sql": ("users_dataset", ["Users", "Tweets_preprocessed", "Friends", "Followers"]),
    "canada_tweets.sql": ("tweets_dataset", ["Tweets", "Tweets_preprocessed"]),
}

_INSERT_RE = re.compile(r"^INSERT INTO `(\w+)` \((.*)\)\s*$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "Z": "\x1a", "b": "\b"}


# --------------------------------------------------------------------------- download / convert


def download(raw_dir: Path = RAW_DIR) -> None:
    """Fetch the public Dropbox folder (about 75 MB) and unzip the SQL dumps."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    folder_zip = raw_dir / "datasets.zip"
    print(f"Downloading datasets to {raw_dir} ...")
    urllib.request.urlretrieve(DROPBOX_URL, folder_zip)
    with zipfile.ZipFile(folder_zip) as z:
        z.extractall(raw_dir)
    for inner in raw_dir.glob("*.sql.zip"):
        with zipfile.ZipFile(inner) as z:
            z.extractall(raw_dir)
        inner.unlink()
    folder_zip.unlink()


def _parse_tuples(body: str) -> Iterator[list]:
    """Yield rows from a MySQL `VALUES (...),(...);` body.

    Handles quoted strings with backslash escapes, NULL and numbers, and
    commas or parentheses inside strings.
    """
    i, n = 0, len(body)
    while i < n:
        while i < n and body[i] != "(":
            i += 1
        if i >= n:
            return
        i += 1
        row: list = []
        while True:
            while body[i] in " \t\r\n":
                i += 1
            c = body[i]
            if c == "'":
                i += 1
                buf = []
                while True:
                    ch = body[i]
                    if ch == "\\":
                        nxt = body[i + 1]
                        buf.append(_ESCAPES.get(nxt, nxt))
                        i += 2
                    elif ch == "'":
                        if i + 1 < n and body[i + 1] == "'":
                            buf.append("'")
                            i += 2
                        else:
                            i += 1
                            break
                    else:
                        buf.append(ch)
                        i += 1
                row.append("".join(buf))
            else:
                j = i
                while body[j] not in ",)":
                    j += 1
                tok = body[i:j].strip()
                if tok.upper() == "NULL":
                    row.append(None)
                else:
                    try:
                        row.append(int(tok))
                    except ValueError:
                        row.append(float(tok))
                i = j
            while body[i] in " \t\r\n":
                i += 1
            if body[i] == ",":
                i += 1
                continue
            if body[i] == ")":
                i += 1
                break
        yield row


def read_dump(path: Path, tables: list[str]) -> dict[str, pd.DataFrame]:
    """Parse the INSERT statements of a mysqldump/Sequel Pro dump."""
    rows: dict[str, list] = {t: [] for t in tables}
    columns: dict[str, list[str]] = {}
    current: str | None = None
    chunk: list[str] = []

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if current is None:
                m = _INSERT_RE.match(line)
                if m and m.group(1) in rows:
                    current = m.group(1)
                    columns[current] = [c.strip(" `") for c in m.group(2).split(",")]
                    chunk = []
                continue
            chunk.append(line)
            # A statement ends with ");" at end of line. Strings in this dump
            # escape newlines, so a ");" line ending inside a string does not occur.
            if line.rstrip().endswith(");"):
                body = "".join(chunk)
                body = body[body.index("VALUES") + len("VALUES") :] if "VALUES" in body[:20] else body
                rows[current].extend(_parse_tuples(body))
                current = None

    return {t: pd.DataFrame(rows[t], columns=columns.get(t)) for t in tables}


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PARQUET_DIR) -> None:
    """Convert both SQL dumps into Parquet files under data/parquet/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, (prefix, tables) in DUMPS.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `geoloc download` first")
        print(f"Parsing {path} ...")
        for table, df in read_dump(path, tables).items():
            out = out_dir / f"{prefix}.{table.lower()}.parquet"
            df.to_parquet(out, index=False)
            print(f"  {table:20s} {len(df):>9,} rows -> {out}")


# --------------------------------------------------------------------------- loaders


def _read(name: str, data_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `geoloc download && geoloc prepare`")
    return pd.read_parquet(path)


def load_users(data_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    """One row per user: screen_name, metro, province, and all tweets joined.

    Fixes from 2019:
    * The metro filter in `Tweets.get_users` built a WHERE clause but never
      added it to the query (`sql + where` discarded its result), so the
      label set was never actually restricted.
    * Metro spellings are normalised, so Quebec City users are not dropped.
    """
    users = _read("users_dataset.users", data_dir)
    tweets = _read("users_dataset.tweets_preprocessed", data_dir)

    users = users.assign(
        metro=users["metro"].map(canonical_metro),
        province=users["province"].map(canonical_province),
    )
    docs = tweets.dropna(subset=["text"]).groupby("screen_name")["text"].agg(lambda s: "\n".join(s)).rename("text")
    n_tweets = tweets.groupby("screen_name").size().rename("n_tweets")
    df = (
        users[["screen_name", "metro", "province", "gender", "followers_count", "friends_count"]]
        .merge(docs, on="screen_name", how="inner")
        .merge(n_tweets, on="screen_name", how="left")
    )
    return df.dropna(subset=["metro"]).reset_index(drop=True)


def load_geotagged_tweets(data_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    """One row per geotagged tweet from the Twitter-CAD-Tweets dataset."""
    df = _read("tweets_dataset.tweets_preprocessed", data_dir)
    df = df.assign(
        metro=df["metro"].map(canonical_metro),
        province=df["province"].map(canonical_province),
    )
    return df.dropna(subset=["metro", "text"])[["screen_name", "text", "metro", "province"]].reset_index(drop=True)


def load_graph(kind: str, data_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    """Friends or followers of each labelled user, with their declared location.

    Returns columns: of (the labelled user), screen_name, user_declared_location.
    """
    if kind not in ("friends", "followers"):
        raise ValueError(kind)
    df = _read(f"users_dataset.{kind}", data_dir)
    of_col = "friend_of" if kind == "friends" else "follower_of"
    return df.rename(columns={of_col: "of"})[["of", "screen_name", "user_declared_location"]]

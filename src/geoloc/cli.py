"""Command line entry point: `geoloc <command>`."""

from __future__ import annotations

import argparse

import numpy as np


def _users_text(args) -> None:
    from .data import load_users
    from .experiments import run_task
    from .models import TEXT_MODELS
    from .social import metro_to_province

    users = load_users()
    models = _select(TEXT_MODELS, args.models)
    X = users["text"].tolist()
    if not args.province_only:
        run_task("user_text_metro", "metro", X, users["metro"].to_numpy(), models)
    if not args.metro_only:
        y = users["metro"].map(metro_to_province).to_numpy()
        run_task("user_text_province", "province", X, y, models)


def _users_social(args) -> None:
    from .data import load_graph, load_users
    from .experiments import run_task
    from .models import NeighbourVote, social_logreg
    from .social import location_counts

    users = load_users()
    y = users["metro"].to_numpy()
    for kind in ("friends", "followers"):
        X = location_counts(load_graph(kind), users["screen_name"])
        run_task(
            f"user_{kind}_metro",
            "metro",
            X,
            y,
            {
                "vote": ("Majority vote of neighbour locations", NeighbourVote),
                "logreg": ("Logistic regression on neighbour location counts", social_logreg),
            },
        )
    X = location_counts(load_graph("friends"), users["screen_name"]) + location_counts(
        load_graph("followers"), users["screen_name"]
    )
    run_task(
        "user_graph_metro",
        "metro",
        X,
        y,
        {
            "vote": ("Majority vote, friends + followers", NeighbourVote),
            "logreg": ("Logistic regression, friends + followers", social_logreg),
        },
    )


def _tweets_text(args) -> None:
    from .data import load_geotagged_tweets
    from .experiments import run_task
    from .models import TEXT_MODELS

    tweets = load_geotagged_tweets()
    models = _select(TEXT_MODELS, args.models)
    run_task(
        "tweet_text_metro",
        "metro",
        tweets["text"].tolist(),
        tweets["metro"].to_numpy(),
        models,
        groups=tweets["screen_name"].to_numpy(),
    )


def _select(models: dict, names: str | None) -> dict:
    if not names:
        return models
    wanted = names.split(",")
    return {k: v for k, v in models.items() if k in wanted}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoloc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="Download the Twitter-CAD datasets (~75 MB)")
    sub.add_parser("prepare", help="Convert the SQL dumps to Parquet")
    for name in ("users-text", "tweets-text"):
        s = sub.add_parser(name)
        s.add_argument("--models", help="comma-separated subset of models")
        s.add_argument("--metro-only", action="store_true")
        s.add_argument("--province-only", action="store_true")
    sub.add_parser("users-social")
    for name in ("users-embed", "tweets-embed"):
        sub.add_parser(name, help="sentence-embedding classifier (needs the [embeddings] extra)")
    s = sub.add_parser("llm", help="zero-shot LLM baseline (needs the [llm] extra and ANTHROPIC_API_KEY)")
    s.add_argument("--level", choices=["user", "tweet"], default="user")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--model", default=None)
    sub.add_parser("report", help="Write results/README tables and figures")
    sub.add_parser("train-demo", help="Train the small model behind the web demo (app/model.joblib)")
    args = p.parse_args(argv)

    np.seterr(all="ignore")
    if args.cmd == "download":
        from .data import download

        download()
    elif args.cmd == "prepare":
        from .data import prepare

        prepare()
    elif args.cmd == "users-text":
        _users_text(args)
    elif args.cmd == "users-social":
        _users_social(args)
    elif args.cmd == "tweets-text":
        _tweets_text(args)
    elif args.cmd in ("users-embed", "tweets-embed"):
        from .embeddings import run

        run("user" if args.cmd == "users-embed" else "tweet")
    elif args.cmd == "llm":
        from .llm import run

        run(level=args.level, limit=args.limit, model=args.model)
    elif args.cmd == "train-demo":
        import json

        from .demo_model import train
        from .labels import METRO_COORDS

        train()
        with open("app/metros.json", "w") as f:
            json.dump(METRO_COORDS, f, ensure_ascii=False, indent=1)
    elif args.cmd == "report":
        from .report import build

        build()


if __name__ == "__main__":
    main()

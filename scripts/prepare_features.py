"""Build the four-group training and validation CSV from the two published tables."""

import argparse
import csv
from collections import defaultdict
from datetime import date, datetime, timedelta

from scripts.baseline import split_name
from scripts.prepare_data import DEFAULT_OUTPUT, FIELDS, FILENAME, WinHistory, write_csv


FEATURE_FIELDS = (
    "overall_matches_a", "overall_wins_a", "overall_matches_b", "overall_wins_b",
    "overall_rate_diff", "h2h_matches", "h2h_wins_a", "h2h_rate_diff",
)


def build_feature_rows(targets, inventory):
    tour = sorted(
        (row for row in inventory if row["source"] == "tour" and 2006 <= int(row["season"]) <= 2023),
        key=lambda row: row["tourney_date"],
    )
    players = defaultdict(WinHistory)
    pairs = defaultdict(WinHistory)
    target_matches = {}
    for match in tour:
        player_a, player_b = match["player_a_id"], match["player_b_id"]
        if player_a >= player_b or match["a_won"] not in ("0", "1"):
            raise ValueError(f"Invalid player order or label for {match['match_id']}")
        start = datetime.strptime(match["tourney_date"], "%Y%m%d").date()
        a_won = int(match["a_won"])
        players[player_a].record(start, a_won)
        players[player_b].record(start, 1 - a_won)
        pairs[player_a, player_b].record(start, a_won)
        if int(match["season"]) >= 2016:
            key = match["match_id"]
            if key in target_matches:
                raise ValueError(f"Duplicate Tour match {key}")
            target_matches[key] = (player_a, player_b, start, match["season"], match["a_won"])

    for row in targets:
        key = "tour:" + row["match_id"]
        player_a, player_b = row["player_a_id"], row["player_b_id"]
        start = date.fromisoformat(row["tourney_date"])
        identity = (player_a, player_b, start, row["season"], row["a_won"])
        if target_matches.pop(key, None) != identity or row["split"] != split_name(int(row["season"])):
            raise ValueError(f"Baseline row does not match its Tour record: {key}")
        earliest = start - timedelta(days=90)
        latest = start - timedelta(days=21)
        played_a, wins_a, rate_a, _ = players[player_a].summarize(earliest, latest)
        played_b, wins_b, rate_b, _ = players[player_b].summarize(earliest, latest)
        meetings, wins_against_b, h2h_rate, _ = pairs[player_a, player_b].summarize(date.min, latest)
        yield {
            **row,
            "overall_matches_a": played_a, "overall_wins_a": wins_a,
            "overall_matches_b": played_b, "overall_wins_b": wins_b,
            "overall_rate_diff": rate_a - rate_b,
            "h2h_matches": meetings, "h2h_wins_a": wins_against_b,
            "h2h_rate_diff": 2 * h2h_rate - 1,
        }
    if target_matches:
        raise ValueError(f"Baseline table is missing {len(target_matches)} eligible Tour matches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with (DEFAULT_OUTPUT / FILENAME).open(encoding="utf-8", newline="") as baseline, (
        DEFAULT_OUTPUT / "mens_singles_2006_2025.csv"
    ).open(encoding="utf-8", newline="") as inventory:
        rows = build_feature_rows(csv.DictReader(baseline), csv.DictReader(inventory))
        destination = write_csv(rows, FIELDS + FEATURE_FIELDS, DEFAULT_OUTPUT / "model_features.csv")
    print(f"Wrote four-group training and validation table to {destination}")


if __name__ == "__main__":
    main()

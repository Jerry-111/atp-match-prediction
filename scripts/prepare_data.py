"""Prepare the training and validation rows for the ATP prediction project."""

import argparse
import csv
import math
import sys
import tempfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from scripts.baseline import DEFAULT_CACHE, FIRST_SEASONS, ROOT, load_matches, split_name


DEFAULT_OUTPUT = ROOT / "data" / "processed"
FILENAME = "model_input.csv"
FIELDS = (
    "match_id", "season", "split", "tourney_date", "tourney_id", "match_num",
    "round", "surface", "player_a_id", "player_b_id", "player_a_name",
    "player_b_name", "rank_a", "rank_b", "a_won",
    "surface_matches_a", "surface_wins_a", "surface_rate_a", "latest_history_date_a",
    "surface_matches_b", "surface_wins_b", "surface_rate_b", "latest_history_date_b",
    "log_rank_diff", "surface_rate_diff",
)


class SurfaceHistory:
    """One player's cumulative wins on one surface, ordered by tournament start."""

    def __init__(self):
        self.dates = []
        self.cumulative_wins = [0]

    def record(self, date, won):
        self.dates.append(date)
        self.cumulative_wins.append(self.cumulative_wins[-1] + int(won))

    def summarize(self, earliest, latest):
        start = bisect_left(self.dates, earliest)
        end = bisect_right(self.dates, latest)
        played = end - start
        wins = self.cumulative_wins[end] - self.cumulative_wins[start]
        last_date = self.dates[end - 1].isoformat() if played else ""
        return played, wins, (wins + 1) / (played + 2), last_date


def build_model_rows(matches, first_season=2015):
    """Build model rows using the selected first season only for warmup history."""
    eligible = (m for m in matches if first_season <= m["season"] <= 2023)
    ordered = sorted(eligible, key=lambda m: (m["tourney_date"], m["tourney_id"], m["match_num"]))
    histories = defaultdict(SurfaceHistory)
    rows = []
    for match in ordered:
        date = match["tourney_date"]
        surface = match["surface"]
        winner = match["winner_id"]
        loser = match["loser_id"]
        split = split_name(match["season"], first_season)
        if split != "warmup":
            a_won = winner < loser
            source_a, source_b = ("winner", "loser") if a_won else ("loser", "winner")
            row = {
                "match_id": f"{match['tourney_id']}:{match['match_num']}",
                "season": match["season"],
                "split": split,
                "tourney_date": date.isoformat(),
                "tourney_id": match["tourney_id"],
                "match_num": match["match_num"],
                "round": match["round"],
                "surface": surface,
                "a_won": int(a_won),
            }
            earliest = date - timedelta(weeks=52)
            latest = date - timedelta(days=21)
            for side, source in (("a", source_a), ("b", source_b)):
                player_id = match[f"{source}_id"]
                row[f"player_{side}_id"] = player_id
                row[f"player_{side}_name"] = match[f"{source}_name"]
                row[f"rank_{side}"] = match[f"{source}_rank"]
                history = histories[(player_id, surface)]
                played, wins, rate, last_date = history.summarize(earliest, latest)
                row[f"surface_matches_{side}"] = played
                row[f"surface_wins_{side}"] = wins
                row[f"surface_rate_{side}"] = rate
                row[f"latest_history_date_{side}"] = last_date
            row["log_rank_diff"] = math.log(row["rank_b"]) - math.log(row["rank_a"])
            row["surface_rate_diff"] = row["surface_rate_a"] - row["surface_rate_b"]
            rows.append(row)

        # Record the result after predicting. Other matches at this tournament
        # share this start date and also fall inside the 21-day exclusion.
        histories[(winner, surface)].record(date, True)
        histories[(loser, surface)].record(date, False)
    return rows


def write_model_csv(rows, output_dir=DEFAULT_OUTPUT, first_season=2015):
    """Replace the local modeling CSV only after the entire file was written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = FILENAME if first_season == 2015 else f"model_input_{first_season}.csv"
    destination = output_dir / filename
    output = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output_dir, delete=False
    )
    temporary = Path(output.name)
    try:
        with output:
            writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--first-season", type=int, choices=FIRST_SEASONS, default=2015,
        help="First season of the cohort, reserved for initializing player history (default: 2015)."
    )
    args = parser.parse_args()
    matches, audit = load_matches(args.cache_dir, args.first_season)
    rows = build_model_rows(matches, args.first_season)
    path = write_model_csv(rows, args.output_dir, args.first_season)
    if args.first_season != 2015:
        raw = sum(counts["raw"] for counts in audit.values())
        warmup = audit[args.first_season]["eligible"]
        test = audit[2024]["eligible"] + audit[2025]["eligible"]
        print(f"Cohort {args.first_season}–2025: {raw:,} raw, {len(matches):,} eligible.")
        print(f"Eligible warmup: {warmup:,}. Reserved test: {test:,}.")
    train_count = sum(row["split"] == "train" for row in rows)
    print(f"Wrote {len(rows):,} rows to {path}")
    print(f"Train: {train_count:,}; validation: {len(rows) - train_count:,}.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Data preparation failed: {error}")

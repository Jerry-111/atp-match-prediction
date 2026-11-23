"""Count reported match endings and evaluate the ranking rule on 2023 only."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from baseline import DEFAULT_CACHE, MANIFEST, SCORE_MARKERS, exclusion_reason, fetch_csv, ranking_result


SET_SCORE = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")
OUTCOMES = ("completed", "RET", "W/O", "DEF", "other")


def score_ending(row):
    """Return the category and observed form, retaining unrecognized scores for review."""
    score = row["score"].strip()
    tokens = score.split()
    if not tokens:
        return "other", "(blank)"
    marker = tokens[-1]
    if marker.upper() in SCORE_MARKERS:
        form = f"score + {marker}" if len(tokens) > 1 else marker
        return marker.upper(), form

    sets = [SET_SCORE.fullmatch(token) for token in tokens]
    if all(sets) and row["best_of"] in {"3", "5"}:
        required = int(row["best_of"]) // 2 + 1
        first_won = [int(s.group(1)) > int(s.group(2)) for s in sets]
        if first_won[-1] and sum(first_won) == required and len(sets) <= int(row["best_of"]):
            return "completed", "set scores"
    return "other", score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    by_stage = {stage: Counter() for stage in ("Before ranking filter", "With valid rankings", "2023 with valid rankings")}
    observed_forms = Counter()
    by_surface = {surface: Counter() for surface in ("Hard", "Clay", "Grass")}
    validation = {"completed": [], "RET": []}
    raw_count = 0
    for entry in entries:
        path = fetch_csv(args.cache_dir, entry)
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                raw_count += 1
                reason = exclusion_reason(row)
                if reason in {"level", "surface", "player_id"}:
                    continue
                ending, form = score_ending(row)
                by_stage["Before ranking filter"][ending] += 1
                observed_forms[form] += 1
                if reason == "ranking":
                    continue
                by_stage["With valid rankings"][ending] += 1
                by_surface[row["surface"]][ending] += 1
                if entry["season"] == 2023:
                    by_stage["2023 with valid rankings"][ending] += 1
                    if ending in validation:
                        validation[ending].append({key: int(row[key]) for key in ("winner_rank", "loser_rank")})

    print(f"Raw ATP rows: {raw_count:,}")
    print("Scope                             total   completed     RET     W/O     DEF   other")
    for stage, counts in by_stage.items():
        columns = " ".join(f"{counts[ending]:>7,}" for ending in OUTCOMES)
        print(f"{stage:30} {sum(counts.values()):>8,} {columns}")
    print("\nScore forms before the ranking filter")
    for form, count in observed_forms.most_common():
        print(f"  {form!r}: {count:,}")
    print("\nRET / all records with valid rankings")
    for surface, counts in by_surface.items():
        total = sum(counts.values())
        print(f"  {surface}: {counts['RET']:,}/{total:,} ({counts['RET'] / total:.2%})")

    print("\n2023 ranking rule, scored against the recorded winner")
    completed = validation["completed"]
    retirements = validation["RET"]
    accuracy = {}
    for label, matches in (("Completed", completed), ("RET", retirements),
                           ("Completed + RET", completed + retirements)):
        correct, scored = ranking_result(matches)
        if not scored:
            raise ValueError(f"No rankable 2023 matches in {label}")
        accuracy[label] = correct / scored
        print(f"  {label}: {correct:,}/{scored:,} ({accuracy[label]:.2%}), {len(matches) - scored} tied")
    change = 100 * (accuracy["Completed + RET"] - accuracy["Completed"])
    print(f"  Change from completed only: {change:+.2f} percentage points")


if __name__ == "__main__":
    main()

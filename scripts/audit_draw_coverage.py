"""Count ATP main draw, Challenger main draw and ATP qualifying coverage by source."""

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from scripts.baseline import (
    DEFAULT_CACHE, HISTORY_MANIFEST, LEVELS, MANIFEST, ROOT, exclusion_reason, fetch_csv, split_name,
)


SUPPLEMENTAL = ROOT / "draw-coverage-manifest.json"
SEASONS = range(2011, 2026)
SOURCES = ("tour", "challenger", "qualifying")
COUNT_FIELDS = ("raw_matches", "raw_event_editions", "eligible_matches", "eligible_event_editions")
DEFAULT_INVENTORY = ROOT / "data" / "processed" / "mens_singles_2011_2025.csv"
INVENTORY_FIELDS = (
    "match_id", "source", "season", "split", "tourney_id", "tourney_name", "tourney_level",
    "stage", "tourney_date", "surface", "round", "match_num", "player_a_id", "player_b_id",
    "player_a_name", "player_b_name", "rank_a", "rank_b", "a_won",
)
# These knockout matches each have two source rows identical except for match_num.
# Keep the smaller match_num. Preserve all other source rows, including round-robin rematches.
DUPLICATE_RECORDS = {
    "tour:2021-416:25": "tour:2021-416:14",
    "tour:2021-321:13": "tour:2021-321:5",
    "qualifying:2011-533:14": "qualifying:2011-533:13",
}
# The official Båstad draw gives Robredo a first-round bye. The source copies
# the Paire–Robredo final into R32 as #1. Paire's actual R32 is already #4.
# https://www.protennislive.com/posting/2015/316/mds.pdf
BASTAD_SPURIOUS = "tour:2015-316:1"
BASTAD_FINAL = "tour:2015-316:28"
BASTAD_R32 = "tour:2015-316:4"
CHECK_IDS = set(DUPLICATE_RECORDS) | set(DUPLICATE_RECORDS.values()) | {
    BASTAD_SPURIOUS, BASTAD_FINAL, BASTAD_R32,
}


def source_entries():
    tour = []
    for manifest in (HISTORY_MANIFEST, MANIFEST):
        tour.extend(json.loads(manifest.read_text(encoding="utf-8"))["files"])
    sources = json.loads(SUPPLEMENTAL.read_text(encoding="utf-8"))["series"]
    sources["tour"] = [entry for entry in tour if entry["season"] in SEASONS]
    for name in SOURCES:
        if [entry["season"] for entry in sources[name]] != list(SEASONS):
            raise ValueError(f"The {name} manifest must pin each season from 2011 through 2025 once.")
    return sources


class Inventory:
    def __init__(self, file):
        self.writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        self.writer.writeheader()
        self.seen = set()
        self.check_rows = {}
        self.written = 0

    def write(self, row, source, season):
        match_id = f"{source}:{row['tourney_id']}:{int(row['match_num'])}"
        if match_id in self.seen:
            raise ValueError(f"Duplicate inventory ID {match_id}.")
        self.seen.add(match_id)
        if match_id in CHECK_IDS:
            self.check_rows[match_id] = {k: v for k, v in row.items() if k != "match_num"}
        if match_id in DUPLICATE_RECORDS or match_id == BASTAD_SPURIOUS:
            return
        role_a, role_b = sorted(("winner", "loser"), key=lambda role: row[f"{role}_id"])
        split = split_name(season, first_season=2011)
        record = {k: row[k] for k in ("tourney_id", "tourney_name", "tourney_level",
                                      "tourney_date", "surface", "round", "match_num")}
        record.update(match_id=match_id, source=source, season=season, split=split,
                      stage="qualifying" if source == "qualifying" else "main",
                      a_won="" if split == "test" else int(role_a == "winner"))
        for side, role in (("a", role_a), ("b", role_b)):
            record[f"player_{side}_id"] = row[f"{role}_id"]
            record[f"player_{side}_name"] = row[f"{role}_name"]
            record[f"rank_{side}"] = row[f"{role}_rank"]
        self.writer.writerow(record)
        self.written += 1

    def verify_exclusions(self):
        for duplicate, retained in DUPLICATE_RECORDS.items():
            twin = self.check_rows.get(duplicate)
            if not twin or twin["round"] == "RR" or twin != self.check_rows.get(retained):
                raise ValueError(f"Source rows {duplicate} and {retained} no longer verify as duplicates.")
        expected_draw = {
            BASTAD_SPURIOUS: ("R32", "PD31", "R419", "7-6(7) 6-3"),
            BASTAD_FINAL: ("F", "PD31", "R419", "7-6(7) 6-3"),
            BASTAD_R32: ("R32", "PD31", "E688", "7-6(4) 6-3"),
        }
        for match_id, expected in expected_draw.items():
            row = self.check_rows.get(match_id, {})
            actual = tuple(row.get(k) for k in ("round", "winner_id", "loser_id", "score"))
            if row.get("tourney_id") != "2015-316" or actual != expected:
                raise ValueError(f"Recheck {match_id} against the official 2015 Båstad draw.")


@contextmanager
def open_inventory(path):
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False
    )
    temporary = Path(file.name)
    try:
        with file:
            inventory = Inventory(file)
            yield inventory
            inventory.verify_exclusions()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_file(path, source, season, inventory=None):
    events = set()
    eligible_events = set()
    matches = set()
    eligible = 0
    dates = []
    levels = {"C"} if source == "challenger" else LEVELS
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event = row["tourney_id"]
            key = (event, int(row["match_num"]))
            if not event or key in matches:
                raise ValueError(f"Missing tournament ID or duplicate match {key} in {path.name}.")
            matches.add(key)
            events.add(event)
            dates.append(row["tourney_date"])
            if exclusion_reason(row, levels=levels) is None:
                eligible += 1
                eligible_events.add(event)
                if inventory is not None:
                    inventory.write(row, source, season)
    if not matches:
        raise ValueError(f"No matches found in {path.name}.")
    counts = Counter(raw_matches=len(matches), raw_event_editions=len(events),
                     eligible_matches=eligible, eligible_event_editions=len(eligible_events))
    return counts, (min(dates), max(dates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help="Raw data root. Challenger and qualifying files use separate subfolders.")
    parser.add_argument("--annual", action="store_true", help="Also print every source-season count.")
    parser.add_argument(
        "--export-inventory", nargs="?", type=Path, const=DEFAULT_INVENTORY, metavar="PATH",
        help="Write the local eligible singles inventory. The default path is under data/processed/.",
    )
    args = parser.parse_args()
    sources = source_entries()
    annual = {}
    dates = {}
    with open_inventory(args.export_inventory) as inventory:
        for name in SOURCES:
            cache = args.cache_dir if name == "tour" else args.cache_dir / name
            cache.mkdir(parents=True, exist_ok=True)
            for entry in sources[name]:
                key = (name, entry["season"])
                path = fetch_csv(cache, entry)
                annual[key], dates[key] = audit_file(path, name, entry["season"], inventory)

    report = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    report.writerow(("period", "source", *COUNT_FIELDS))
    for first in (2011, 2016):
        for name in SOURCES:
            total = Counter()
            for year in range(first, 2026):
                total.update(annual[name, year])
            report.writerow((f"{first}-2025", name, *(total[field] for field in COUNT_FIELDS)))
    if args.annual:
        for (name, year), counts in annual.items():
            report.writerow((year, name, *(counts[field] for field in COUNT_FIELDS)))
    print()
    report.writerow(("season", "source", *COUNT_FIELDS, "first_tourney_date", "last_tourney_date"))
    for year in (2019, 2024):
        counts = annual["challenger", year]
        report.writerow((year, "challenger", *(counts[k] for k in COUNT_FIELDS),
                         *dates["challenger", year]))
    if inventory is not None:
        print(f"Inventory: {len(inventory.seen):,} eligible source records, "
              f"{len(DUPLICATE_RECORDS)} duplicate records and 1 spurious first-round record removed, "
              f"{inventory.written:,} distinct matches.")
        print(f"Local CSV: {args.export_inventory}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Coverage audit failed: {error}")

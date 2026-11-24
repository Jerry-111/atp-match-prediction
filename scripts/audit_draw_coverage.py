"""Count ATP main draw, Challenger main draw and ATP qualifying coverage by source."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.baseline import (
    DEFAULT_CACHE, HISTORY_MANIFEST, LEVELS, MANIFEST, ROOT, exclusion_reason, fetch_csv,
)


SUPPLEMENTAL = ROOT / "draw-coverage-manifest.json"
SEASONS = range(2011, 2026)
SOURCES = ("tour", "challenger", "qualifying")
COUNT_FIELDS = ("raw_matches", "raw_event_editions", "eligible_matches", "eligible_event_editions")


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


def audit_file(path, levels):
    events = set()
    eligible_events = set()
    matches = set()
    eligible = 0
    dates = []
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
    args = parser.parse_args()
    sources = source_entries()
    annual = {}
    dates = {}
    for name in SOURCES:
        cache = args.cache_dir if name == "tour" else args.cache_dir / name
        cache.mkdir(parents=True, exist_ok=True)
        levels = {"C"} if name == "challenger" else LEVELS
        for entry in sources[name]:
            key = (name, entry["season"])
            path = fetch_csv(cache, entry)
            annual[key], dates[key] = audit_file(path, levels)

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


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Coverage audit failed: {error}")

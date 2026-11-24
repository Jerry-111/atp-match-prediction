"""Resolve missing 2010 qualifying tier and surface from matching ATP main draws."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.baseline import DEFAULT_CACHE, HISTORY_MANIFEST, LEVELS, SURFACES, fetch_csv


# Editions whose pinned qualifying file omits both event fields.
MISSING_METADATA = {
    "2010-315", "2010-316", "2010-321", "2010-341", "2010-414", "2010-418",
    "2010-422", "2010-439", "2010-540", "2010-773", "2010-1720",
}


def load_2010_event_metadata(cache_dir=DEFAULT_CACHE):
    """Return unique (name, start date, tier, surface) tuples from the pinned main draw."""
    files = json.loads(HISTORY_MANIFEST.read_text(encoding="utf-8"))["files"]
    entry = next(entry for entry in files if entry["season"] == 2010)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    metadata = defaultdict(set)
    with fetch_csv(cache, entry).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            event = row["tourney_id"]
            if event in MISSING_METADATA:
                metadata[event].add((row["tourney_name"], row["tourney_date"],
                                     row["tourney_level"], row["surface"]))
    return {event: next(iter(values)) for event, values in metadata.items() if len(values) == 1}


def effective_qualifying_row(row, season, metadata):
    """Return an effective row and source reference, preserving every reported field in row."""
    if season != 2010 or row["tourney_level"] or row["surface"]:
        return row, ""
    event = row["tourney_id"]
    if event not in MISSING_METADATA or event not in metadata:
        return row, ""
    name, date, tier, surface = metadata[event]
    if (row["tourney_name"], row["tourney_date"]) != (name, date):
        return row, ""
    if tier not in LEVELS or surface not in SURFACES:
        return row, ""
    return dict(row, tourney_level=tier, surface=surface), f"tour:{event}"

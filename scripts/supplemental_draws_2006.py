"""Recover four 2006 major qualifying draws and whole missing Challenger editions.

Source: Jeff Sackmann's tennis_atp, CC BY-NC-SA 4.0, archived by Aneeshers.
https://github.com/Aneeshers/tennis-sackmann-archive/tree/83733587353df8a41f2fd4f516147d5aa83f5a8d
The archive's date is the main-draw tournament week, including for qualifying.
The two original Sackmann player IDs remain on each returned row for traceability.
"""

import csv
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

from scripts.baseline import HISTORY_MANIFEST, exclusion_reason, fetch_csv


ARCHIVE = {
    "season": 2006,
    "url": (
        "https://cdn.jsdelivr.net/gh/Aneeshers/tennis-sackmann-archive@"
        "83733587353df8a41f2fd4f516147d5aa83f5a8d/atp/atp_matches_qual_chall_2006.csv"
    ),
    "sha256": "9c904151837cf669851f8983b03e24c4838db84eadec55afaf3328e7f4a2eb33",
}
MAJORS = {"2006-580", "2006-520", "2006-540", "2006-560"}
EXPECTED_ROUNDS = {
    "qualifying": {"Q1": 64, "Q2": 32, "Q3": 16},
    "challenger": {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1},
}
ROLES = ("winner", "loser")


def normalized_name(name):
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(char for char in decomposed if char.isalnum())


def player_observations(rows):
    for row in rows:
        day = datetime.strptime(row["tourney_date"], "%Y%m%d").date().toordinal()
        for role in ROLES:
            age = row[f"{role}_age"]
            birth = day - round(float(age) * 365.25) if age else None
            yield row[f"{role}_id"], normalized_name(row[f"{role}_name"]), birth


def reference_data(cache_dir, sources):
    """Index verifiable identities and record primary coverage before filling gaps."""
    by_name = defaultdict(lambda: defaultdict(set))
    challenger_editions = set()
    challenger_finals = set()
    # Two qualifiers last appeared in these reference files in the 2005 tour.
    historical = json.loads(HISTORY_MANIFEST.read_text(encoding="utf-8"))["files"]
    series = {
        "tour": [e for e in historical if 2005 <= e["season"] <= 2010],
        "challenger": [e for e in sources["challenger"] if 2006 <= e["season"] <= 2010],
        "qualifying": [e for e in sources["qualifying"] if 2007 <= e["season"] <= 2010],
    }
    for source, entries in series.items():
        cache = cache_dir if source == "tour" else cache_dir / source
        cache.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            with fetch_csv(cache, entry).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if source == "challenger" and entry["season"] == 2006:
                challenger_editions.update(row["tourney_id"] for row in rows)
                challenger_finals.update(
                    tuple(normalized_name(row[f"{role}_name"]) for role in ROLES)
                    for row in rows if row["round"] == "F"
                )
            for player_id, name, birth in player_observations(rows):
                if player_id and name and birth is not None:
                    by_name[name][player_id].add(birth)
    return by_name, challenger_editions, challenger_finals


def crosswalk_players(rows, by_name):
    """Require a unique normalized full name and compatible identity dates.

    The archived ages have one decimal place (about 37 days of resolution) and
    the sources may label different days of a tournament week. Allow 45 days.
    Countries can change between seasons; they are not identity keys.
    """
    observations = defaultdict(set)
    for native_id, name, birth in player_observations(rows):
        observations[native_id].add((name, birth))

    mapped = {}
    for native_id, observed in observations.items():
        candidates = set()
        for name, birth in observed:
            if birth is None:
                continue
            for candidate, reference_births in by_name.get(name, {}).items():
                if any(abs(birth - ref) <= 45 for ref in reference_births):
                    candidates.add(candidate)
        if len(candidates) == 1:
            mapped[native_id] = candidates.pop()

    counts = Counter(mapped.values())
    return {native: atp for native, atp in mapped.items() if counts[atp] == 1}


def supplemental_draws_2006(cache_dir, sources):
    """Return qualifying/Challenger source rows and crosswalk/coverage counts.

    Unresolved IDs are blank so the shared eligibility filter omits those
    matches while the raw coverage audit can still count the archived record.
    """
    archive_cache = cache_dir / "sackmann"
    archive_cache.mkdir(parents=True, exist_ok=True)
    with fetch_csv(archive_cache, ARCHIVE).open(encoding="utf-8-sig", newline="") as handle:
        archive = list(csv.DictReader(handle))

    by_name, known_editions, known_finals = reference_data(cache_dir, sources)
    qualifying = [
        r for r in archive if r["tourney_level"] == "G" and r["round"] in EXPECTED_ROUNDS["qualifying"]
    ]
    challengers = [
        r for r in archive
        if r["tourney_level"] == "C" and r["round"] in EXPECTED_ROUNDS["challenger"]
        and r["tourney_id"] not in known_editions
    ]

    if {row["tourney_id"] for row in qualifying} != MAJORS:
        raise ValueError("Recheck the four major qualifying draws in the 2006 archive.")
    grouped = {"qualifying": qualifying, "challenger": challengers}
    for source, rows in grouped.items():
        by_event = defaultdict(Counter)
        for row in rows:
            by_event[row["tourney_id"]][row["round"]] += 1
        for event, rounds in by_event.items():
            if rounds != EXPECTED_ROUNDS[source]:
                raise ValueError(f"Unexpected {source} bracket for {event}: {dict(rounds)}.")
    # An event ID can differ across vendors. Also check the champion and
    # runner-up in the primary file before adding a whole challenger edition.
    for row in challengers:
        if row["round"] == "F":
            final = tuple(normalized_name(row[f"{role}_name"]) for role in ROLES)
            if final in known_finals:
                raise ValueError(f"Check possible overlap for {row['tourney_id']}: {final}.")

    all_rows = qualifying + challengers
    by_id = crosswalk_players(all_rows, by_name)
    player_names = {r[f"{role}_id"]: r[f"{role}_name"] for r in all_rows for role in ROLES}
    if "" in player_names:
        raise ValueError("A 2006 archive record has no native player ID.")
    for row in all_rows:
        for role in ROLES:
            native = row[f"{role}_id"]
            row[f"{role}_sackmann_id"] = native
            row[f"{role}_id"] = by_id.get(native, "")

    report = {
        "total_players": len(player_names),
        "mapped_players": len(by_id),
        "unmapped_players": {pid: player_names[pid] for pid in sorted(player_names.keys() - by_id.keys())},
        "groups": {},
    }
    for source, rows in grouped.items():
        levels = {"G"} if source == "qualifying" else {"C"}
        report["groups"][source] = {
            "raw_rows": len(rows),
            "raw_editions": len({r["tourney_id"] for r in rows}),
            "identified_rows": sum(bool(r["winner_id"] and r["loser_id"]) for r in rows),
            "eligible_rows": sum(exclusion_reason(r, levels) is None for r in rows),
        }
    return grouped, report

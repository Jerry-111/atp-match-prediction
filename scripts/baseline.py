"""Fetch the pinned ATP data and score the higher-ranking rule on validation."""

import argparse
import csv
import hashlib
import json
import shutil
import ssl
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "raw"
MANIFEST = ROOT / "data-manifest.json"
HISTORY_MANIFEST = ROOT / "history-manifest.json"
FIRST_SEASONS = (1990, 2000, 2006, 2009, 2010, 2011, 2015, 2016)
LEVELS = {"G", "M", "500", "250"}
SURFACES = {"Hard", "Clay", "Grass"}
SCORE_MARKERS = {"W/O": "walkover", "RET": "retirement", "DEF": "default"}


def split_name(season, first_season=2015):
    if first_season not in FIRST_SEASONS:
        raise ValueError(f"Unsupported first season {first_season}; choose one of {FIRST_SEASONS}.")
    if season == first_season:
        return "warmup"
    if first_season < season <= 2022:
        return "train"
    if season == 2023:
        return "validation"
    if season in (2024, 2025):
        return "test"
    raise ValueError(f"Season {season} is outside the project split.")


def verify_file(path, entry, from_cache):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        if from_cache:
            action = f"Move {path} out of the cache and retry with a fresh download."
        else:
            action = "The source may have changed. Review the new snapshot with the team before updating the manifest."
        raise ValueError(
            f"{entry['season']} CSV checksum mismatch. Expected {entry['sha256']}, "
            f"received {actual}. {action}"
        )


def fetch_csv(cache_dir, entry):
    path = cache_dir / f"{entry['season']}.csv"
    if path.exists():
        verify_file(path, entry, from_cache=True)
        return path

    request = Request(entry["url"], headers={"User-Agent": "ATP-course-project/1.0"})
    context = ssl.create_default_context()
    # Some python.org macOS installations ship without a configured CA bundle.
    system_bundle = Path("/etc/ssl/cert.pem")
    if sys.platform == "darwin" and not context.get_ca_certs() and system_bundle.is_file():
        context.load_verify_locations(cafile=str(system_bundle))
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        with urlopen(request, context=context, timeout=45) as response:
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        verify_file(temporary_path, entry, from_cache=False)
        temporary_path.replace(path)
    except OSError as error:
        raise OSError(f"Cannot obtain {entry['url']}: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def exclusion_reason(row, levels=LEVELS):
    """Assign each excluded row to the first filter it fails."""
    if row["tourney_level"] not in levels:
        return "level"
    if row["surface"] not in SURFACES:
        return "surface"
    if not row["winner_id"] or not row["loser_id"] or row["winner_id"] == row["loser_id"]:
        return "player_id"
    for key in ("winner_rank", "loser_rank"):
        if not row[key].isdigit() or int(row[key]) <= 0:
            return "ranking"
    if not row["score"].strip() or any(character.isalpha() for character in row["score"]):
        return "score"
    return None


def load_matches(cache_dir=DEFAULT_CACHE, first_season=2015):
    """Return eligible matches and first-failure counts by season.

    The score count includes its walkover, retirement, default and other_score
    subtotals. Match records retain outcomes for downstream supervised analysis.
    """
    if first_season not in FIRST_SEASONS:
        raise ValueError(f"Unsupported first season {first_season}; choose one of {FIRST_SEASONS}.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    if first_season < 2015:
        historical = json.loads(HISTORY_MANIFEST.read_text(encoding="utf-8"))["files"]
        entries = [entry for entry in historical if entry["season"] >= first_season] + entries
    entries = [entry for entry in entries if entry["season"] >= first_season]
    if [entry["season"] for entry in entries] != list(range(first_season, 2026)):
        raise ValueError(f"The manifests must cover each season from {first_season} to 2025 once, in order.")
    matches = []
    audit = {}
    seen = set()
    for entry in entries:
        season = entry["season"]
        counts = Counter()
        with fetch_csv(cache_dir, entry).open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                counts["raw"] += 1
                reason = exclusion_reason(row)
                if reason:
                    counts[reason] += 1
                    if reason == "score":
                        status = row["score"].strip().upper().rpartition(" ")[-1]
                        counts[SCORE_MARKERS.get(status, "other_score")] += 1
                    continue

                match_key = (row["tourney_id"], int(row["match_num"]))
                if match_key in seen:
                    raise ValueError(f"Duplicate eligible match {match_key} in {season}.")
                seen.add(match_key)
                match = {
                    key: row[key]
                    for key in ("tourney_id", "surface", "round", "winner_id",
                                "winner_name", "loser_id", "loser_name")
                }
                match.update(
                    season=season,
                    tourney_date=datetime.strptime(row["tourney_date"], "%Y%m%d").date(),
                    match_num=match_key[1],
                    winner_rank=int(row["winner_rank"]),
                    loser_rank=int(row["loser_rank"]),
                )
                matches.append(match)
                counts["eligible"] += 1
        audit[season] = counts
    return matches, audit


def ranking_result(matches):
    """Return (correct, scored) for the higher-ranked player; abstain on equal ranks."""
    correct = 0
    scored = 0
    for match in matches:
        winner_rank = match["winner_rank"]
        loser_rank = match["loser_rank"]
        if winner_rank != loser_rank:
            scored += 1
            correct += winner_rank < loser_rank
    return correct, scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    matches, audit = load_matches(args.cache_dir)
    counts = Counter(split_name(match["season"]) for match in matches)
    print(f"Raw matches: {sum(season['raw'] for season in audit.values()):,}")
    for split in ("warmup", "train", "validation", "test"):
        print(f"{split.capitalize()} eligible: {counts[split]:,}")
    correct, scored = ranking_result(match for match in matches if match["season"] == 2023)
    if not scored:
        raise ValueError("No 2023 validation matches with distinct ranks.")
    print(f"2023 higher-ranked player: {correct:,}/{scored:,} correct "
          f"({correct / scored:.2%}); equal ranks excluded: {counts['validation'] - scored}.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Baseline failed: {error}")

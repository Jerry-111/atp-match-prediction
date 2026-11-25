"""Pinned, past-only serve/return summaries for training and 2023 validation.

Both players' service statistics must be valid for a match to contribute.
The published eligible inventory supplies the population, dates and identities;
raw annual files supply only technical counts. Files for 2024/2025 are never
downloaded or opened. Target/future statistics cannot enter T-364..T-21 and
T-728..T-21 inclusive windows. Every ``_diff`` column is odd under player
exchange; all other columns are even. Fixed 100-point priors are .65 on serve
and .35 on return, declared before observing validation performance.
"""

import csv
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from math import isfinite, log1p
from pathlib import Path

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE, HISTORY_MANIFEST, MANIFEST, fetch_csv
from scripts.model_event_metadata import corrected_tour_row


FEATURE_VERSION = "1.0"
WINDOWS = (364, 728)
LAG = 21
PRIOR_POINTS = 100.0
SERVE_PRIOR = .65
RETURN_PRIOR = .35


def development_entries():
    """Filter manifests before calling fetch_csv; never return test seasons."""
    entries = []
    for path in (HISTORY_MANIFEST, MANIFEST):
        entries.extend(json.loads(path.read_text(encoding="utf-8"))["files"])
    entries = sorted((e for e in entries if 2006 <= int(e["season"]) <= 2023),
                     key=lambda e: int(e["season"]))
    if [int(e["season"]) for e in entries] != list(range(2006, 2024)):
        raise ValueError("Expected one pinned annual file per season 2006..2023")
    return entries


def _counts(raw):
    """Return paired (service wins, opportunities), or reject the whole pair."""
    values = []
    for prefix in ("w", "l"):
        try:
            first, second, total = (float(raw[f"{prefix}_{name}"])
                                    for name in ("1stWon", "2ndWon", "svpt"))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(isfinite(v) and v >= 0 and v.is_integer() for v in (first, second, total)):
            return None
        if total <= 0 or first + second > total:
            return None
        values.append((first + second, total))
    return values


def _annual_rows(cache_dir, entries):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        with fetch_csv(cache_dir, entry).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield int(entry["season"]), row


@dataclass
class PointHistory:
    dates: list = field(default_factory=list)
    cumulative: list = field(default_factory=lambda: [(0.,) * 6])

    def record(self, day, counts):
        self.dates.append(day)
        self.cumulative.append(tuple(x + y for x, y in zip(self.cumulative[-1], counts)))

    def summary(self, earliest, latest):
        low, high = bisect_left(self.dates, earliest), bisect_right(self.dates, latest)
        values = tuple(x - y for x, y in zip(self.cumulative[high], self.cumulative[low]))
        sw, sn, rw, rn, valid, missing = values
        return {
            "serve_rate": (sw + SERVE_PRIOR * PRIOR_POINTS) / (sn + PRIOR_POINTS),
            "return_rate": (rw + RETURN_PRIOR * PRIOR_POINTS) / (rn + PRIOR_POINTS),
            "serve_points": sn, "return_points": rn,
            "valid_matches": valid, "missing_matches": missing,
            "unseen": float(valid == 0),
        }


def build_serve_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None,
                         first_target_season=2016):
    """Return a numeric frame with the same row index as ``targets``.

    ``raw_rows`` is an optional iterable of (season, raw row) for offline unit
    tests. Production loads and SHA-256 verifies the pinned 2006..2023 files.
    Provenance and missing-statistics counts are stored in ``result.attrs``.
    """
    if first_target_season not in range(2009, 2017):
        raise ValueError("first_target_season must be an integer from 2009 through 2016")
    if not pd.to_numeric(targets["season"]).between(first_target_season, 2023).all():
        raise ValueError(f"Final holdout is locked: targets must be from {first_target_season}..2023")
    eligible = inventory.loc[inventory["source"].eq("tour") &
                             pd.to_numeric(inventory["season"]).between(2006, 2023)]
    expected = {}
    for row in eligible.to_dict("records"):
        key = str(row["match_id"])
        if key in expected:
            raise ValueError(f"Duplicate inventory match: {key}")
        expected[key] = row
    entries = development_entries() if raw_rows is None else []
    rows = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    matched = {}
    for season, raw in rows:
        if not 2006 <= int(season) <= 2023:
            continue  # Do not inspect any test-season fields, even in fixtures.
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in matched:
            raise ValueError(f"Duplicate technical-statistics match: {key}")
        raw = corrected_tour_row(raw)
        match = expected[key]
        a, b = str(match["player_a_id"]), str(match["player_b_id"])
        winner, loser = str(raw["winner_id"]), str(raw["loser_id"])
        if ({a, b} != {winner, loser} or a == b or
                _date(raw["tourney_date"]) != _date(match["tourney_date"]) or
                raw["surface"] != match["surface"] or int(match["season"]) != int(season)):
            raise ValueError(f"Technical statistics do not match inventory identity: {key}")
        stats = _counts(raw)
        # Labels from raw and inventory are never used as a numeric predictor.
        if stats is None:
            pair = {a: (0., 0., 0., 0., 0., 1.), b: (0., 0., 0., 0., 0., 1.)}
        else:
            (ww, wn), (lw, ln) = stats
            pair = {winner: (ww, wn, ln - lw, ln, 1., 0.),
                    loser: (lw, ln, wn - ww, wn, 1., 0.)}
        matched[key] = (_date(match["tourney_date"]), key, str(match["surface"]), pair)
    absent = expected.keys() - matched.keys()
    if absent:
        raise ValueError(f"Technical source omitted {len(absent)} eligible matches; first: {sorted(absent)[0]}")

    overall, surface_history = defaultdict(PointHistory), defaultdict(PointHistory)
    valid = 0
    for day, _, surface, pair in sorted(matched.values()):
        valid += int(next(iter(pair.values()))[4])
        for player, stats in pair.items():
            overall[player].record(day, stats)
            surface_history[player, surface].record(day, stats)

    features = []
    for target in targets.to_dict("records"):
        day = _date(target["tourney_date"])
        a, b = str(target["player_a_id"]), str(target["player_b_id"])
        if a == b:
            raise ValueError("A match must have two different players")
        surface = str(target["surface"])
        result = {}
        for window in WINDOWS:
            for scope, histories in (("overall", (overall[a], overall[b])),
                                     ("surface", (surface_history[a, surface], surface_history[b, surface]))):
                first, second = [h.summary(day - timedelta(days=window), day - timedelta(days=LAG))
                                 for h in histories]
                prefix = f"points_{scope}_{window}"
                for rate in ("serve_rate", "return_rate"):
                    result[f"{prefix}_{rate}_diff"] = first[rate] - second[rate]
                    result[f"{prefix}_{rate}_sum"] = first[rate] + second[rate]
                for count in ("serve_points", "return_points", "valid_matches", "missing_matches"):
                    result[f"{prefix}_{count}_log_diff"] = log1p(first[count]) - log1p(second[count])
                    result[f"{prefix}_{count}_log_sum"] = log1p(first[count]) + log1p(second[count])
                    result[f"{prefix}_{count}_min"] = min(first[count], second[count])
                result[f"{prefix}_unseen_sum"] = first["unseen"] + second["unseen"]
        features.append(result)
    frame = pd.DataFrame(features, index=targets.index, dtype=float)
    frame.attrs.update({"feature_version": FEATURE_VERSION, "history_lag_days": LAG,
                        "windows_days": list(WINDOWS), "prior_points": PRIOR_POINTS,
                        "serve_prior": SERVE_PRIOR, "return_prior": RETURN_PRIOR,
                        "eligible_history_matches": len(matched), "valid_history_matches": valid,
                        "missing_or_invalid_history_matches": len(matched) - valid,
                        "source_files": entries, "injected_raw_rows": raw_rows is not None})
    return frame

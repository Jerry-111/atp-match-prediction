"""Matched Elo banks with additional official senior singles history.

This changes historical event coverage, never the prediction target cohort.
Extra events require an explicit event-name/level allowlist and conventional
completed best-of-three/five scores. Queries retain the inclusive T-21 lag.
All *_diff features are odd under player exchange; all others are even.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from itertools import groupby
from math import fsum, isfinite, log1p

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE, exclusion_reason
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import (
    RatingHistory, _player, _season_date, elo_expectation, historical_records,
    surface_blend,
)
from scripts.paper_margin_features import experience_k, score_weights
from scripts.serve_features import _annual_rows, development_entries

FEATURE_VERSION = "extended-official-history-1.0"
LAG_DAYS = 21
INITIAL_RATING = 1500.0
EXTRA_EVENT_NAMES = {
    "A": frozenset({"world team championship", "atp cup", "united cup"}),
    "F": frozenset({"tour finals", "masters cup", "atp finals"}),
    "O": frozenset({"beijing olympics", "london olympics", "rio olympics", "tokyo olympics"}),
}


def extra_event_reason(raw):
    """Return None only for explicitly accepted conventional senior matches.

    D is a source classification, not sufficient evidence by itself. Davis
    Cup must also have the expected event name. Dead rubbers remain eligible
    if actually played to a conventional completed score; their incentive
    differences are an acknowledged limitation, not an inferred exclusion.
    """
    level = str(raw.get("tourney_level", ""))
    name = " ".join(str(raw.get("tourney_name", "")).casefold().split())
    accepted = ((level == "D" and name.startswith("davis cup ")) or
                name in EXTRA_EVENT_NAMES.get(level, ()))
    if not accepted:
        return "event_not_allowlisted"
    reason = exclusion_reason(raw, {level})
    if reason:
        return f"source_{reason}"
    try:
        score_weights(raw["score"], raw["best_of"])
    except (KeyError, TypeError, ValueError):
        return "nonstandard_or_incomplete_score"
    return None


def target_queries(targets):
    """Validate development targets before inspecting outcomes or raw files."""
    if targets.empty or not pd.to_numeric(targets.season, errors="coerce").isin(range(2009, 2024)).all():
        raise ValueError("Targets must be nonempty 2009--2023; final holdout is locked")
    queries = []
    for row in targets.to_dict("records"):
        day, season = _date(row["tourney_date"]), int(row["season"])
        _season_date(day, season)
        a, b = _player(row["player_a_id"]), _player(row["player_b_id"])
        surface = str(row["surface"])
        if a == b or surface not in ("Hard", "Clay", "Grass"):
            raise ValueError("Invalid target players or surface")
        queries.append((day, a, b, surface))
    return queries


def collect_extended_history(inventory, latest, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Join published Tour controls and audit extra raw histories before use."""
    tour = historical_records(inventory.loc[inventory.source.eq("tour")], latest)
    expected = {r[1]: r for r in tour}
    entries = development_entries() if raw_rows is None else []
    source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    seen, extras, exclusions, levels, names = set(), [], Counter(), Counter(), Counter()
    raw_tour_seen = set()
    for season, raw in source:
        if not 2006 <= int(season) <= 2023:
            continue  # Do not read held-out fields, identities or outcomes.
        raw = corrected_tour_row(raw)
        day = _date(raw["tourney_date"])
        _season_date(day, int(season))
        if day > latest:
            continue
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key in expected:
            if key in raw_tour_seen:
                raise ValueError(f"Duplicate raw Tour history: {key}")
            raw_tour_seen.add(key)
            _, _, _, a, b, won, surface = expected[key]
            winner, loser = _player(raw["winner_id"]), _player(raw["loser_id"])
            if (day != expected[key][0] or {a, b} != {winner, loser} or
                    str(raw["surface"]) != surface or float(winner == a) != won):
                raise ValueError(f"Raw/published Tour identity/date/outcome mismatch: {key}")
            continue
        reason = extra_event_reason(raw)
        if reason:
            exclusions[reason] += 1
            continue
        if key in seen:
            raise ValueError(f"Duplicate extra historical source ID: {key}")
        seen.add(key)
        winner, loser = _player(raw["winner_id"]), _player(raw["loser_id"])
        extras.append((day, key, "extra", winner, loser, 1., str(raw["surface"])))
        levels[str(raw["tourney_level"])] += 1
        names[str(raw["tourney_name"])] += 1
    if expected.keys()-raw_tour_seen:
        raise ValueError("Pinned raw source omitted published Tour history")
    metadata = {
        "tour_history_matches": len(tour), "extra_history_matches": len(extras),
        "extra_matches_by_level": dict(sorted(levels.items())),
        "extra_matches_by_event": dict(sorted(names.items())),
        "excluded_raw_rows_by_reason": dict(sorted(exclusions.items())),
        "source_files": entries, "injected_raw_rows": raw_rows is not None,
        "event_policy": "explicit senior event name + source level + standard completed score",
    }
    return sorted(tour+extras), metadata


class OfficialRatingBank:
    """Ordinary Elo with pre-date states, simultaneous updates and snapshots."""

    def __init__(self, records, *, extended, adaptive):
        self.overall, self.surface = defaultdict(RatingHistory), defaultdict(RatingHistory)
        for day, grouped in groupby(records, key=lambda r: r[0]):
            batch = [r for r in grouped if extended or r[2] == "tour"]
            overall_delta, surface_delta = defaultdict(list), defaultdict(list)
            overall_n, surface_n = Counter(), Counter()
            for _, _, _, a, b, won, surface in batch:
                oa = self.overall[a].at(day)
                ob = self.overall[b].at(day)
                # A new surface starts from already observed overall strength,
                # never from an end-of-day or future player estimate.
                sa = self.surface[a, surface].at(day, default=oa[0])
                sb = self.surface[b, surface].at(day, default=ob[0])
                for histories, counts, states, keys in (
                    (overall_delta, overall_n, (oa, ob), (a, b)),
                    (surface_delta, surface_n, (sa, sb), ((a, surface), (b, surface))),
                ):
                    expected = elo_expectation(states[0][0], states[1][0])
                    for i, (key, outcome) in enumerate(zip(keys, (won, 1.-won))):
                        k = experience_k(states[i][1]) if adaptive else 32.
                        histories[key].append(k * (outcome-(expected if i == 0 else 1.-expected)))
                        counts[key] += 1
            # Compute every new snapshot before committing either bank. This
            # also preserves pre-date overall initialization for new surfaces.
            pending = []
            for key, changes in surface_delta.items():
                default = self.overall[key[0]].at(day)[0]
                rating, count = self.surface[key].at(day, default=default)
                pending.append((self.surface[key], rating+fsum(changes), count+surface_n[key]))
            for key, changes in overall_delta.items():
                rating, count = self.overall[key].at(day)
                pending.append((self.overall[key], rating+fsum(changes), count+overall_n[key]))
            for history, rating, count in pending:
                history.append(day, rating, count)

    def at(self, player, surface, cutoff):
        overall, count = self.overall[player].at(cutoff)
        surface_rating, n_surface = self.surface[player, surface].at(cutoff, default=overall)
        blended, _ = surface_blend(overall, surface_rating, n_surface)
        return {"overall": overall, "surface": surface_rating, "blend": blended, "count": count}


def build_extended_history_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Return 20 finite columns, with identical target index and row order."""
    queries = target_queries(targets)
    latest = max(q[0] for q in queries)-timedelta(days=LAG_DAYS)
    records, metadata = collect_extended_history(inventory, latest, cache_dir=cache_dir, raw_rows=raw_rows)
    banks = {f"{coverage}_{kind}": OfficialRatingBank(records, extended=coverage == "extended", adaptive=kind == "adaptive")
             for coverage in ("tour", "extended") for kind in ("k32", "adaptive")}
    output = []
    for day, a, b, surface in queries:
        cutoff = day-timedelta(days=LAG_DAYS)
        values = {name: [bank.at(p, surface, cutoff) for p in (a, b)] for name, bank in banks.items()}
        row = {f"official_{name}_{scope}_diff": pair[0][scope]-pair[1][scope]
               for name, pair in values.items() for scope in ("overall", "surface", "blend")}
        for kind in ("k32", "adaptive"):
            for scope in ("overall", "blend"):
                row[f"official_added_{kind}_{scope}_diff"] = (
                    row[f"official_extended_{kind}_{scope}_diff"]-row[f"official_tour_{kind}_{scope}_diff"])
        extra = [values["extended_k32"][i]["count"]-values["tour_k32"][i]["count"] for i in (0, 1)]
        row["official_extra_exposure_log_diff"] = log1p(extra[0])-log1p(extra[1])
        row["official_extra_exposure_log_sum"] = log1p(extra[0])+log1p(extra[1])
        row["official_extra_exposure_min"] = float(min(extra))
        row["official_extra_unseen_sum"] = float(sum(n == 0 for n in extra))
        output.append(row)
    frame = pd.DataFrame(output, index=targets.index, dtype=float)
    if len(frame.columns) != 20 or not all(isfinite(v) for v in frame.to_numpy().flat):
        raise ValueError("Unexpected official-history feature schema/non-finite values")
    frame.attrs.update(metadata, feature_version=FEATURE_VERSION, history_lag_days=LAG_DAYS,
                       initial_rating=INITIAL_RATING, fixed_k=32., surface_shrink_matches=20.,
                       adaptive_k="250/(max(1, pre-date appearances)+5)^0.4",
                       odd_columns=[c for c in frame if c.endswith("_diff")],
                       even_columns=[c for c in frame if not c.endswith("_diff")], final_test_evaluated=False)
    return frame

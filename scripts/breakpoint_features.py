"""Past-only break-point, hold/break and point-to-game residual features.

These are descriptive historical rates, not identified psychological clutch
ability. Ordinary-point IID hold probabilities provide a fixed reference,
with explicit sample shrinkage and exposure features. No target stats enter.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from math import isfinite, log1p

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.extended_history_features import target_queries
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _player, _season_date
from scripts.serve_features import _annual_rows, development_entries

FEATURE_VERSION = "breakpoint-game-residual-1.0"
LAG_DAYS = 21
WINDOW_DAYS = 364
BP_PRIOR_OPPORTUNITIES = 20.
GAME_PRIOR_OPPORTUNITIES = 20.
POINT_PRIOR_OPPORTUNITIES = 100.
SERVE_POINT_PRIOR = .65


def iid_hold_probability(p):
    """Exact conventional advantage-game probability for IID serve points."""
    if not isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Serve point probability must lie in [0, 1]")
    q = 1.-p
    return p**4*(1.+4.*q+10.*q*q) + 20.*p**3*q**3*p*p/(p*p+q*q)


def breakpoint_pair(raw):
    """Return each player's 12 rate counts, or reject the complete pair.

    A converted break point ends exactly one service game. Consequently
    holds = service games - (break points faced - saved). Reject impossible
    marginals instead of clipping them to plausible rates.
    """
    values = []
    for side in ("w", "l"):
        try:
            saved, faced, games, points, first, second = [
                float(raw[f"{side}_{c}"]) for c in ("bpSaved", "bpFaced", "SvGms", "svpt", "1stWon", "2ndWon")]
        except (KeyError, TypeError, ValueError):
            return None
        row = (saved, faced, games, points, first, second)
        if not all(isfinite(v) and v >= 0 and v.is_integer() for v in row):
            return None
        lost = faced-saved
        won = first+second
        if (games <= 0 or points <= 0 or not 0 <= lost <= games or won > points
                or faced > points or saved > won or lost > points-won):
            return None
        values.append((saved, faced, games-lost, games, first+second, points))
    result = []
    for own, opponent in ((values[0], values[1]), (values[1], values[0])):
        saved, faced, holds, games, won, points = own
        osaved, ofaced, oholds, ogames, owon, opoints = opponent
        result.append((saved, faced, ofaced-osaved, ofaced, holds, games,
                       ogames-oholds, ogames, won, points, opoints-owon, opoints))
    return result


@dataclass
class BreakpointHistory:
    days: list = field(default_factory=list)
    cumulative: list = field(default_factory=lambda: [(0.,)*14])

    def append(self, day, vector):
        self.days.append(day)
        self.cumulative.append(tuple(a+b for a, b in zip(self.cumulative[-1], vector)))

    def summary(self, first, last):
        low, high = bisect_left(self.days, first), bisect_right(self.days, last)
        counts = [b-a for a, b in zip(self.cumulative[low], self.cumulative[high])]
        hold_prior = iid_hold_probability(SERVE_POINT_PRIOR)
        priors = (SERVE_POINT_PRIOR, 1.-SERVE_POINT_PRIOR, hold_prior, 1.-hold_prior,
                  SERVE_POINT_PRIOR, 1.-SERVE_POINT_PRIOR)
        sizes = (BP_PRIOR_OPPORTUNITIES,)*2+(GAME_PRIOR_OPPORTUNITIES,)*2+(POINT_PRIOR_OPPORTUNITIES,)*2
        rates = [(counts[2*i]+prior*size)/(counts[2*i+1]+size)
                 for i, (prior, size) in enumerate(zip(priors, sizes))]
        return {"save": rates[0], "convert": rates[1], "hold": rates[2], "break": rates[3],
                "hold_residual": rates[2]-iid_hold_probability(rates[4]),
                "break_residual": rates[3]-(1.-iid_hold_probability(1.-rates[5])),
                "exposure": counts[1]+counts[3], "valid": counts[12], "missing": counts[13]}


def build_breakpoint_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Return 20 finite numeric columns from eligible past Tour observations."""
    queries = target_queries(targets)
    latest = max(q[0] for q in queries)-timedelta(days=LAG_DAYS)
    mask = inventory.source.eq("tour") & pd.to_numeric(inventory.season, errors="coerce").isin(range(2006, 2024))
    expected = {}
    for row in inventory.loc[mask].to_dict("records"):
        day = _date(row["tourney_date"])
        _season_date(day, int(row["season"]))
        if day > latest:
            continue
        key = str(row["match_id"])
        if key in expected:
            raise ValueError(f"Duplicate BP history: {key}")
        expected[key] = row
    entries = development_entries() if raw_rows is None else []
    source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    matched, valid = {}, 0
    for season, raw in source:
        if not 2006 <= int(season) <= 2023:
            continue
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in matched:
            raise ValueError(f"Duplicate raw BP history: {key}")
        raw, match = corrected_tour_row(raw), expected[key]
        a, b = _player(match["player_a_id"]), _player(match["player_b_id"])
        winner, loser = _player(raw["winner_id"]), _player(raw["loser_id"])
        if ({a, b} != {winner, loser} or a == b or int(match["season"]) != int(season)
                or _date(raw["tourney_date"]) != _date(match["tourney_date"])
                or str(raw["surface"]) != str(match["surface"])):
            raise ValueError(f"Raw BP identity/date/surface mismatch: {key}")
        pair = breakpoint_pair(raw)
        valid += pair is not None
        vectors = {p: (0.,)*12+(0., 1.) for p in (winner, loser)} if pair is None else {
            winner: tuple(pair[0])+(1., 0.), loser: tuple(pair[1])+(1., 0.)}
        matched[key] = (_date(match["tourney_date"]), key, str(match["surface"]), vectors)
    if expected.keys()-matched.keys():
        raise ValueError("Raw BP source omitted eligible Tour matches")
    histories = defaultdict(BreakpointHistory)
    for day, _, surface, pair in sorted(matched.values(), key=lambda r: (r[0], r[1])):
        for player, vector in pair.items():
            histories[player, "overall"].append(day, vector)
            histories[player, surface].append(day, vector)
    output = []
    for day, a, b, surface in queries:
        first, last = day-timedelta(days=WINDOW_DAYS), day-timedelta(days=LAG_DAYS)
        row = {}
        for label, scope in (("overall", "overall"), ("surface", surface)):
            pair = [histories[p, scope].summary(first, last) for p in (a, b)]
            for metric in ("save", "convert", "hold", "break", "hold_residual", "break_residual"):
                row[f"bp_{label}_{metric}_diff"] = pair[0][metric]-pair[1][metric]
            row[f"bp_{label}_exposure_log_diff"] = log1p(pair[0]["exposure"])-log1p(pair[1]["exposure"])
            for metric in ("exposure", "valid", "missing"):
                row[f"bp_{label}_{metric}_log_sum"] = log1p(pair[0][metric])+log1p(pair[1][metric])
        output.append(row)
    frame = pd.DataFrame(output, index=targets.index, dtype=float)
    if len(frame.columns) != 20 or not all(isfinite(v) for v in frame.to_numpy().flat):
        raise ValueError("Unexpected BP feature schema/non-finite values")
    frame.attrs.update(feature_version=FEATURE_VERSION, history_lag_days=LAG_DAYS,
                       window_days=WINDOW_DAYS, serve_point_prior=SERVE_POINT_PRIOR,
                       bp_prior_opportunities=BP_PRIOR_OPPORTUNITIES,
                       game_prior_opportunities=GAME_PRIOR_OPPORTUNITIES,
                       point_prior_opportunities=POINT_PRIOR_OPPORTUNITIES,
                       eligible_history_matches=len(matched), valid_history_matches=valid,
                       invalid_history_matches=len(matched)-valid, source_files=entries,
                       injected_raw_rows=raw_rows is not None,
                       odd_columns=[c for c in frame if c.endswith("_diff")],
                       even_columns=[c for c in frame if not c.endswith("_diff")], final_test_evaluated=False)
    return frame

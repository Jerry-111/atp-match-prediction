"""Compact serve composition and matchup features using only T-21 history.

Technical counts and height/hand attributes come from pinned 2006--2023 Tour
raw files, joined to eligible historical matches. Even player attributes use
the last unambiguous historical snapshot, never the target match's raw row.
All *_diff outputs are odd under player exchange; other outputs are even.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import groupby
from math import isfinite, log1p

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _player, _season_date
from scripts.serve_features import _annual_rows, development_entries

FEATURE_VERSION = "style-composition-1.0"
LAG_DAYS = 21
WINDOW_DAYS = 364
PRIOR_OPPORTUNITIES = 100.0
HEIGHT_PRIOR = 185.0
RATE_PRIORS = {"ace": .06, "double_fault": .08, "first_in": .60,
               "first_won": .70, "second_won": .50,
               "return_first_won": .30, "return_second_won": .50}


def technical_pair(raw):
    """Return two 14-count numerator/denominator vectors, or reject the pair.

    DF opportunities are all second-serve attempts, including double faults.
    Opponent first/second serve counts define the respective return rates.
    """
    values = []
    for side in ("w", "l"):
        try:
            row = [float(raw[f"{side}_{c}"]) for c in ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon")]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(isfinite(x) and x >= 0 and x.is_integer() for x in row):
            return None
        ace, df, total, first_in, first_won, second_won = row
        second = total - first_in
        if (total <= 0 or first_in > total or first_won > first_in or second_won > second
                or df + second_won > second or ace > first_won + second_won):
            return None
        values.append((ace, df, total, first_in, first_won, second_won))
    result = []
    for own, opponent in ((values[0], values[1]), (values[1], values[0])):
        ace, df, total, first_in, first_won, second_won = own
        _, _, ototal, ofirst, ofirst_won, osecond_won = opponent
        second, osecond = total - first_in, ototal - ofirst
        result.append((ace, total, df, second, first_in, total, first_won, first_in,
                       second_won, second, ofirst-ofirst_won, ofirst,
                       osecond-osecond_won, osecond))
    return result


def player_attributes(raw, prefix):
    """Read plausible height/known hand only, without outcome-dependent coding."""
    try:
        height = float(raw.get(f"{prefix}_ht", ""))
        if not isfinite(height) or not 130 <= height <= 230:
            height = None
    except (TypeError, ValueError):
        height = None
    hand = str(raw.get(f"{prefix}_hand", "")).upper()
    return height, hand if hand in ("L", "R") else None


@dataclass
class StyleHistory:
    days: list = field(default_factory=list)
    cumulative: list = field(default_factory=lambda: [(0.,)*16])

    def append(self, day, vector):
        self.days.append(day)
        self.cumulative.append(tuple(a+b for a, b in zip(self.cumulative[-1], vector)))

    def summary(self, first, last):
        low, high = bisect_left(self.days, first), bisect_right(self.days, last)
        values = [b-a for a, b in zip(self.cumulative[low], self.cumulative[high])]
        rates = {name: (values[2*i] + prior*PRIOR_OPPORTUNITIES) /
                 (values[2*i+1] + PRIOR_OPPORTUNITIES)
                 for i, (name, prior) in enumerate(RATE_PRIORS.items())}
        rates["exposure"] = values[5] + values[11] + values[13]
        rates["missing"] = values[15]
        return rates


@dataclass
class AttributeHistory:
    days: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def append(self, day, observations):
        previous = self.values[-1] if self.values else (None, None)
        current, conflicts = [], 0
        for i in (0, 1):
            candidates = {v[i] for v in observations if v[i] is not None}
            conflicts += len(candidates) > 1
            # Missing or conflicting same-day metadata cannot erase previously
            # known information or choose a winner-dependent last record.
            current.append(next(iter(candidates)) if len(candidates) == 1 else previous[i])
        self.days.append(day)
        self.values.append(tuple(current))
        return conflicts

    def at(self, cutoff):
        position = bisect_right(self.days, cutoff)-1
        height, hand = self.values[position] if position >= 0 else (None, None)
        return {"height": height if height is not None else HEIGHT_PRIOR,
                "height_missing": float(height is None), "left": float(hand == "L"),
                "right": float(hand == "R"), "hand_missing": float(hand is None)}


def build_style_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Return thirty finite numeric features, preserving target index/order."""
    if targets.empty or not pd.to_numeric(targets.season, errors="coerce").isin(range(2009, 2024)).all():
        raise ValueError("Style targets must be nonempty 2009--2023; final holdout is locked")
    queries = []
    for row in targets.to_dict("records"):
        day = _date(row["tourney_date"])
        _season_date(day, int(row["season"]))
        a, b = _player(row["player_a_id"]), _player(row["player_b_id"])
        if a == b:
            raise ValueError("Style target players must differ")
        queries.append((day, a, b))
    latest = max(day for day, _, _ in queries)-timedelta(days=LAG_DAYS)
    mask = inventory.source.eq("tour") & pd.to_numeric(inventory.season, errors="coerce").isin(range(2006, 2024))
    expected = {}
    for row in inventory.loc[mask].to_dict("records"):
        day = _date(row["tourney_date"])
        _season_date(day, int(row["season"]))
        if day > latest:
            continue
        identifier = str(row["match_id"])
        if identifier in expected:
            raise ValueError(f"Duplicate style history: {identifier}")
        expected[identifier] = row
    entries = development_entries() if raw_rows is None else []
    source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    matched, valid = {}, 0
    for season, raw in source:
        if not 2006 <= int(season) <= 2023:
            continue  # Held-out fixture fields cannot even be inspected.
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue  # Includes every target's own/future technical statistics.
        if key in matched:
            raise ValueError(f"Duplicate raw style history: {key}")
        raw, match = corrected_tour_row(raw), expected[key]
        a, b = _player(match["player_a_id"]), _player(match["player_b_id"])
        winner, loser = _player(raw["winner_id"]), _player(raw["loser_id"])
        if ({a, b} != {winner, loser} or a == b or int(match["season"]) != int(season)
                or _date(raw["tourney_date"]) != _date(match["tourney_date"])
                or str(raw["surface"]) != str(match["surface"])):
            raise ValueError(f"Raw style identity/date/surface mismatch: {key}")
        counts = technical_pair(raw)
        valid += counts is not None
        vectors = {p: ((0.,)*14+(0., 1.)) for p in (winner, loser)} if counts is None else {
            winner: tuple(counts[0])+(1., 0.), loser: tuple(counts[1])+(1., 0.)}
        attributes = {winner: player_attributes(raw, "winner"), loser: player_attributes(raw, "loser")}
        matched[key] = (_date(match["tourney_date"]), key, vectors, attributes)
    if expected.keys()-matched.keys():
        raise ValueError("Raw style source omitted eligible historical matches")
    histories, attributes = defaultdict(StyleHistory), defaultdict(AttributeHistory)
    conflicts = 0
    for day, batch in groupby(sorted(matched.values(), key=lambda r: (r[0], r[1])), key=lambda r: r[0]):
        observed = defaultdict(list)
        for _, _, pair, metadata in batch:
            for player, vector in pair.items():
                histories[player].append(day, vector)
                observed[player].append(metadata[player])
        for player, observations in observed.items():
            conflicts += attributes[player].append(day, observations)
    rows = []
    for day, a, b in queries:
        cutoff, first = day-timedelta(days=LAG_DAYS), day-timedelta(days=WINDOW_DAYS)
        pair = [histories[p].summary(first, cutoff) for p in (a, b)]
        meta = [attributes[p].at(cutoff) for p in (a, b)]
        row = {}
        for name in RATE_PRIORS:
            row[f"style_{name}_diff"] = pair[0][name]-pair[1][name]
            row[f"style_{name}_sum"] = pair[0][name]+pair[1][name]
        for name in ("exposure", "missing"):
            row[f"style_{name}_log_diff"] = log1p(pair[0][name])-log1p(pair[1][name])
            row[f"style_{name}_log_sum"] = log1p(pair[0][name])+log1p(pair[1][name])
        row["style_height_diff"] = meta[0]["height"]-meta[1]["height"]
        row["style_height_mean"] = (meta[0]["height"]+meta[1]["height"])/2
        for name in ("height_missing", "hand_missing"):
            row[f"style_{name}_diff"] = meta[0][name]-meta[1][name]
            row[f"style_{name}_sum"] = meta[0][name]+meta[1][name]
        row["style_left_diff"] = meta[0]["left"]-meta[1]["left"]
        row["style_mixed_hands"] = meta[0]["left"]*meta[1]["right"] + meta[1]["left"]*meta[0]["right"]
        row["style_height_ace_diff"] = ((meta[0]["height"]-HEIGHT_PRIOR)*pair[0]["ace"] -
                                        (meta[1]["height"]-HEIGHT_PRIOR)*pair[1]["ace"])/10
        returns = [(r["return_first_won"]+r["return_second_won"])/2 for r in pair]
        row["style_left_opponent_return_diff"] = meta[0]["left"]*returns[1]-meta[1]["left"]*returns[0]
        for name in ("first", "second"):
            row[f"style_{name}_matchup_diff"] = (
                pair[0][f"{name}_won"]*(1-pair[1][f"return_{name}_won"]) -
                pair[1][f"{name}_won"]*(1-pair[0][f"return_{name}_won"]))
        rows.append(row)
    frame = pd.DataFrame(rows, index=targets.index, dtype=float)
    if len(frame.columns) != 30 or not all(isfinite(x) for x in frame.to_numpy().flat):
        raise ValueError("Unexpected style schema or non-finite values")
    frame.attrs.update(feature_version=FEATURE_VERSION, history_lag_days=LAG_DAYS,
                       window_days=WINDOW_DAYS, prior_opportunities=PRIOR_OPPORTUNITIES,
                       rate_priors=dict(RATE_PRIORS), height_prior=HEIGHT_PRIOR,
                       attribute_policy="last unambiguous historical record through T-21",
                       eligible_history_matches=len(matched), valid_history_matches=valid,
                       invalid_history_matches=len(matched)-valid,
                       conflicting_same_day_attributes=conflicts, source_files=entries,
                       injected_raw_rows=raw_rows is not None,
                       odd_columns=[c for c in frame if c.endswith("_diff")],
                       even_columns=[c for c in frame if not c.endswith("_diff")],
                       final_test_evaluated=False)
    return frame

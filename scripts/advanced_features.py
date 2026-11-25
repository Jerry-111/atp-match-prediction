"""Past-only numeric features for the extended training/validation experiment.

All historical results are completed, ranked Tour/major main-draw matches in
the published inventory. Dates are tournament start dates, not match dates.
Queries include history through T-21 days. Elo updates sharing a date are
computed from the ratings at the start of that date and applied together; no
invented within-tournament ordering is used. All Elo variants start at 1500.

Columns ending ``_diff`` change sign when A and B are exchanged. Every other
column is invariant. No player identifier, target outcome, tournament ID or
round enters the numeric matrix. Hyperparameter/feature selection is the
caller's responsibility and must use training-period forward validation.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import groupby
from math import log, log1p

import pandas as pd


FEATURE_VERSION = "1.0"
ELO_K = (16, 32, 64)
WINDOWS = (90, 364, 728)
HISTORY_LAG_DAYS = 21
ELO_INITIAL = 1500.0
SURFACE_PSEUDO_MATCHES = 20.0


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    return datetime.strptime(text, "%Y%m%d").date() if len(text) == 8 else date.fromisoformat(text[:10])


def elo_probability(rating_difference):
    """Base-10, 400-point-scale Elo probability of A winning."""
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def odd_feature_columns(columns):
    """Return names that change sign under player exchange."""
    return [name for name in columns if name.endswith("_diff")]


def mirror_features(frame):
    """Exchange A/B in this module's features without mutating the input."""
    result = frame.copy()
    names = odd_feature_columns(result.columns)
    result[names] = -result[names]
    return result


@dataclass
class ResultHistory:
    dates: list = field(default_factory=list)
    wins: list = field(default_factory=lambda: [0.0])
    residuals: list = field(default_factory=lambda: [0.0])

    def record(self, day, won, residual=0.0):
        self.dates.append(day)
        self.wins.append(self.wins[-1] + won)
        self.residuals.append(self.residuals[-1] + residual)

    def summary(self, earliest, latest):
        start, end = bisect_left(self.dates, earliest), bisect_right(self.dates, latest)
        return end - start, self.wins[end] - self.wins[start], self.residuals[end] - self.residuals[start]

    def last_date(self, latest):
        end = bisect_right(self.dates, latest)
        return self.dates[end - 1] if end else None


@dataclass
class RatingHistory:
    dates: list = field(default_factory=list)
    ratings: list = field(default_factory=list)

    def record(self, day, rating):
        self.dates.append(day)
        self.ratings.append(rating)

    def at(self, latest):
        end = bisect_right(self.dates, latest)
        return self.ratings[end - 1] if end else ELO_INITIAL


class HistoricalState:
    """Build immutable, cutoff-queryable histories from eligible past seasons."""

    def __init__(self, inventory):
        # Filter seasons and sources BEFORE touching outcomes, IDs or dates.
        # Held-out outcomes may be missing or deliberately invalid sentinels.
        eligible = inventory.loc[
            inventory["source"].eq("tour") & pd.to_numeric(inventory["season"]).between(2006, 2023)
        ]
        records = []
        seen = set()
        for raw in eligible.to_dict("records"):
            identifier = str(raw["match_id"])
            if identifier in seen:
                raise ValueError(f"Duplicate history match: {identifier}")
            seen.add(identifier)
            a, b = str(raw["player_a_id"]), str(raw["player_b_id"])
            label = float(raw["a_won"])
            if a == b or label not in (0.0, 1.0):
                raise ValueError(f"Invalid player or outcome: {identifier}")
            surface = str(raw["surface"])
            if surface not in ("Hard", "Clay", "Grass"):
                raise ValueError(f"Unexpected Tour surface: {surface}")
            records.append((_date(raw["tourney_date"]), identifier, a, b, label, surface))
        records.sort()
        self.overall = defaultdict(ResultHistory)
        self.surface = defaultdict(ResultHistory)
        self.pair = defaultdict(ResultHistory)
        self.ratings = {k: defaultdict(RatingHistory) for k in ELO_K}
        self.surface_ratings = {k: defaultdict(RatingHistory) for k in ELO_K}
        current = {k: defaultdict(lambda: ELO_INITIAL) for k in ELO_K}
        surface_current = {k: defaultdict(lambda: ELO_INITIAL) for k in ELO_K}
        for day, day_rows in groupby(records, key=lambda row: row[0]):
            deltas = {k: defaultdict(float) for k in ELO_K}
            surface_deltas = {k: defaultdict(float) for k in ELO_K}
            for _, _, a, b, won, surface in day_rows:
                residual = won - elo_probability(current[32][a] - current[32][b])
                self.overall[a].record(day, won, residual)
                self.overall[b].record(day, 1 - won, -residual)
                self.surface[a, surface].record(day, won, residual)
                self.surface[b, surface].record(day, 1 - won, -residual)
                low, high = sorted((a, b))
                self.pair[low, high].record(day, won if a == low else 1 - won)
                for k in ELO_K:
                    change = k * (won - elo_probability(current[k][a] - current[k][b]))
                    deltas[k][a] += change
                    deltas[k][b] -= change
                    change_surface = k * (won - elo_probability(
                        surface_current[k][a, surface] - surface_current[k][b, surface]))
                    surface_deltas[k][a, surface] += change_surface
                    surface_deltas[k][b, surface] -= change_surface
            for k in ELO_K:
                for player, delta in deltas[k].items():
                    current[k][player] += delta
                    self.ratings[k][player].record(day, current[k][player])
                for key, delta in surface_deltas[k].items():
                    surface_current[k][key] += delta
                    self.surface_ratings[k][key].record(day, surface_current[k][key])

    def features(self, target):
        day = _date(target["tourney_date"])
        cutoff = day - timedelta(days=HISTORY_LAG_DAYS)
        a, b, surface = str(target["player_a_id"]), str(target["player_b_id"]), str(target["surface"])
        if a == b:
            raise ValueError("A match must have two different players")
        result = {}
        all_count = [self.overall[p].summary(date.min, cutoff)[0] for p in (a, b)]
        surf_count = [self.surface[p, surface].summary(date.min, cutoff)[0] for p in (a, b)]
        for label, counts in (("career", all_count), ("surface_career", surf_count)):
            result[f"{label}_log_count_diff"] = log1p(counts[0]) - log1p(counts[1])
            result[f"{label}_log_count_sum"] = log1p(counts[0]) + log1p(counts[1])
            result[f"{label}_min_count"] = min(counts)
        for k in ELO_K:
            overall = [self.ratings[k][p].at(cutoff) for p in (a, b)]
            on_surface = [self.surface_ratings[k][p, surface].at(cutoff) for p in (a, b)]
            shrunk = [
                overall[i] + surf_count[i] / (surf_count[i] + SURFACE_PSEUDO_MATCHES)
                * (on_surface[i] - overall[i]) for i in range(2)
            ]
            result[f"elo_k{k}_diff"] = overall[0] - overall[1]
            result[f"surface_elo_k{k}_diff"] = on_surface[0] - on_surface[1]
            result[f"shrunk_surface_elo_k{k}_diff"] = shrunk[0] - shrunk[1]
            # Fixed 50/50 blend differs from evidence-dependent shrinkage.
            result[f"blend_elo_k{k}_diff"] = .5 * (overall[0] - overall[1] + on_surface[0] - on_surface[1])
        result["elo_k32_mean"] = sum(self.ratings[32][p].at(cutoff) for p in (a, b)) / 2
        trend = [self.ratings[32][p].at(cutoff) - self.ratings[32][p].at(cutoff - timedelta(days=90)) for p in (a, b)]
        result["elo_k32_trend90_diff"] = trend[0] - trend[1]
        for window in WINDOWS:
            earliest = day - timedelta(days=window)
            for name, histories in (("form", self.overall), ("surface_form", self.surface)):
                stats = [histories[p if name == "form" else (p, surface)].summary(earliest, cutoff) for p in (a, b)]
                counts = [s[0] for s in stats]
                rates = [(s[1] + 1) / (s[0] + 2) for s in stats]
                residuals = [s[2] / (s[0] + 5) for s in stats]
                stem = f"{name}_{window}"
                result[f"{stem}_rate_diff"] = rates[0] - rates[1]
                result[f"{stem}_residual_diff"] = residuals[0] - residuals[1]
                result[f"{stem}_log_count_diff"] = log1p(counts[0]) - log1p(counts[1])
                result[f"{stem}_log_count_sum"] = log1p(counts[0]) + log1p(counts[1])
                result[f"{stem}_min_count"] = min(counts)
                # A symmetric measure of posterior uncertainty; fixed Beta(1,1).
                result[f"{stem}_variance_sum"] = sum(rates[i] * (1 - rates[i]) / (counts[i] + 3) for i in range(2))
        for name, histories in (("overall", self.overall), ("surface", self.surface)):
            last = [histories[p if name == "overall" else (p, surface)].last_date(cutoff) for p in (a, b)]
            gaps = [min((day - previous).days, 1095) if previous else 1095 for previous in last]
            result[f"{name}_inactivity_log_diff"] = log1p(gaps[0]) - log1p(gaps[1])
            result[f"{name}_inactivity_log_sum"] = log1p(gaps[0]) + log1p(gaps[1])
            result[f"{name}_unseen_sum"] = sum(previous is None for previous in last)
        low, high = sorted((a, b))
        for window in (364, 1095, None):
            earliest = day - timedelta(days=window) if window else date.min
            n, wins_low, _ = self.pair[low, high].summary(earliest, cutoff)
            advantage = 2 * (wins_low + 1) / (n + 2) - 1
            label = str(window) if window else "career"
            result[f"h2h_{label}_rate_diff"] = advantage if a == low else -advantage
            result[f"h2h_{label}_log_count"] = log1p(n)
        rank_a, rank_b = float(target["rank_a"]), float(target["rank_b"])
        if min(rank_a, rank_b) <= 0:
            raise ValueError("Rank features require positive rankings")
        result["reciprocal_rank_diff"] = 1 / rank_a - 1 / rank_b
        result["sqrt_rank_diff"] = rank_b ** .5 - rank_a ** .5
        result["log_rank_mean"] = (log(rank_a) + log(rank_b)) / 2
        result["log_rank_abs_gap"] = abs(log(rank_b) - log(rank_a))
        for name in ("Hard", "Clay", "Grass"):
            result[f"surface_is_{name.lower()}"] = float(surface == name)
        return result


def build_advanced_features(targets, inventory):
    """Return numeric features in exact target index/order, without labels.

    Only 2016–2023 targets are accepted in this development entry point. The
    next evaluator must explicitly adapt this gate to score the frozen model;
    this prevents silently treating held-out outcomes as development data.
    Original four published features are intentionally not duplicated here.
    """
    seasons = pd.to_numeric(targets["season"])
    if not seasons.between(2016, 2023).all():
        raise ValueError("Development features accept only 2016–2023 targets; final holdout is locked")
    state = HistoricalState(inventory)
    frame = pd.DataFrame([state.features(row) for row in targets.to_dict("records")], index=targets.index)
    frame.attrs["feature_version"] = FEATURE_VERSION
    frame.attrs["odd_columns"] = odd_feature_columns(frame.columns)
    frame.attrs["history_lag_days"] = HISTORY_LAG_DAYS
    frame.attrs["elo_update"] = "same-date batch from pre-date ratings; K=16,32,64; initial=1500"
    return frame

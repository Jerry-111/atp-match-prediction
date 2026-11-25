"""Uncertainty-aware, calendar-time Glicko features for Round 2 exploration.

Uses Mark Glickman's original Glicko update, with tournament start dates as
simultaneous rating periods and continuous elapsed weeks for RD diffusion.
The fixed 12 rating-points / sqrt(week) diffusion is a prespecified pragmatic
tennis assumption, not an estimated optimum. No score-dependent tuning occurs
here. See docs/research/round2_ratings.md for the methodological adaptations.

Ratings use only eligible Tour inventory rows in 2006--2023. Every target
queries results through T-21 inclusive and propagates their uncertainty to T.
All *_diff columns are odd under player exchange; every other column is even.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import groupby
from math import exp, fsum, isfinite, log, pi, sqrt

import pandas as pd

from scripts.advanced_features import _date

FEATURE_VERSION = "dynamic-glicko-1.0"
INITIAL_RATING = 1500.0
INITIAL_RD = 350.0
DIFFUSION_PER_SQRT_WEEK = 12.0
LAG_DAYS = 21
Q = log(10.0) / 400.0


def glicko_g(rd):
    """Opponent-uncertainty attenuation from the original Glicko system."""
    return 1.0 / sqrt(1.0 + 3.0 * Q * Q * rd * rd / (pi * pi))


def _logistic(value):
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    e = exp(value)
    return e / (1.0 + e)


def inflated_rd(rd, elapsed_days, diffusion=DIFFUSION_PER_SQRT_WEEK):
    """Calendar-time uncertainty diffusion, capped at the initial RD."""
    if elapsed_days < 0:
        raise ValueError("Cannot propagate uncertainty backward in time")
    return min(INITIAL_RD, sqrt(rd * rd + diffusion * diffusion * elapsed_days / 7.0))


def update_glicko(rating, rd, opponents):
    """Batch one player's period from pre-period (rating, RD, outcome) rows.

    This pure update implements Glickman's equations directly. It deliberately
    performs no RD diffusion; callers supply the inflated pre-period RD.
    """
    gradients, information = [], []
    for opponent_rating, opponent_rd, won in opponents:
        if won not in (0.0, 1.0) or opponent_rd <= 0:
            raise ValueError("Glicko requires positive RD and binary outcomes")
        g = glicko_g(opponent_rd)
        expected = _logistic(Q * g * (rating - opponent_rating))
        gradients.append(g * (won - expected))
        information.append(Q * Q * g * g * expected * (1.0 - expected))
    if rd <= 0:
        raise ValueError("Glicko requires positive own RD")
    precision = 1.0 / (rd * rd) + fsum(information)
    variance = 1.0 / precision
    return rating + Q * variance * fsum(gradients), sqrt(variance)


@dataclass
class DynamicHistory:
    dates: list = field(default_factory=list)
    ratings: list = field(default_factory=list)
    deviations: list = field(default_factory=list)

    def append(self, day, rating, rd):
        if self.dates and day <= self.dates[-1]:
            raise ValueError("Glicko snapshots require strictly increasing dates")
        self.dates.append(day)
        self.ratings.append(rating)
        self.deviations.append(rd)

    def at(self, cutoff, forecast_date=None):
        """Read an earlier posterior and inflate RD to a known forecast date."""
        forecast_date = cutoff if forecast_date is None else forecast_date
        if forecast_date < cutoff:
            raise ValueError("Forecast date must not precede information cutoff")
        end = bisect_right(self.dates, cutoff)
        if not end:
            return INITIAL_RATING, INITIAL_RD
        pos = end - 1
        return self.ratings[pos], inflated_rd(self.deviations[pos], (forecast_date - self.dates[pos]).days)


class DynamicRatings:
    """Build immutable overall and independent same-surface Glicko histories."""

    def __init__(self, inventory):
        mask = inventory["source"].eq("tour") & pd.to_numeric(inventory["season"]).between(2006, 2023)
        # Held-out labels and non-Tour results are discarded before conversion.
        records, seen = [], set()
        for row in inventory.loc[mask].to_dict("records"):
            match_id = str(row["match_id"])
            if match_id in seen:
                raise ValueError(f"Duplicate historical match {match_id}")
            seen.add(match_id)
            a, b = str(row["player_a_id"]), str(row["player_b_id"])
            won, surface = float(row["a_won"]), str(row["surface"])
            if a == b or won not in (0.0, 1.0):
                raise ValueError(f"Invalid historical player/outcome in {match_id}")
            if surface not in ("Hard", "Clay", "Grass"):
                raise ValueError(f"Invalid historical surface in {match_id}")
            records.append((_date(row["tourney_date"]), match_id, a, b, won, surface))
        records.sort()
        self.overall = defaultdict(DynamicHistory)
        self.surface = defaultdict(DynamicHistory)
        for day, matches in groupby(records, key=lambda row: row[0]):
            overall_games, surface_games = defaultdict(list), defaultdict(list)
            for _, _, a, b, won, surface in matches:
                ra, da = self.overall[a].at(day)
                rb, db = self.overall[b].at(day)
                overall_games[a].append((rb, db, won))
                overall_games[b].append((ra, da, 1 - won))
                ra, da = self.surface[a, surface].at(day)
                rb, db = self.surface[b, surface].at(day)
                surface_games[a, surface].append((rb, db, won))
                surface_games[b, surface].append((ra, da, 1 - won))
            # Update only after every game's inputs have been captured, so a
            # result cannot alter another result's pre-period opponent rating.
            for histories, games in ((self.overall, overall_games), (self.surface, surface_games)):
                updated = {}
                for key, opponents in games.items():
                    rating, rd = histories[key].at(day)
                    updated[key] = update_glicko(rating, rd, opponents)
                for key, (rating, rd) in updated.items():
                    histories[key].append(day, rating, rd)

    def features(self, row):
        day = _date(row["tourney_date"])
        cutoff = day - timedelta(days=LAG_DAYS)
        a, b, surface = str(row["player_a_id"]), str(row["player_b_id"]), str(row["surface"])
        if a == b or surface not in ("Hard", "Clay", "Grass"):
            raise ValueError("Target requires different players and a supported surface")
        overall = [self.overall[p].at(cutoff, day) for p in (a, b)]
        on_surface = [self.surface[p, surface].at(cutoff, day) for p in (a, b)]
        result = {}
        for stem, values in (("glicko", overall), ("surface_glicko", on_surface)):
            (ra, da), (rb, db) = values
            result[f"{stem}_rating_diff"] = ra - rb
            # The common attenuation uses both RDs, guaranteeing complementary
            # probabilities; it is an approximation, not a standard update.
            result[f"{stem}_adjusted_diff"] = glicko_g(sqrt(da*da + db*db)) * (ra-rb)
            result[f"{stem}_rd_log_diff"] = log(da) - log(db)
            result[f"{stem}_rd_sum"] = da + db
            result[f"{stem}_rating_mean"] = (ra + rb) / 2
        pooled, weights = [], []
        for (global_rating, global_rd), (surface_rating, surface_rd) in zip(overall, on_surface):
            # Remove the initial prior precision: unseen surfaces have zero
            # incremental evidence, hence fall back exactly to overall skill.
            surface_information = max(1 / (surface_rd * surface_rd) - 1 / INITIAL_RD**2, 0)
            weight = surface_information / (surface_information + 1 / (global_rd * global_rd))
            pooled.append(global_rating + weight * (surface_rating-global_rating))
            weights.append(weight)
        result["pooled_glicko_rating_diff"] = pooled[0] - pooled[1]
        # Deliberately do not claim independent evidence or compute a falsely
        # precise fused RD: global/surface histories overlap.
        conservative_rd = [max(g[1], s[1]) for g, s in zip(overall, on_surface)]
        result["pooled_glicko_adjusted_diff"] = glicko_g(sqrt(sum(d*d for d in conservative_rd))) * (pooled[0]-pooled[1])
        result["glicko_surface_weight_diff"] = weights[0]-weights[1]
        result["glicko_surface_weight_sum"] = sum(weights)
        result["glicko_specialization_diff"] = (on_surface[0][0]-overall[0][0]) - (on_surface[1][0]-overall[1][0])
        for days in (90, 364):
            earlier = cutoff - timedelta(days=days)
            trend = [overall[i][0] - self.overall[p].at(earlier)[0] for i, p in enumerate((a, b))]
            result[f"glicko_trend{days}_diff"] = trend[0] - trend[1]
        return result


def build_dynamic_features(targets, inventory, *, first_target_season=2016):
    """Return 17 numeric features in target index/order without using labels."""
    if first_target_season not in range(2009, 2017):
        raise ValueError("first_target_season must be an integer from 2009 through 2016")
    if not pd.to_numeric(targets["season"]).between(first_target_season, 2023).all():
        raise ValueError(f"Development targets must be {first_target_season}--2023; final holdout is locked")
    state = DynamicRatings(inventory)
    result = pd.DataFrame([state.features(row) for row in targets.to_dict("records")], index=targets.index)
    if not all(isfinite(v) for v in result.to_numpy().flat):
        raise ValueError("Non-finite dynamic features")
    result.attrs.update(feature_version=FEATURE_VERSION, history_lag_days=LAG_DAYS,
                        diffusion_per_sqrt_week=DIFFUSION_PER_SQRT_WEEK,
                        odd_columns=[c for c in result if c.endswith("_diff")])
    return result

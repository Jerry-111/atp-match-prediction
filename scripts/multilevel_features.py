"""Past-only cross-level Elo with fixed common and first-level priors.

Tour, Challenger and Tour/major qualifying history update shared player
ratings. A matched Tour-only bank isolates the addition of historical results.
Ratings are simultaneous within each source tournament-start date. Surface
ratings begin at the player's pre-date overall rating and are shrunk toward
overall strength by n/(n+20). See docs/research/round5_multilevel.md.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import groupby
from math import exp, fsum, isfinite, log, log1p

import pandas as pd

from scripts.advanced_features import _date

FEATURE_VERSION = "multilevel-elo-1.0"
SOURCES = ("tour", "challenger", "qualifying")
LEVEL_PRIORS = {"tour": 1500.0, "challenger": 1300.0, "qualifying": 1400.0}
INITIAL_RATING = 1500.0
K_FACTOR = 32.0
SURFACE_SHRINK_MATCHES = 20.0
LAG_DAYS = 21


def elo_expectation(rating, opponent):
    """Stable ordinary Elo probability; 400 rating points imply 10:1 odds."""
    value = log(10.0) * (rating - opponent) / 400.0
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def surface_blend(overall, on_surface, matches):
    """Fixed evidence-count shrinkage, not a claim of independent evidence."""
    if matches < 0:
        raise ValueError("Surface history count must be nonnegative")
    weight = matches / (matches + SURFACE_SHRINK_MATCHES)
    return overall + weight * (on_surface - overall), weight


def _player(value):
    if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none"):
        raise ValueError("Missing historical/target player ID")
    return str(value)


def _season_date(day, season):
    # The source legitimately labels late-December opening weeks as next season.
    if day.year == season or (day.year == season - 1 and day.month == 12 and day.day >= 25):
        return
    raise ValueError("Tournament date does not match its development season")


def historical_records(inventory, latest=None):
    """Discard held-out seasons before reading identities or outcomes.

    Original match IDs, including legitimate qualifying/main-draw repeat
    pairings, remain distinct. Never infer duplicate matches from player pairs.
    """
    seasons = pd.to_numeric(inventory["season"], errors="coerce")
    mask = inventory["source"].isin(SOURCES) & seasons.isin(range(2006, 2024))
    records, seen = [], set()
    for row in inventory.loc[mask].to_dict("records"):
        day = _date(row["tourney_date"])
        _season_date(day, int(row["season"]))
        if latest is not None and day > latest:
            continue
        identifier = str(row["match_id"])
        if identifier in seen:
            raise ValueError(f"Duplicate multilevel historical match: {identifier}")
        seen.add(identifier)
        a, b = _player(row["player_a_id"]), _player(row["player_b_id"])
        won, surface = float(row["a_won"]), str(row["surface"])
        if a == b or won not in (0.0, 1.0) or surface not in ("Hard", "Clay", "Grass"):
            raise ValueError(f"Invalid multilevel history: {identifier}")
        records.append((day, identifier, str(row["source"]), a, b, won, surface))
    return sorted(records)


@dataclass
class RatingHistory:
    days: list = field(default_factory=list)
    ratings: list = field(default_factory=list)
    counts: list = field(default_factory=list)

    def append(self, day, rating, count):
        if self.days and day <= self.days[-1]:
            raise ValueError("Rating snapshot dates must increase")
        self.days.append(day)
        self.ratings.append(rating)
        self.counts.append(count)

    def at(self, cutoff, default=INITIAL_RATING):
        position = bisect_right(self.days, cutoff) - 1
        if position < 0:
            # Do not return a prior based on the player's future first level.
            return default, 0
        return self.ratings[position], self.counts[position]


class RatingBank:
    """Date-batched Elo with first-observed-level or common initialization."""

    def __init__(self, records, *, level_aware=False, tour_only=False):
        self.overall, self.surface = defaultdict(RatingHistory), defaultdict(RatingHistory)
        for day, raw_batch in groupby(records, key=lambda r: r[0]):
            batch = [r for r in raw_batch if not tour_only or r[2] == "tour"]
            if not batch:
                continue
            # Highest level among SAME-DATE observed entries resolves ties.
            # A later Tour appearance cannot change an earlier Challenger prior.
            initial = {}
            for _, _, source, a, b, _, _ in batch:
                prior = LEVEL_PRIORS[source] if level_aware else INITIAL_RATING
                for player in (a, b):
                    initial[player] = max(initial.get(player, prior), prior)
            overall_pre = {p: self.overall[p].at(day, prior) for p, prior in initial.items()}
            surface_pre = {}
            overall_updates, surface_updates = defaultdict(list), defaultdict(list)
            for _, _, _, a, b, won, surface in batch:
                ra, rb = overall_pre[a][0], overall_pre[b][0]
                delta = K_FACTOR * (won - elo_expectation(ra, rb))
                overall_updates[a].append(delta)
                overall_updates[b].append(-delta)
                for player in (a, b):
                    key = (player, surface)
                    if key not in surface_pre:
                        surface_pre[key] = self.surface[key].at(day, overall_pre[player][0])
                ra, rb = surface_pre[a, surface][0], surface_pre[b, surface][0]
                delta = K_FACTOR * (won - elo_expectation(ra, rb))
                surface_updates[a, surface].append(delta)
                surface_updates[b, surface].append(-delta)
            # All expectations above use pre-period states, including same-date
            # qualifying and main-draw games. Update only when the batch ends.
            for histories, previous, updates in (
                (self.overall, overall_pre, overall_updates),
                (self.surface, surface_pre, surface_updates),
            ):
                for key, deltas in updates.items():
                    rating, count = previous[key]
                    histories[key].append(day, rating + fsum(deltas), count + len(deltas))

    def player(self, player, surface, cutoff):
        overall, count = self.overall[player].at(cutoff)
        on_surface, surface_count = self.surface[player, surface].at(cutoff, overall)
        blend, weight = surface_blend(overall, on_surface, surface_count)
        return {"overall": overall, "surface": on_surface, "blend": blend,
                "count": count, "surface_count": surface_count, "surface_weight": weight}


class MultilevelRatings:
    def __init__(self, records):
        self.banks = {"tour": RatingBank(records, tour_only=True),
                      "common": RatingBank(records),
                      "level": RatingBank(records, level_aware=True)}
        self.source_days = defaultdict(list)
        for day, _, source, a, b, _, _ in records:
            self.source_days[source, a].append(day)
            self.source_days[source, b].append(day)

    def features(self, day, a, b, surface):
        cutoff = day - timedelta(days=LAG_DAYS)
        states, result = {}, {}
        for name, bank in self.banks.items():
            states[name] = pair = [bank.player(p, surface, cutoff) for p in (a, b)]
            for rating in ("overall", "surface", "blend"):
                result[f"ml_{name}_{rating}_diff"] = pair[0][rating] - pair[1][rating]
        for name in ("common", "level"):
            pair, bank = states[name], self.banks[name]
            for rating in ("overall", "blend"):
                result[f"ml_{name}_{rating}_minus_tour_diff"] = (
                    result[f"ml_{name}_{rating}_diff"] - result[f"ml_tour_{rating}_diff"])
            for window in (90, 364):
                earlier = cutoff - timedelta(days=window)
                old = [bank.overall[p].at(earlier) for p in (a, b)]
                # A player's first level prior is not a measured deterioration
                # from 1500. No earlier posterior means no estimable trend.
                trend = [pair[i]["overall"] - old[i][0] if old[i][1] else 0.0 for i in (0, 1)]
                result[f"ml_{name}_trend{window}_diff"] = trend[0] - trend[1]
            result[f"ml_{name}_overall_mean"] = (pair[0]["overall"] + pair[1]["overall"]) / 2
        earliest = day - timedelta(days=364)
        recent, career = [], []
        for player in (a, b):
            totals, windows = {}, {}
            for source in SOURCES:
                days = self.source_days[source, player]
                totals[source] = bisect_right(days, cutoff)
                windows[source] = totals[source] - bisect_left(days, earliest)
            recent.append(windows)
            career.append(totals)
        recent_total = [sum(r.values()) for r in recent]
        for source in SOURCES:
            counts = [r[source] for r in recent]
            result[f"ml_{source}_recent364_log_count_diff"] = log1p(counts[0]) - log1p(counts[1])
            result[f"ml_{source}_recent364_log_count_sum"] = log1p(counts[0]) + log1p(counts[1])
            # Fixed five-match pseudo-exposure shrinks a sparse level share.
            result[f"ml_{source}_recent364_share_diff"] = (
                counts[0] / (recent_total[0] + 5) - counts[1] / (recent_total[1] + 5))
        total = [sum(c.values()) for c in career]
        lower = [c["challenger"] + c["qualifying"] for c in career]
        for stem, counts in (("career", total), ("lower_career", lower)):
            result[f"ml_{stem}_log_count_diff"] = log1p(counts[0]) - log1p(counts[1])
            result[f"ml_{stem}_log_count_sum"] = log1p(counts[0]) + log1p(counts[1])
        result["ml_career_min_count"] = min(total)
        weights = [s["surface_weight"] for s in states["common"]]
        result["ml_surface_weight_diff"] = weights[0] - weights[1]
        result["ml_surface_weight_sum"] = sum(weights)
        fractions = [(r["challenger"] + r["qualifying"]) / (sum(r.values()) + 5) for r in recent]
        result["ml_lower_recent364_share_diff"] = fractions[0] - fractions[1]
        result["ml_unseen_sum"] = float(sum(n == 0 for n in total))
        return result


def build_multilevel_features(targets, inventory):
    """Return 37 finite columns in identical target index/order, without labels.

    Every *_diff column is odd under exchanging A/B; all other columns are
    invariant. Target seasons outside 2009--2023 raise before history loading.
    """
    seasons = pd.to_numeric(targets["season"], errors="coerce")
    if targets.empty or not seasons.isin(range(2009, 2024)).all():
        raise ValueError("Multilevel targets must be nonempty 2009--2023; final holdout is locked")
    query = []
    for row in targets.to_dict("records"):
        day = _date(row["tourney_date"])
        _season_date(day, int(row["season"]))
        a, b = _player(row["player_a_id"]), _player(row["player_b_id"])
        surface = str(row["surface"])
        if a == b or surface not in ("Hard", "Clay", "Grass"):
            raise ValueError("Multilevel targets require distinct players and a supported surface")
        query.append((day, a, b, surface))
    latest = max(q[0] for q in query) - timedelta(days=LAG_DAYS)
    records = historical_records(inventory, latest=latest)
    state = MultilevelRatings(records)
    frame = pd.DataFrame([state.features(*q) for q in query], index=targets.index, dtype=float)
    if len(frame.columns) != 37 or not all(isfinite(v) for v in frame.to_numpy().flat):
        raise ValueError("Unexpected multilevel feature schema or non-finite values")
    frame.attrs.update(feature_version=FEATURE_VERSION, history_lag_days=LAG_DAYS,
                       initial_rating=INITIAL_RATING, level_priors=dict(LEVEL_PRIORS),
                       k_factor=K_FACTOR, surface_shrink_matches=SURFACE_SHRINK_MATCHES,
                       source_history_counts={s: sum(r[2] == s for r in records) for s in SOURCES},
                       odd_columns=[c for c in frame if c.endswith("_diff")],
                       even_columns=[c for c in frame if not c.endswith("_diff")],
                       latest_history_cutoff=latest.isoformat(),
                       future_roster_used=False, final_test_evaluated=False)
    return frame

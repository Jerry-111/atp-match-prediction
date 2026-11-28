"""Dynamic opponent-adjusted serve/return skills, with past-only queries.

For server i and returner j, p(point won on serve) = sigmoid(b_surface+s_i-r_j).
An online diagonal Laplace approximation uses fractional-binomial information,
with at most 100 effective service points per player per match. Updates are
damped, simultaneous within each tournament-start date, and discounted with
fixed 365/730-day half-lives. Separate surface skills are pooled toward overall
skills. This is a new approximate model, not an implementation of a published
state-of-the-art algorithm. No parameters are fitted on 2023 outcomes.
"""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from itertools import groupby
from math import exp, log, log1p

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.serve_features import _annual_rows, _counts, development_entries


FEATURE_VERSION = "1.0"
HALF_LIVES = (365, 730)
LAG = 21
PRIOR_PRECISION = 25.0
BASELINE_PRIOR_POINTS = 2000.0
BASELINE_PRIOR_RATE = .65
MATCH_POINT_CAP = 100.0
UPDATE_DAMPING = .5
SURFACE_POOL_POINTS = 400.0


def _sigmoid(value):
    if value >= 0:
        return 1 / (1 + exp(-value))
    e = exp(value)
    return e / (1 + e)


def hold_probability(point_probability):
    """Exact IID service-game win probability, including an unlimited deuce."""
    p, q = point_probability, 1 - point_probability
    return p**4 * (1 + 4*q + 10*q*q) + 20*p**3*q**3 * p*p / (p*p + q*q)


@dataclass(frozen=True)
class Skill:
    mean: float = 0.
    precision: float = PRIOR_PRECISION
    points: float = 0.
    day: object = None

    def at(self, day, half_life):
        if self.day is None:
            return Skill(day=day)
        days = (day - self.day).days
        if days < 0:
            raise ValueError("Cannot propagate a skill backwards in time")
        decay = 2. ** (-days / half_life)
        precision = PRIOR_PRECISION + decay * (self.precision - PRIOR_PRECISION)
        # Discount the likelihood's natural parameters, retaining a zero-mean prior.
        mean = decay * self.precision * self.mean / precision
        return Skill(mean, precision, decay * self.points, day)


class SkillHistory:
    def __init__(self):
        self.dates, self.states = [], []

    def record(self, state):
        self.dates.append(state.day)
        self.states.append(state)

    def at(self, day, half_life):
        index = bisect_right(self.dates, day)
        state = self.states[index - 1] if index else Skill()
        return state.at(day, half_life)


class BaselineHistory:
    def __init__(self):
        self.dates, self.counts = [], []

    def at(self, day, half_life):
        index = bisect_right(self.dates, day)
        if not index:
            return 0., 0.
        won, total = self.counts[index - 1]
        decay = 2. ** (-(day - self.dates[index - 1]).days / half_life)
        return won * decay, total * decay

    def logit(self, day, half_life):
        won, total = self.at(day, half_life)
        rate = (won + BASELINE_PRIOR_RATE * BASELINE_PRIOR_POINTS) / (total + BASELINE_PRIOR_POINTS)
        return log(rate / (1 - rate))

    def record(self, day, half_life, won, total):
        past_won, past_total = self.at(day, half_life)
        self.dates.append(day)
        self.counts.append((past_won + won, past_total + total))


def _joined_history(inventory, cache_dir, raw_rows):
    eligible = inventory.loc[inventory.source.eq("tour") &
                             pd.to_numeric(inventory.season).between(2006, 2023)]
    expected = {}
    for record in eligible.to_dict("records"):
        key = str(record["match_id"])
        if key in expected:
            raise ValueError(f"Duplicate inventory match: {key}")
        expected[key] = record
    entries = development_entries() if raw_rows is None else []
    rows = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    seen, records = set(), []
    missing = 0
    for season, raw in rows:
        if not 2006 <= int(season) <= 2023:
            continue
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in seen:
            raise ValueError(f"Duplicate statistics match: {key}")
        seen.add(key)
        raw = corrected_tour_row(raw)
        match = expected[key]
        a, b = str(match["player_a_id"]), str(match["player_b_id"])
        winner, loser = str(raw["winner_id"]), str(raw["loser_id"])
        if (a == b or {a, b} != {winner, loser} or int(match["season"]) != int(season) or
                _date(match["tourney_date"]) != _date(raw["tourney_date"]) or
                match["surface"] != raw["surface"]):
            raise ValueError(f"Statistics/inventory identity mismatch: {key}")
        counts = _counts(raw)
        if counts is None:
            missing += 1
            continue
        (ww, wn), (lw, ln) = counts
        records.append((_date(match["tourney_date"]), key, str(match["surface"]),
                        winner, loser, ww, wn, lw, ln))
    if expected.keys() - seen:
        raise ValueError("Technical source omitted eligible inventory matches")
    return sorted(records), {"source_files": entries, "eligible_history_matches": len(expected),
                             "valid_history_matches": len(records), "missing_statistics_matches": missing,
                             "injected_raw_rows": raw_rows is not None}


class AdjustedPointState:
    def __init__(self, records, half_life):
        self.half_life = half_life
        self.overall, self.surface = defaultdict(SkillHistory), defaultdict(SkillHistory)
        self.baselines = defaultdict(BaselineHistory)
        for day, grouped in groupby(records, key=lambda r: r[0]):
            matches = list(grouped)
            # All gradients see the same pre-date state, including the baseline.
            deltas = {"overall": defaultdict(lambda: [0., 0., 0.]),
                      "surface": defaultdict(lambda: [0., 0., 0.])}
            states = {"overall": {}, "surface": {}}
            baseline_deltas = defaultdict(lambda: [0., 0.])
            for _, _, surface, a, b, aw, an, bw, bn in matches:
                baseline = self.baselines[surface].logit(day, half_life)
                for server, returner, won, total in ((a, b, aw, an), (b, a, bw, bn)):
                    weight = min(total, MATCH_POINT_CAP) / total
                    effective_n, effective_w = weight * total, weight * won
                    baseline_deltas[surface][0] += effective_w
                    baseline_deltas[surface][1] += effective_n
                    for scope, bank in (("overall", self.overall), ("surface", self.surface)):
                        sk = (server, "serve") if scope == "overall" else (server, surface, "serve")
                        rk = (returner, "return") if scope == "overall" else (returner, surface, "return")
                        for key in (sk, rk):
                            if key not in states[scope]:
                                states[scope][key] = bank[key].at(day, half_life)
                        serve, ret = states[scope][sk], states[scope][rk]
                        p = _sigmoid(baseline + serve.mean - ret.mean)
                        residual, information = effective_w - effective_n*p, effective_n*p*(1-p)
                        for key, sign in ((sk, 1), (rk, -1)):
                            delta = deltas[scope][key]
                            delta[0] += sign * residual
                            delta[1] += information
                            delta[2] += effective_n
            for scope, bank in (("overall", self.overall), ("surface", self.surface)):
                for key in sorted(deltas[scope]):
                    gradient, information, points = deltas[scope][key]
                    old = states[scope][key]
                    precision = old.precision + information
                    mean = old.mean + UPDATE_DAMPING * gradient / precision
                    bank[key].record(Skill(mean, precision, old.points + points, day))
            for surface, (won, total) in baseline_deltas.items():
                self.baselines[surface].record(day, half_life, won, total)

    def features(self, target):
        cutoff = _date(target["tourney_date"]) - timedelta(days=LAG)
        a, b, surface = str(target["player_a_id"]), str(target["player_b_id"]), str(target["surface"])
        if a == b:
            raise ValueError("A match must have two different players")
        global_states, local_states = [], []
        for player in (a, b):
            glob = [self.overall[player, role].at(cutoff, self.half_life) for role in ("serve", "return")]
            local = [self.surface[player, surface, role].at(cutoff, self.half_life) for role in ("serve", "return")]
            global_states.append(glob)
            local_states.append([g.mean + s.points / (s.points + SURFACE_POOL_POINTS) * (s.mean - g.mean)
                                 for g, s in zip(glob, local)])
        baseline = self.baselines[surface].logit(cutoff, self.half_life)
        prefix = f"adjpoints_h{self.half_life}"
        result = {}
        for scope, skills in (("overall", [[s.mean for s in g] for g in global_states]),
                              ("surface", local_states)):
            (sa, ra), (sb, rb) = skills
            result[f"{prefix}_{scope}_serve_skill_diff"] = sa - sb
            result[f"{prefix}_{scope}_return_skill_diff"] = ra - rb
            pa, pb = _sigmoid(baseline + sa - rb), _sigmoid(baseline + sb - ra)
            for label, va, vb in (("serve_prob", pa, pb),
                                  ("hold_prob", hold_probability(pa), hold_probability(pb))):
                result[f"{prefix}_{scope}_{label}_diff"] = va - vb
                result[f"{prefix}_{scope}_{label}_sum"] = va + vb
        exposures = [sum(s.points for s in g) for g in global_states]
        result[f"{prefix}_exposure_log_diff"] = log1p(exposures[0]) - log1p(exposures[1])
        result[f"{prefix}_exposure_log_sum"] = log1p(exposures[0]) + log1p(exposures[1])
        result[f"{prefix}_exposure_min"] = min(exposures)
        result[f"{prefix}_unseen_sum"] = float(exposures[0] == 0) + float(exposures[1] == 0)
        result[f"{prefix}_surface_baseline"] = _sigmoid(baseline)
        return result


def build_adjusted_point_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None,
                                  first_target_season=2016):
    """Return 34 numeric columns with identical index; *_diff odd, others even."""
    if first_target_season not in range(2009, 2017):
        raise ValueError("first_target_season must be an integer from 2009 through 2016")
    if not pd.to_numeric(targets.season).between(first_target_season, 2023).all():
        raise ValueError(f"Final holdout is locked: targets must be from {first_target_season}..2023")
    records, metadata = _joined_history(inventory, cache_dir, raw_rows)
    states = [AdjustedPointState(records, half_life) for half_life in HALF_LIVES]
    output = []
    for target in targets.to_dict("records"):
        row = {}
        for state in states:
            row.update(state.features(target))
        output.append(row)
    result = pd.DataFrame(output, index=targets.index, dtype=float)
    result.attrs.update(metadata)
    result.attrs.update({"feature_version": FEATURE_VERSION, "half_lives_days": list(HALF_LIVES),
                         "history_lag_days": LAG, "prior_precision": PRIOR_PRECISION,
                         "match_point_cap": MATCH_POINT_CAP, "update_damping": UPDATE_DAMPING,
                         "baseline_prior_points": BASELINE_PRIOR_POINTS,
                         "baseline_prior_rate": BASELINE_PRIOR_RATE,
                         "surface_pool_points": SURFACE_POOL_POINTS})
    return result

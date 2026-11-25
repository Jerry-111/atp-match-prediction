"""Paper-grounded WElo game/set ratings under this repository's time protocol.

Angelini, Candila & De Angelis (2022), doi:10.1016/j.ejor.2021.04.011:
Eq. (1) gives the base-10, 400-point probability; Eq. (3) multiplies each
player's Elo residual by the WINNER's game share. Appendix A.1 replaces games
with sets. Section 3 specifies initial1500 and K=250/(N+5)^.4. Author software
uses N=max(1, previous appearances). No R source code is copied here.

Adaptations: eligible2006..2023 Tour history, simultaneous tournament-date
updates, T-21 queries, independent same-surface banks, derived contrasts/trends.
This does not reproduce the paper's original cohort or sequential forecasts.
"""

import re
from bisect import bisect_right
from collections import defaultdict
from datetime import timedelta
from itertools import groupby
from math import fsum

import numpy as np
import pandas as pd

from scripts.advanced_features import _date, elo_probability
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.serve_features import _annual_rows, development_entries


FEATURE_VERSION = "paper-welo-1.0"
PAPER_DOI = "10.1016/j.ejor.2021.04.011"
PAPER_PDF_SHA256 = "09fa99db3bfe84ae9cbf7f846455c07a04059fc224d00c3264151a0048688b50"
INITIAL_RATING = 1500.
LAG_DAYS = 21
VARIANTS = ("elo", "welo_games", "welo_sets")
SET_TOKEN = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")


def score_weights(score, best_of):
    """Return winner's (game share, set share), preserving sub-.5 game shares.

    Scores must be written from the match winner's perspective, as in the raw
    source. Tiebreak annotations are points, not extra games. Advantage-set
    scores such as70-68 and final-set tiebreak scores13-12 are retained.
    """
    if int(best_of) not in (3, 5):
        raise ValueError("WElo requires best_of3 or5")
    tokens = str(score).strip().split()
    parsed = [SET_TOKEN.fullmatch(token) for token in tokens]
    if not tokens or any(token is None for token in parsed):
        raise ValueError(f"Unsupported or incomplete score: {score}")
    sets = [tuple(map(int, token.groups())) for token in parsed]
    def complete_set(a, b):
        high, low = max(a, b), min(a, b)
        return ((high == 6 and low <= 4) or (high == 7 and low in (5, 6)) or
                (high >= 8 and high-low == 2) or (high, low) == (13, 12))
    if not all(complete_set(a, b) for a, b in sets):
        raise ValueError(f"Incomplete set score: {score}")
    wins, losses = sum(a > b for a, b in sets), sum(a < b for a, b in sets)
    needed = int(best_of) // 2 + 1
    if wins != needed or losses >= needed or len(sets) > int(best_of):
        raise ValueError(f"Score is inconsistent with winner/best_of: {score}")
    if sum(a > b for a, b in sets[:-1]) >= needed or sets[-1][0] <= sets[-1][1]:
        raise ValueError(f"Sets continue after the match is decided: {score}")
    games_won = sum(a for a, _ in sets)
    games_total = sum(a + b for a, b in sets)
    return games_won / games_total, wins / (wins + losses)


def experience_k(previous_appearances):
    """Paper Section3 with the author's first-appearance N=1 convention."""
    if previous_appearances < 0:
        raise ValueError("Previous appearance count cannot be negative")
    return 250. / (max(1, previous_appearances) + 5.) ** .4


def paper_update(rating, opponent_rating, won, previous_appearances, weight=1.):
    """Single-player Eq.3 increment, useful for hand-equation verification."""
    if won not in (0, 1) or not 0 < weight <= 1:
        raise ValueError("Binary outcome and a positive weight no greater than1 required")
    return experience_k(previous_appearances) * (won - elo_probability(rating - opponent_rating)) * weight


class SnapshotHistory:
    def __init__(self):
        self.dates, self.ratings, self.counts = [], [], []

    def append(self, day, rating, count):
        self.dates.append(day)
        self.ratings.append(rating)
        self.counts.append(count)

    def at(self, cutoff):
        end = bisect_right(self.dates, cutoff)
        return (self.ratings[end-1], self.counts[end-1]) if end else (INITIAL_RATING, 0)


def _source_records(inventory, cache_dir, raw_rows):
    eligible = inventory.loc[inventory.source.eq("tour") &
                             pd.to_numeric(inventory.season).between(2006, 2023)]
    expected = {}
    for row in eligible.to_dict("records"):
        key = str(row["match_id"])
        if key in expected:
            raise ValueError(f"Duplicate history match: {key}")
        expected[key] = row
    entries = development_entries() if raw_rows is None else []
    raw_rows = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    records, seen = [], set()
    for season, raw in raw_rows:
        if not 2006 <= int(season) <= 2023:
            continue
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in seen:
            raise ValueError(f"Duplicate score source: {key}")
        seen.add(key)
        raw = corrected_tour_row(raw)
        row = expected[key]
        a, b = str(row["player_a_id"]), str(row["player_b_id"])
        won = float(row["a_won"])
        winner, loser = str(raw["winner_id"]), str(raw["loser_id"])
        if (a == b or won not in (0, 1) or winner != (a if won else b) or
                loser != (b if won else a) or int(row["season"]) != int(season) or
                _date(row["tourney_date"]) != _date(raw["tourney_date"]) or row["surface"] != raw["surface"]):
            raise ValueError(f"Score/inventory identity or winner mismatch: {key}")
        game_weight, set_weight = score_weights(raw["score"], raw["best_of"])
        records.append((_date(row["tourney_date"]), key, str(row["surface"]), winner, loser,
                        game_weight, set_weight))
    if expected.keys() - seen:
        raise ValueError("Score source omitted eligible history matches")
    return sorted(records), entries


class WeightedRatingState:
    def __init__(self, records):
        self.banks = {(scope, variant): defaultdict(SnapshotHistory)
                      for scope in ("overall", "surface") for variant in VARIANTS}
        for day, grouped in groupby(records, key=lambda r: r[0]):
            matches = list(grouped)
            for scope in ("overall", "surface"):
                for variant in VARIANTS:
                    bank = self.banks[scope, variant]
                    deltas, appearances = defaultdict(list), defaultdict(int)
                    for _, _, surface, winner, loser, game_weight, set_weight in matches:
                        wk = winner if scope == "overall" else (winner, surface)
                        lk = loser if scope == "overall" else (loser, surface)
                        wr, wn = bank[wk].at(day)
                        lr, ln = bank[lk].at(day)
                        weight = {"elo": 1., "welo_games": game_weight, "welo_sets": set_weight}[variant]
                        deltas[wk].append(paper_update(wr, lr, 1, wn, weight))
                        deltas[lk].append(paper_update(lr, wr, 0, ln, weight))
                        appearances[wk] += 1
                        appearances[lk] += 1
                    # No bank is changed until all matches on this date have
                    # seen pre-date ratings AND pre-date appearance counts.
                    for key in sorted(deltas):
                        rating, count = bank[key].at(day)
                        bank[key].append(day, rating + fsum(deltas[key]), count + appearances[key])

    def features(self, target):
        day = _date(target["tourney_date"])
        cutoff = day - timedelta(days=LAG_DAYS)
        a, b, surface = str(target["player_a_id"]), str(target["player_b_id"]), str(target["surface"])
        if a == b or surface not in ("Hard", "Clay", "Grass"):
            raise ValueError("Target requires different players and a supported surface")
        output = {}
        differences = {}
        for scope in ("overall", "surface"):
            keys = (a, b) if scope == "overall" else ((a, surface), (b, surface))
            for variant in VARIANTS:
                bank = self.banks[scope, variant]
                (ra, na), (rb, nb) = [bank[key].at(cutoff) for key in keys]
                name = f"paper_{scope}_{variant}"
                differences[scope, variant] = ra-rb
                output[f"{name}_diff"] = ra-rb
                output[f"{name}_mean"] = (ra+rb)/2
            for variant in ("welo_games", "welo_sets"):
                output[f"paper_{scope}_{variant}_minus_elo_diff"] = (
                    differences[scope, variant] - differences[scope, "elo"])
            output[f"paper_{scope}_min_matches"] = min(na, nb)
        for variant in ("welo_games", "welo_sets"):
            output[f"paper_blend_{variant}_diff"] = .5 * (
                differences["overall", variant] + differences["surface", variant])
            bank = self.banks["overall", variant]
            old_a, old_b = [bank[key].at(cutoff-timedelta(days=90))[0] for key in (a, b)]
            output[f"paper_overall_{variant}_trend90_diff"] = differences["overall", variant] - (old_a-old_b)
        return output


def build_paper_margin_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Return22 numeric columns, same index; *_diff odd and all others even."""
    if not pd.to_numeric(targets.season).between(2009, 2023).all():
        raise ValueError("Paper development targets must be2009..2023; final holdout is locked")
    records, entries = _source_records(inventory, cache_dir, raw_rows)
    state = WeightedRatingState(records)
    output = pd.DataFrame([state.features(row) for row in targets.to_dict("records")], index=targets.index)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("Non-finite paper-margin features")
    output.attrs.update({"feature_version": FEATURE_VERSION, "paper_doi": PAPER_DOI,
                         "paper_pdf_sha256": PAPER_PDF_SHA256, "equations": ["1", "3", "Appendix A.1"],
                         "initial_rating": INITIAL_RATING, "history_lag_days": LAG_DAYS,
                         "scale_factor": "250/(max(1, pre-date appearances)+5)^0.4",
                         "date_update": "simultaneous from pre-date ratings and appearance counts",
                         "adaptations": ["eligible Tour2006..2023", "T-21", "same-date batch",
                                         "separate surface ratings", "half overall/surface blend", "90-day trends"],
                         "history_matches": len(records), "source_files": entries,
                         "injected_raw_rows": raw_rows is not None})
    return output

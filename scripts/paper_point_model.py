"""Paper-supported point -> game -> set -> match probability aggregation.

Primary equations: Spanias, Professional Tennis: Quantitative Models and Ranking
Algorithms (Imperial PhD, 2015), chapter3 pp60--66, equations3.1--3.7:
https://www.doc.ic.ac.uk/~wjk/publications/spanias-2015.pdf
Related article: Spanias & Knottenbelt (2013), doi:10.1093/imaman/dps010.
Game formula/check: O'Malley's2007 NESSIS presentation, slides4 and18:
https://nessis.org/nessis07/James_OMalley.pdf

We implement the hierarchy, not the papers' point-estimation algorithms. Inputs
are this project's prior opponent-adjusted service-point estimates. Scoring is
STANDARDIZED: advantage games, 7-point win-by-two tiebreak at6--6 in EVERY set.
Historical advantage/10-point deciding-set rules are not reconstructed. Thus
the recursion is exact for the stated IID scoring model, not every actual event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_VERSION = "standardized-point-markov-1.0"
HALF_LIVES = (365, 730)
SCOPES = ("overall", "surface")


def _inputs(p_a, p_b):
    a, b = np.broadcast_arrays(np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float))
    if (not np.isfinite(a).all() or not np.isfinite(b).all() or
            np.any((a <= 0) | (a >= 1)) or np.any((b <= 0) | (b >= 1))):
        raise ValueError("Service-point probabilities must lie strictly between0 and1")
    return a.ravel(), b.ravel(), a.shape


def _restore(values, shape):
    result = np.clip(np.asarray(values).reshape(shape), 0., 1.)
    return float(result) if not shape else result


def game_probability(p):
    """Server's advantage-game probability, with the infinite deuce tail solved."""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Point probabilities must be in[0,1]")
    q = 1 - p
    result = p**4 * (1 + 4*q + 10*q*q) + 20*p**3*q**3 * p*p / (p*p + q*q)
    return float(result) if not result.shape else result


def _tiebreak(a, b, first_server):
    """Vectorized backwards DP, where0=A serves first and1=B serves first."""
    win_two, lose_two = a * (1 - b), (1 - a) * b
    grid = np.zeros((8, 8, len(a)))
    grid[7, :6] = 1.
    grid[6, 6] = win_two / (win_two + lose_two)
    for aw in range(6, -1, -1):
        for bw in range(6, -1, -1):
            if aw == bw == 6:
                continue
            # A-B-B-A-A-B-B-... relative to the first tiebreak server.
            relative_server = 0 if (aw + bw) % 4 in (0, 3) else 1
            server = first_server ^ relative_server
            point = a if server == 0 else 1 - b
            grid[aw, bw] = point * grid[aw + 1, bw] + (1 - point) * grid[aw, bw + 1]
    return grid[0, 0]


def tiebreak_probability(p_a, p_b, first_server=0):
    """A's probability in a standard 7-point tiebreak, from0--0."""
    if first_server not in (0, 1):
        raise ValueError("first_server must be0(A) or1(B)")
    a, b, shape = _inputs(p_a, p_b)
    return _restore(_tiebreak(a, b, first_server), shape)


def _set_transitions(a, b):
    """Joint set winner/next-server probabilities, conditional on first server.

    Output dimensions: first_server, winner(0=A), next_server, vectorized row.
    Retaining the next-server state avoids assuming independent identically
    distributed set winners when propagating a multi-set match.
    """
    holds = (game_probability(a), game_probability(b))
    kernel = np.zeros((2, 2, 2, len(a)))
    for first in (0, 1):
        mass = np.zeros((8, 8, len(a)))
        mass[0, 0] = 1.
        tie = _tiebreak(a, b, first)
        for total in range(13):
            for aw in range(max(0, total - 7), min(total, 7) + 1):
                bw = total - aw
                current = mass[aw, bw]
                if aw == bw == 6:
                    # The first tiebreak receiver serves the following set.
                    kernel[first, 0, 1 - first] += current * tie
                    kernel[first, 1, 1 - first] += current * (1 - tie)
                elif max(aw, bw) >= 6 and abs(aw - bw) >= 2:
                    winner = 0 if aw > bw else 1
                    next_server = first ^ (total % 2)
                    kernel[first, winner, next_server] += current
                elif aw < 7 and bw < 7:
                    server = first ^ (total % 2)
                    win_game = holds[0] if server == 0 else 1 - holds[1]
                    mass[aw + 1, bw] += current * win_game
                    mass[aw, bw + 1] += current * (1 - win_game)
    return kernel


def _match_from_kernel(kernel, sets_needed):
    mass = np.zeros((sets_needed + 1, sets_needed + 1, 2, kernel.shape[-1]))
    # First server is not an observed input: average the two possibilities.
    mass[0, 0, :] = .5
    for total in range(2 * sets_needed - 1):
        for aw in range(sets_needed):
            bw = total - aw
            if not 0 <= bw < sets_needed:
                continue
            for server in (0, 1):
                for next_server in (0, 1):
                    mass[aw + 1, bw, next_server] += mass[aw, bw, server] * kernel[server, 0, next_server]
                    mass[aw, bw + 1, next_server] += mass[aw, bw, server] * kernel[server, 1, next_server]
    return mass[sets_needed, :sets_needed].sum(axis=(0, 1))


def set_probability(p_a, p_b):
    """A's standardized set probability, averaging the unknown first server."""
    a, b, shape = _inputs(p_a, p_b)
    kernel = _set_transitions(a, b)
    return _restore(.5 * kernel[:, 0].sum(axis=(0, 1)), shape)


def match_probability(p_a, p_b, best_of=3):
    """A's standardized IID match probability; best_of may be3/5 row-wise."""
    a0, b0, best0 = np.broadcast_arrays(np.asarray(p_a), np.asarray(p_b), np.asarray(best_of))
    a, b, shape = _inputs(a0, b0)
    best = best0.ravel()
    if not np.isin(best, [3, 5]).all():
        raise ValueError("Only best-of3 and best-of5 matches are supported")
    kernel = _set_transitions(a, b)
    result = np.zeros(len(a))
    for number in (3, 5):
        mask = best == number
        if mask.any():
            result[mask] = _match_from_kernel(kernel[..., mask], number // 2 + 1)
    return _restore(result, shape)


def _logit(values):
    values = np.clip(values, 1e-8, 1 - 1e-8)
    return np.log(values) - np.log1p(-values)


def build_paper_point_features(targets, features):
    """Return12 odd probability-logit columns from existing past-only estimates.

    Reads target season only for the holdout guard, never labels, scores or stats.
    Point probabilities are reconstructed from their symmetric sum and odd
    difference; all target/feature indices must agree exactly.
    """
    if len(targets) == 0 or not targets.index.equals(features.index):
        raise ValueError("Target and feature indices must match and be nonempty")
    if not pd.to_numeric(targets.season).between(2009, 2023).all():
        raise ValueError("Development targets must be2009--2023; final holdout is locked")
    flag = features["format_best_of_five"].to_numpy(dtype=float)
    if not np.isin(flag, [0, 1]).all():
        raise ValueError("Known format flag must be binary")
    best = np.where(flag == 1, 5, 3)
    result = {}
    for half in HALF_LIVES:
        for scope in SCOPES:
            source = f"adjpoints_h{half}_{scope}_serve_prob"
            total = features[source + "_sum"].to_numpy(dtype=float)
            difference = features[source + "_diff"].to_numpy(dtype=float)
            a, b, _ = _inputs((total + difference) / 2, (total - difference) / 2)
            kernel = _set_transitions(a, b)
            set_win = .5 * kernel[:, 0].sum(axis=(0, 1))
            tie_win = .5 * (_tiebreak(a, b, 0) + _tiebreak(a, b, 1))
            match_win = np.zeros(len(a))
            for number in (3, 5):
                mask = best == number
                if mask.any():
                    match_win[mask] = _match_from_kernel(kernel[..., mask], number // 2 + 1)
            prefix = f"paperpoint_h{half}_{scope}"
            for label, values in (("match", match_win), ("set", set_win), ("tiebreak", tie_win)):
                result[f"{prefix}_{label}_logit_diff"] = _logit(values)
    frame = pd.DataFrame(result, index=targets.index)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite paper point features")
    frame.attrs.update({"feature_version": FEATURE_VERSION,
                        "scoring_rules": "advantage games;7-point win-by-two tiebreak at6--6 in every set",
                        "historical_deciding_set_rules_reconstructed": False,
                        "first_server": "average A/B; carry next server across sets",
                        "assumptions": "IID points conditional on server; constant skills through match",
                        "point_estimator": "existing opponent-adjusted dynamic point skills, not paper estimator",
                        "primary_equations": "Spanias2015 thesis ch3 pp60--66 eq3.1--3.7",
                        "source_url": "https://www.doc.ic.ac.uk/~wjk/publications/spanias-2015.pdf",
                        "related_paper_doi": "10.1093/imaman/dps010"})
    return frame

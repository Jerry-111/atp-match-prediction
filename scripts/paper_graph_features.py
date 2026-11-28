"""Temporal Bonacich features adapted from Arcagni, Candila & Grassi (2023).

Faithful mathematical core: loser-to-winner adjacency; hyperbolic historical
weights 1/(1+age_days/365); Euclidean-normalized Perron vector of W transpose.
Documented adaptations: T-21 tournament-date snapshots, past-only vertex sets,
uniform positive pseudolinks for reducibility, neutral unseen-player priors,
and a separate same-surface graph. This is not a replication of paper scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs

from scripts.advanced_features import _date

FEATURE_VERSION = "arcagni-bonacich-1.0"
ALPHA_DAYS = 365.0
PSEUDOLINK_FRACTION = 0.01
LOG_PRIOR_FRACTION = 0.05
LAG_DAYS = 21
PAPER_DOI = "10.1007/s10479-022-04594-7"


def hyperbolic_weights(ages, alpha=ALPHA_DAYS):
    ages = np.asarray(ages, dtype=float)
    if not np.isfinite(ages).all() or np.any(ages < 0) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Graph aging requires nonnegative finite ages and positive alpha")
    return 1.0 / (1.0 + ages / alpha)


def bonacich_scores(adjacency, regularization=PSEUDOLINK_FRACTION):
    """Return incoming Perron centrality, L2-normalized, from loser-row W.

    Production uses W.T + epsilon/n * 11.T, where epsilon is the fixed
    pseudolink fraction of mean weighted out-strength. The operation stays
    sparse via a rank-one LinearOperator. Set regularization=0 for the exact
    paper core on graphs with a well-defined positive principal eigenvector.
    """
    matrix = csr_matrix(adjacency, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Adjacency must be square, finite and nonnegative")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Graph regularization must be finite and nonnegative")
    n = matrix.shape[0]
    if not n:
        return np.array([], dtype=float)
    total_weight = float(matrix.sum())
    if not total_weight:
        return np.full(n, 1.0 / np.sqrt(n))
    epsilon = regularization * total_weight / n
    incoming = matrix.T.tocsr()

    def multiply(vector):
        return incoming @ vector + (epsilon / n) * np.sum(vector)

    if n <= 3:
        dense = incoming.toarray() + epsilon / n
        values, vectors = np.linalg.eig(dense)
        selected = int(np.argmax(values.real))
        value, vector = values[selected], vectors[:, selected]
    else:
        operator = LinearOperator((n, n), matvec=multiply, dtype=float)
        values, vectors = eigs(operator, k=1, which="LR", v0=np.full(n, 1.0 / np.sqrt(n)),
                               tol=1e-10, maxiter=2000)
        value, vector = values[0], vectors[:, 0]
    if abs(value.imag) > 1e-7 or np.max(np.abs(vector.imag)) > 1e-7 or value.real <= 0:
        raise ValueError("Graph does not have a usable positive real principal eigenpair")
    vector = vector.real
    if vector.sum() < 0:
        vector = -vector
    if np.min(vector) < -1e-8:
        raise ValueError("Principal centrality vector contains negative coordinates")
    vector = np.maximum(vector, 0)
    vector /= np.linalg.norm(vector)
    residual = np.linalg.norm(multiply(vector) - value.real * vector)
    if residual > 1e-7 * max(1, abs(value.real)):
        raise ValueError("Graph eigensolver residual exceeds tolerance")
    return vector


@dataclass
class GraphSnapshot:
    players: dict
    centrality: np.ndarray
    wins: np.ndarray
    losses: np.ndarray

    def features(self, player_a, player_b, stem):
        # A missing player's centrality is the current observed-player mean,
        # not zero or a future-learned skill. Both unseen players are neutral.
        mean = float(self.centrality.mean()) if len(self.centrality) else 1.0
        prior = LOG_PRIOR_FRACTION * mean
        score, win, loss, seen = [], [], [], []
        for player in (str(player_a), str(player_b)):
            position = self.players.get(player)
            seen.append(int(position is not None))
            score.append(float(self.centrality[position]) if position is not None else mean)
            win.append(float(self.wins[position]) if position is not None else 0.0)
            loss.append(float(self.losses[position]) if position is not None else 0.0)
        return {
            f"{stem}_bscore_diff": score[0]-score[1],
            f"{stem}_log_bscore_diff": float(np.log(score[0]+prior)-np.log(score[1]+prior)),
            f"{stem}_win_strength_log_diff": log1p(win[0])-log1p(win[1]),
            f"{stem}_loss_strength_log_diff": log1p(loss[0])-log1p(loss[1]),
            f"{stem}_observed_diff": seen[0]-seen[1],
            f"{stem}_observed_sum": sum(seen),
        }


class TemporalGraph:
    """Store eligible historical edges; query each target cutoff only once."""

    def __init__(self, inventory):
        mask = inventory.source.eq("tour") & pd.to_numeric(inventory.season).between(2006, 2023)
        records, seen = [], set()
        for row in inventory.loc[mask].to_dict("records"):
            identifier = str(row["match_id"])
            if identifier in seen:
                raise ValueError(f"Duplicate graph history match: {identifier}")
            seen.add(identifier)
            a, b = str(row["player_a_id"]), str(row["player_b_id"])
            won, surface = float(row["a_won"]), str(row["surface"])
            if a == b or won not in (0.0, 1.0) or surface not in ("Hard", "Clay", "Grass"):
                raise ValueError(f"Invalid graph history row: {identifier}")
            winner, loser = (a, b) if won else (b, a)
            records.append((_date(row["tourney_date"]).toordinal(), identifier, loser, winner, surface))
        records.sort()
        self.days = np.array([r[0] for r in records], dtype=np.int64)
        self.losers = np.array([r[2] for r in records], dtype=object)
        self.winners = np.array([r[3] for r in records], dtype=object)
        self.surfaces = np.array([r[4] for r in records], dtype=object)

    def snapshot(self, cutoff_ordinal, surface=None):
        end = np.searchsorted(self.days, cutoff_ordinal, side="right")
        days, losers, winners = self.days[:end], self.losers[:end], self.winners[:end]
        if surface is not None:
            mask = self.surfaces[:end] == surface
            days, losers, winners = days[mask], losers[mask], winners[mask]
        if not len(days):
            empty = np.array([], dtype=float)
            return GraphSnapshot({}, empty, empty, empty)
        # Crucially this roster is built AFTER the cutoff, excluding player IDs
        # first appearing in future matches and even unseen target participants.
        players, positions = np.unique(np.concatenate([losers, winners]), return_inverse=True)
        count, n = len(days), len(players)
        source, destination = positions[:count], positions[count:]
        weight = hyperbolic_weights(cutoff_ordinal-days)
        matrix = coo_matrix((weight, (source, destination)), shape=(n,n)).tocsr()
        scores = bonacich_scores(matrix)
        return GraphSnapshot({player:i for i,player in enumerate(players)}, scores,
                             np.asarray(matrix.sum(axis=0)).ravel(), np.asarray(matrix.sum(axis=1)).ravel())


def build_paper_graph_features(targets, inventory):
    """Return twelve finite graph features, preserving exact target index/order.

    Uses expanded 2009--2023 development targets. All *_diff columns reverse
    sign under A/B exchange and all other columns remain unchanged.
    """
    if targets.empty or not pd.to_numeric(targets.season).between(2009, 2023).all():
        raise ValueError("Graph development targets must be nonempty 2009--2023; final holdout is locked")
    if not targets.surface.isin(["Hard", "Clay", "Grass"]).all():
        raise ValueError("Unknown target graph surface")
    if targets.player_a_id.astype(str).eq(targets.player_b_id.astype(str)).any():
        raise ValueError("Graph target players must differ")
    state = TemporalGraph(inventory)
    cutoffs = np.array([_date(d).toordinal()-LAG_DAYS for d in targets.tourney_date])
    records = targets.to_dict("records")
    output = [None] * len(records)
    for cutoff in sorted(set(cutoffs)):
        positions = np.flatnonzero(cutoffs == cutoff)
        overall = state.snapshot(cutoff)
        surfaces = {records[p]["surface"] for p in positions}
        by_surface = {surface: state.snapshot(cutoff, surface) for surface in sorted(surfaces)}
        for position in positions:
            row = records[position]
            a, b = row["player_a_id"], row["player_b_id"]
            output[position] = {**overall.features(a,b,"paper_graph"),
                                **by_surface[row["surface"]].features(a,b,"paper_surface_graph")}
    result = pd.DataFrame(output, index=targets.index)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite paper graph features")
    result.attrs.update(feature_version=FEATURE_VERSION, source_doi=PAPER_DOI,
                        alpha_days=ALPHA_DAYS, pseudolink_fraction=PSEUDOLINK_FRACTION,
                        log_prior_fraction=LOG_PRIOR_FRACTION, history_lag_days=LAG_DAYS,
                        odd_columns=[c for c in result if c.endswith("_diff")],
                        reproduction_status="Exact weighting/eigenvector core; regularized temporal/surface adaptation")
    return result

"""Joint binary residual states with frozen shared features at as-of queries.

No file access, empirical runner, model selection or validation scoring occurs
here. Quarterly smoothing is restricted to the currently admissible prefix.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from scripts.round2_learning import AntisymmetricSplineClassifier
from scripts.train_distillation_round7 import validate_rows

VERSION = 'joint-binary-residual-quarter-v1'
INITIAL_SD = .35
INNOVATION_SD = .15
C = .01
HALF_LIFE = 3.
LAG_DAYS = 21
GRADIENT_TOLERANCE = 1e-8
SOLVER_OPTIONS = dict(maxiter=5000, maxcor=30, ftol=0., gtol=1e-10)
METADATA = ('match_id', 'season', 'tourney_date', 'player_a_id', 'player_b_id')


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _spec(spec):
    value = deepcopy(spec)
    p = value.get('params', {})
    if (value.get('kind') != 'odd_spline' or p.get('C') != C
            or p.get('half_life') != HALF_LIFE or p.get('shape_weight') != .3
            or p.get('interaction_weight') != .3):
        raise ValueError('The fixed original R5 spline specification is required')
    json.dumps(value, allow_nan=False)
    return value


def _metadata(rows, *, fitting=False, query=False):
    if (not isinstance(rows, pd.DataFrame) or rows.columns.duplicated().any()
            or not rows.index.is_unique or any(c not in rows for c in METADATA)):
        raise ValueError('Unique aligned match metadata required')
    if query and 'a_won' in rows:
        raise ValueError('Query metadata must not contain target labels')
    # This selects metadata before examining any history outcomes/features.
    meta = rows.loc[:, list(METADATA)].copy()
    dates = validate_rows(meta, fitting=fitting)
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Nominal dates must be timezone-naive whole civil days')
    for col in ('match_id', 'player_a_id', 'player_b_id'):
        if meta[col].isna().any() or meta[col].astype(str).str.strip().isin(('', 'nan', 'None')).any():
            raise ValueError('Nonmissing match and player identities required')
        meta[col] = meta[col].astype(str)
    if meta.match_id.duplicated().any() or meta.player_a_id.eq(meta.player_b_id).any():
        raise ValueError('Unique matches and distinct players required')
    meta['season'] = pd.to_numeric(meta.season).astype(int)
    meta['tourney_date'] = dates
    return meta


def _alignment(x, rows):
    if not isinstance(x, pd.DataFrame) or not x.index.equals(rows.index):
        raise ValueError('Feature and metadata row indices must be identical')


def _labels(rows):
    if 'a_won' not in rows:
        raise ValueError('Eligible history/training labels required')
    y = rows.a_won.to_numpy()
    if y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError('Labels must be finite binary values without coercion')
    return y.astype(float)


def _fingerprint(x, meta, y, columns):
    values = x.loc[:, columns].to_numpy(dtype='<f8')
    if not np.isfinite(values).all():
        raise ValueError('Eligible model features must be finite')
    h = hashlib.sha256()
    records = meta.assign(tourney_date=meta.tourney_date.dt.strftime('%Y-%m-%d')).values.tolist()
    h.update(json.dumps(records, separators=(',', ':'), ensure_ascii=False).encode())
    h.update(np.ascontiguousarray(values).tobytes())
    h.update(np.asarray(y, dtype='<f8').tobytes())
    h.update(json.dumps(list(columns), separators=(',', ':')).encode())
    return h.hexdigest()


def _transform_hash(model):
    def canonical(value):
        if isinstance(value, np.ndarray):
            return dict(shape=list(value.shape), dtype=value.dtype.str,
                        data=np.ascontiguousarray(value).tobytes().hex())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(v) for v in value]
        return value
    names = ('spec', 'odd_scale_', 'odd_indices_', 'spline_scale_',
             'context_mean_', 'context_scale_', 'interaction_scale_', 'design_width_')
    state = {key: getattr(model, key) for key in names}
    state['splines_'] = {name: dict(params=spline.get_params(deep=False),
        n_features_in=spline.n_features_in_, n_features_out=spline.n_features_out_,
        bsplines=[dict(t=b.t, c=b.c, k=b.k, axis=b.axis, extrapolate=b.extrapolate)
                  for b in spline.bsplines_]) for name, spline in model.splines_.items()}
    return _digest(json.dumps(canonical(state), sort_keys=True, separators=(',', ':'),
                              allow_nan=False).encode())


def fit_shared_control(base_spec, x_train, rows_train):
    """One unchanged R5 fit, plus provenance attributes; no second fit."""
    spec = _spec(base_spec)
    _alignment(x_train, rows_train)
    meta = _metadata(rows_train, fitting=True)
    y = _labels(rows_train)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = AntisymmetricSplineClassifier(spec).fit(x_train, y, meta.season.to_numpy())
    if (any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or np.max(model.estimator_.n_iter_) >= model.estimator_.max_iter
            or not np.isfinite(model.estimator_.coef_).all()):
        raise RuntimeError('The unchanged shared control failed to converge')
    model.residual_training_fingerprint_ = _fingerprint(x_train, meta, y, spec['columns'])
    model.residual_training_match_ids_ = tuple(meta.match_id)
    model.residual_transform_sha256_ = _transform_hash(model)
    return model


@dataclass
class StateDesign:
    matrix: csr_matrix
    penalty_matrix: csr_matrix
    keys: tuple
    node_dates: tuple
    prior_links: tuple
    latest_indices: dict


def build_state_design(meta, structure):
    """Metadata-only sparse player differences and proper Gaussian priors."""
    if structure not in ('static', 'dynamic') or meta.empty:
        raise ValueError('Nonempty static/dynamic state population required')
    dates = pd.to_datetime(meta.tourney_date)
    quarters = dates.dt.to_period('Q')
    tags = np.repeat('static', len(meta)) if structure == 'static' else quarters.astype(str).to_numpy()
    appearances = [(str(player), str(tag)) for a, b, tag in
                   zip(meta.player_a_id, meta.player_b_id, tags) for player in (a, b)]
    keys = tuple(sorted(set(appearances)))
    lookup = {key: i for i, key in enumerate(keys)}
    nodes = np.array([lookup[key] for key in appearances], dtype=int)
    matrix = csr_matrix((np.tile((1., -1.), len(meta)),
                         (np.repeat(np.arange(len(meta)), 2), nodes)), shape=(len(meta), len(keys)))
    first_days = {}
    for a, b, day in zip(meta.player_a_id, meta.player_b_id, dates):
        for player in (a, b):
            first_days[player] = min(first_days.get(player, day), day)
    node_dates = tuple((first_days[p] if tag == 'static' else pd.Period(tag, freq='Q').start_time)
                       .strftime('%Y-%m-%d') for p, tag in keys)
    ri, ci, vv, links, latest = [], [], [], [], {}
    for i, (player, tag) in enumerate(keys):
        if player not in latest:
            ri.append(i); ci.append(i); vv.append(1 / INITIAL_SD)
        else:
            previous = latest[player]
            elapsed = (pd.Timestamp(node_dates[i]) - pd.Timestamp(node_dates[previous])).days / 365.25
            if elapsed <= 0:
                raise ValueError('Dynamic state links must advance calendar quarters')
            inverse_sd = 1 / (INNOVATION_SD * np.sqrt(elapsed))
            ri.extend((i, i)); ci.extend((previous, i)); vv.extend((-inverse_sd, inverse_sd))
            links.append((previous, i, float(elapsed)))
        latest[player] = i
    penalty = csr_matrix((vv, (ri, ci)), shape=(len(keys), len(keys)))
    return StateDesign(matrix, penalty, keys, node_dates, tuple(links), latest)


def sample_weights(meta):
    anchor = int(meta.season.max())
    return 2. ** (-(anchor - meta.season.to_numpy()) / HALF_LIFE), anchor


def map_objective(parameters, design, states, y, weights, *, fixed_beta=None):
    """Unscaled sum-NLL objective/gradient; fixed-beta constant is omitted."""
    if fixed_beta is None:
        width = design.shape[1]
        beta, u = parameters[:width], parameters[width:]
        offset = design @ beta
    else:
        beta, u = fixed_beta, parameters
        offset = design  # Already computed fixed shared offsets, not new features.
    eta = offset + states.matrix @ u
    prior = states.penalty_matrix @ u
    loss = np.dot(weights, np.logaddexp(0., eta) - y * eta) + .5 * np.dot(prior, prior)
    errors = weights * (expit(eta) - y)
    ug = np.asarray(states.matrix.T @ errors + states.penalty_matrix.T @ prior)
    if fixed_beta is None:
        loss += .5 * np.dot(beta, beta) / C
        return float(loss), np.r_[design.T @ errors + beta / C, ug]
    return float(loss), ug


def _solve(design, states, y, weights, *, beta_initial=None, fixed_beta=None):
    n = len(y)
    start = np.zeros(len(states.keys))
    if fixed_beta is None:
        start = np.r_[np.asarray(beta_initial, dtype=float), start]
    def scaled(parameters):
        value, gradient = map_objective(parameters, design, states, y, weights, fixed_beta=fixed_beta)
        return value / n, gradient / n
    begin = time.perf_counter()
    result = minimize(scaled, start, jac=True, method='L-BFGS-B',
                      options=dict(SOLVER_OPTIONS))
    elapsed = time.perf_counter() - begin
    value, gradient = scaled(result.x)
    max_gradient = float(np.max(np.abs(gradient)))
    diagnostics = dict(optimizer='L-BFGS-B', success=bool(result.success), status=int(result.status),
        message=str(result.message), iterations=int(result.nit), function_evaluations=int(result.nfev),
        seconds=float(elapsed), rows=n, parameters=len(start), objective_scaled=float(value),
        objective_unscaled=float(value*n), max_gradient_scaled=max_gradient,
        max_gradient_unscaled=max_gradient*n, accepted_gradient_tolerance=GRADIENT_TOLERANCE,
        initial_objective_scaled=float(scaled(start)[0]),
        mode='joint_beta_and_u' if fixed_beta is None else 'fixed_beta_u_only',
        shared_ridge_constant_omitted=fixed_beta is not None,
        solver_options=dict(SOLVER_OPTIONS))
    if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(value)
            or not np.isfinite(gradient).all() or max_gradient > GRADIENT_TOLERANCE):
        raise RuntimeError('Residual MAP did not converge: ' + json.dumps(diagnostics))
    return result.x, diagnostics


@dataclass
class StateSnapshot:
    cutoff: str
    structure: str
    keys: tuple
    coefficients: np.ndarray
    node_dates: tuple
    prior_links: tuple
    latest_indices: dict
    eligible_match_ids: tuple
    history_sha256: str
    diagnostics: dict

    def differences(self, meta):
        def value(player):
            index = self.latest_indices.get(str(player))
            return 0. if index is None else float(self.coefficients[index])
        return np.array([value(a) - value(b) for a, b in zip(meta.player_a_id, meta.player_b_id)])


class ResidualStateModel:
    """Shared R5 transform with static or quarterly Gaussian player residuals."""

    def __init__(self, base_spec, structure):
        self.base_spec = _spec(base_spec)
        if structure not in ('static', 'dynamic'):
            raise ValueError('Only the two registered residual structures are allowed')
        self.structure = structure

    def _assert_frozen(self):
        if (_transform_hash(self.shared_model_) != self.transform_sha256_
                or _digest(np.asarray(self.beta_, dtype='<f8').tobytes()) != self.beta_sha256_
                or self.structure != self.training_metadata_['structure']
                or self.base_spec != self.shared_model_.spec
                or self.training_metadata_['initial_sd'] != INITIAL_SD
                or self.training_metadata_['innovation_sd'] != INNOVATION_SD
                or self.training_metadata_['C'] != C
                or self.training_metadata_['half_life'] != HALF_LIFE):
            raise ValueError('The frozen transform, coefficients, structure or priors changed')

    def _design(self, x):
        return self.shared_model_._design(self.shared_model_._input(x))

    def fit(self, x_train, rows_train, *, shared_model):
        _alignment(x_train, rows_train)
        meta = _metadata(rows_train, fitting=True)
        y = _labels(rows_train)
        binding = _fingerprint(x_train, meta, y, self.base_spec['columns'])
        if (not isinstance(shared_model, AntisymmetricSplineClassifier)
                or shared_model.spec != self.base_spec
                or getattr(shared_model, 'residual_training_fingerprint_', None) != binding
                or getattr(shared_model, 'residual_transform_sha256_', None) != _transform_hash(shared_model)):
            raise ValueError('Use the bound shared control fitted on exactly these training rows')
        self.shared_model_ = deepcopy(shared_model)
        self.transform_sha256_ = _transform_hash(self.shared_model_)
        self.training_match_ids_ = tuple(meta.match_id)
        self.training_fingerprint_ = binding
        self.training_max_date_ = meta.tourney_date.max()
        self.prediction_season_ = int(meta.season.max()) + 1
        states = build_state_design(meta, self.structure)
        weights, anchor = sample_weights(meta)
        design = self._design(x_train)
        coefficients, diagnostics = _solve(design, states, y, weights,
            beta_initial=self.shared_model_.estimator_.coef_.ravel())
        self.beta_ = coefficients[:design.shape[1]].copy()
        self.beta_sha256_ = _digest(np.asarray(self.beta_, dtype='<f8').tobytes())
        diagnostics.update(weight_anchor_season=anchor,
            max_eligible_date=self.training_max_date_.strftime('%Y-%m-%d'), states=len(states.keys))
        self.fit_diagnostics_ = diagnostics
        self.training_metadata_ = dict(version=VERSION, structure=self.structure,
            rows=len(meta), prediction_season=self.prediction_season_,
            match_ids_sha256=_digest('\n'.join(self.training_match_ids_).encode()),
            training_fingerprint=binding, transform_sha256=self.transform_sha256_, beta_sha256=self.beta_sha256_,
            initial_sd=INITIAL_SD, innovation_sd=INNOVATION_SD, C=C, half_life=HALF_LIFE,
            history_first_season=2009, allowed_history_last_season=2023,
            max_eligible_date=diagnostics['max_eligible_date'], weight_anchor_season=anchor)
        self.training_state_ = StateSnapshot(diagnostics['max_eligible_date'], self.structure,
            states.keys, coefficients[design.shape[1]:].copy(), states.node_dates, states.prior_links,
            states.latest_indices, self.training_match_ids_, binding, dict(diagnostics))
        self.state_snapshots_ = {}
        self._assert_frozen()
        return self

    def _queries(self, x_query, query_metadata):
        self._assert_frozen()
        _alignment(x_query, query_metadata)
        meta = _metadata(query_metadata, query=True)
        if not meta.season.eq(self.prediction_season_).all():
            raise ValueError('Query season must be the frozen outer prediction year')
        cutoffs = meta.tourney_date - pd.Timedelta(days=LAG_DAYS)
        if (cutoffs < self.training_max_date_).any():
            raise ValueError('Global training includes labels later than a query T-minus-21 cutoff')
        return meta, cutoffs

    def _training_history_check(self, x_history, rows_history, meta):
        by_id = pd.Series(np.arange(len(meta)), index=meta.match_id)
        if any(key not in by_id for key in self.training_match_ids_):
            raise ValueError('History must retain the complete global training population')
        positions = by_id.loc[list(self.training_match_ids_)].to_numpy()
        selected_meta = meta.iloc[positions]
        y = _labels(rows_history.iloc[positions])
        binding = _fingerprint(x_history.iloc[positions], selected_meta, y, self.base_spec['columns'])
        if binding != self.training_fingerprint_:
            raise ValueError('The bound pre-year training history changed')

    def predict_asof(self, x_query, query_metadata, x_history, rows_history):
        """Condition u using eligible rows only; store auditable cutoff snapshots."""
        meta, cutoffs = self._queries(x_query, query_metadata)
        _alignment(x_history, rows_history)
        history_meta = _metadata(rows_history)
        # The original training prefix is guaranteed eligible by _queries.
        self._training_history_check(x_history, rows_history, history_meta)
        for cutoff in sorted(cutoffs.unique()):
            cutoff = pd.Timestamp(cutoff)
            key = cutoff.strftime('%Y-%m-%d')
            positions = np.flatnonzero(history_meta.tourney_date.le(cutoff).to_numpy())
            eligible = history_meta.iloc[positions]
            order = np.lexsort((eligible.match_id.to_numpy(), eligible.tourney_date.to_numpy()))
            positions = positions[order]
            eligible = history_meta.iloc[positions]
            # Only after the date mask do outcome and feature values enter.
            y = _labels(rows_history.iloc[positions])
            x = x_history.iloc[positions]
            binding = _fingerprint(x, eligible, y, self.base_spec['columns'])
            if key in self.state_snapshots_:
                if self.state_snapshots_[key].history_sha256 != binding:
                    raise ValueError('An existing cutoff snapshot received different eligible history')
                continue
            states = build_state_design(eligible, self.structure)
            weights, anchor = sample_weights(eligible)
            offset = self._design(x) @ self.beta_
            coefficients, diagnostics = _solve(offset, states, y, weights, fixed_beta=self.beta_)
            diagnostics.update(weight_anchor_season=anchor, states=len(states.keys),
                max_eligible_date=eligible.tourney_date.max().strftime('%Y-%m-%d'),
                eligible_match_ids_sha256=_digest('\n'.join(eligible.match_id).encode()),
                transform_sha256=self.transform_sha256_, beta_sha256=self.beta_sha256_,
                zero_initialization=True, conditional_smoothing_within_cutoff=True)
            self.state_snapshots_[key] = StateSnapshot(key, self.structure, states.keys,
                coefficients.copy(), states.node_dates, states.prior_links, states.latest_indices,
                tuple(eligible.match_id), binding, diagnostics)
        self._assert_frozen()
        return self.predict_from_snapshots(x_query, query_metadata)

    def predict_from_snapshots(self, x_query, query_metadata):
        """Reload-only prediction; never optimize or inspect historical labels."""
        meta, cutoffs = self._queries(x_query, query_metadata)
        eta = self._design(x_query) @ self.beta_
        for cutoff in sorted(cutoffs.unique()):
            key = pd.Timestamp(cutoff).strftime('%Y-%m-%d')
            if key not in self.state_snapshots_:
                raise ValueError('No saved as-of snapshot for cutoff ' + key)
            mask = cutoffs.eq(cutoff).to_numpy()
            eta[mask] += self.state_snapshots_[key].differences(meta.iloc[np.flatnonzero(mask)])
        return np.clip(expit(eta), 1e-8, 1 - 1e-8)

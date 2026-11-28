"""Independent arithmetic and temporal checks for the Round 12 state experiment.

This module does not import the experiment's optimizer or selection helpers.
It reconstructs the registered checks from stored inputs; optimizer replay and
the original historical feature construction are separate, explicitly limited
claims. The artifact integration is added after the runner schema is frozen.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from math import ceil, fsum, log, log1p
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import expit

STRUCTURES = ('shared_only', 'shared_static_residual', 'shared_dynamic_residual')
REFERENCE = 'previous_probability_recommendation'
ROOT = Path(__file__).resolve().parents[1]
META = ['match_id', 'season', 'tourney_date', 'player_a_id', 'player_b_id']


def independent_metrics(y, probability):
    labels, p = np.asarray(y), np.asarray(probability, dtype=float)
    if (labels.ndim != 1 or p.shape != labels.shape or not len(labels)
            or not np.isin(labels, (0, 1)).all() or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError('Aligned nonempty binary outcomes and finite probabilities required')
    losses = [(-log(float(v)) if v > 0 else float('inf')) if a else
              (-log1p(-float(v)) if v < 1 else float('inf')) for a, v in zip(labels, p)]
    correct = int(np.sum((p >= .5) == labels))
    return dict(matches=len(labels), correct=correct, accuracy=correct / len(labels),
                mean_log_loss=fsum(losses) / len(labels))


def independent_selection(rows, predictions, reference=REFERENCE):
    """Recompute early choice, integer gate and pooled choice without runner code."""
    names = (*STRUCTURES, reference)
    if (not rows.index.equals(predictions.index) or rows.index.duplicated().any()
            or rows.match_id.duplicated().any() or predictions.columns.duplicated().any()
            or set(predictions) != set(names)):
        raise ValueError('Exact aligned rows and four selection candidates required')
    years = pd.to_numeric(rows.season, errors='raise').to_numpy()
    if set(years) != {2020, 2021, 2022}:
        raise ValueError('Only complete 2020--2022 OOF seasons are selection inputs')
    labels = rows.a_won.to_numpy()
    for name in names:
        independent_metrics(labels, predictions[name].to_numpy())

    def choose(mask):
        metrics = {name: independent_metrics(labels[mask], predictions[name].to_numpy()[mask])
                   for name in names}
        allowed = [name for name in names if metrics[name]['mean_log_loss'] <=
                   metrics[reference]['mean_log_loss'] + .005]
        selected = min(allowed, key=lambda name: (-metrics[name]['correct'],
                       metrics[name]['mean_log_loss'], name))
        return selected, metrics

    early, early_metrics = choose(np.isin(years, [2020, 2021]))
    pooled, pooled_metrics = choose(np.ones(len(years), dtype=bool))
    gate_mask = years == 2022
    gate_metrics = {name: independent_metrics(labels[gate_mask], predictions[name].to_numpy()[gate_mask])
                    for name in (early, reference)}
    count = int(gate_mask.sum())
    required = ceil(.005 * count)
    net = gate_metrics[early]['correct'] - gate_metrics[reference]['correct']
    loss_delta = gate_metrics[early]['mean_log_loss'] - gate_metrics[reference]['mean_log_loss']
    passed = bool(net >= required and loss_delta <= .005)
    return dict(early_selected=early, pooled_selected=pooled, gate_passed=passed,
                recommended=pooled if passed else reference, gate_matches=count,
                required_additional_correct=required, additional_correct=net,
                gate_loss_delta=loss_delta, early_metrics=early_metrics,
                pooled_metrics=pooled_metrics)


def metadata_dates(rows):
    """Validate metadata without touching outcome or feature columns."""
    required = {'match_id', 'season', 'tourney_date', 'player_a_id', 'player_b_id'}
    if (not required.issubset(rows) or rows.empty or rows.match_id.duplicated().any()
            or rows.index.duplicated().any()):
        raise ValueError('Unique nonempty development metadata required')
    seasons = pd.to_numeric(rows.season, errors='raise').to_numpy()
    if not np.isin(seasons, np.arange(2009, 2024)).all():
        raise ValueError('Only 2009--2023 development histories are permitted')
    dates = pd.to_datetime(rows.tourney_date, errors='raise')
    if (dates.isna().any() or getattr(dates.dt, 'tz', None) is not None
            or not dates.eq(dates.dt.normalize()).all()):
        raise ValueError('Finite nominal calendar dates required')
    actual_year = dates.dt.year.to_numpy()
    prior_december = ((actual_year == seasons - 1) & (dates.dt.month.to_numpy() == 12)
                      & (dates.dt.day.to_numpy() >= 25))
    if not ((actual_year == seasons) | prior_december).all():
        raise ValueError('Season/date mismatch outside legitimate opening weeks')
    if (rows.player_a_id.isna().any() or rows.player_b_id.isna().any()
            or rows.player_a_id.astype(str).eq(rows.player_b_id.astype(str)).any()):
        raise ValueError('Distinct nonmissing players required')
    return dates


def independent_outer_indices(rows, year):
    dates = metadata_dates(rows)
    if year not in (2020, 2021, 2022, 2023):
        raise ValueError('Unsupported outer year')
    query = rows.season.eq(year)
    if not query.any():
        raise ValueError('Missing outer prediction rows')
    cutoff = dates[query].min() - pd.Timedelta(days=21)
    mask = rows.season.lt(year) & dates.le(cutoff)
    return rows.index[mask], cutoff.date().isoformat()


def independent_prefix_indices(rows, cutoff):
    dates = metadata_dates(rows)
    timestamp = pd.Timestamp(cutoff)
    if timestamp.tz is not None or not 2009 <= timestamp.year <= 2023:
        raise ValueError('Development-only finite cutoff required')
    return rows.index[dates.le(timestamp)]


def quarter_start(value):
    day = pd.Timestamp(value).date()
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1).isoformat()


def independent_nodes(history, structure):
    """Canonical node/date mapping from eligible metadata, not saved vocabularies."""
    if structure not in ('static', 'dynamic'):
        raise ValueError('Unknown residual structure')
    dates = metadata_dates(history)
    observed = {}
    for a, b, timestamp in zip(history.player_a_id, history.player_b_id, dates):
        day = timestamp.date().isoformat()
        for player in (str(a), str(b)):
            key = (player, 'static' if structure == 'static' else quarter_start(day))
            observed[key] = min(observed.get(key, day), day)
    keys = sorted(observed)
    node_dates = [observed[key] if structure == 'static' else key[1] for key in keys]
    return keys, node_dates


def independent_state_prior(keys, node_dates, values, structure):
    """Gaussian chain penalty/gradient, independently reconstructed by player/date."""
    u = np.asarray(values, dtype=float)
    if (structure not in ('static', 'dynamic') or u.shape != (len(keys),)
            or len(keys) != len(node_dates) or len(set(keys)) != len(keys)
            or not np.isfinite(u).all()):
        raise ValueError('Unique aligned finite state nodes required')
    by_player = {}
    for i, (key, day) in enumerate(zip(keys, node_dates)):
        stamp = date.fromisoformat(str(day))
        if structure == 'dynamic' and quarter_start(stamp) != stamp.isoformat():
            raise ValueError('Dynamic nodes must be calendar-quarter starts')
        by_player.setdefault(str(key[0]), []).append((stamp, i))
    value, gradient, links = 0., np.zeros_like(u), []
    for nodes in by_player.values():
        nodes.sort()
        if structure == 'static' and len(nodes) != 1:
            raise ValueError('One static state per player required')
        first = nodes[0][1]
        value += .5 * u[first] ** 2 / .35 ** 2
        gradient[first] += u[first] / .35 ** 2
        for (before, i), (after, j) in zip(nodes, nodes[1:]):
            elapsed = (after - before).days / 365.25
            if elapsed <= 0:
                raise ValueError('Distinct increasing player quarters required')
            precision = 1 / (.15 ** 2 * elapsed)
            delta = u[j] - u[i]
            value += .5 * delta ** 2 * precision
            gradient[i] -= delta * precision
            gradient[j] += delta * precision
            links.append((i, j, elapsed))
    return float(value), gradient, links


def independent_design(control, features):
    """Reconstruct old spline design without calling _design/transform/predict."""
    spec = control.spec
    selected = features.loc[:, spec['columns']].to_numpy(dtype=float)
    if not np.isfinite(selected).all():
        raise ValueError('Finite required input columns required')
    odd = np.clip(features[spec['odd_columns']].to_numpy(float) / control.odd_scale_, -8, 8)
    blocks = [odd]
    shape = []
    for column in spec['spline_columns']:
        stored = control.splines_[column].bsplines_[0]
        # Explicit constant boundary extension, then direct B-spline basis.
        basis = BSpline(stored.t, stored.c, stored.k, extrapolate=False)
        low, high = stored.t[stored.k], stored.t[-stored.k - 1]
        values = odd[:, control.odd_indices_[column]]
        shape.append(.5 * (basis(np.clip(values, low, high)) - basis(np.clip(-values, low, high))))
    if shape:
        blocks.append(np.column_stack(shape) / control.spline_scale_ * spec['params']['shape_weight'])
    interaction = []
    for odd_name, context_name in spec['interactions']:
        context = np.clip((features[context_name].to_numpy(float) - control.context_mean_[context_name])
                          / control.context_scale_[context_name], -3, 3)
        interaction.append(odd[:, control.odd_indices_[odd_name]] * context)
    if interaction:
        blocks.append(np.column_stack(interaction) / control.interaction_scale_
                      * spec['params']['interaction_weight'])
    return np.column_stack(blocks)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close(actual, expected, tolerance=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Finite identically shaped audit arrays required')
    error = float(np.max(np.abs(a - b), initial=0.))
    if error > tolerance:
        raise AssertionError(f'Audit disagreement {error} exceeds {tolerance}')
    return error


def _rms(a):
    rms = np.sqrt(np.mean(np.asarray(a) ** 2, axis=0))
    return np.where(rms > 1e-10, rms, 1.)


def audit_transform(control, train):
    """Compare all fitted numerical scales with a fresh training-only calculation."""
    spec = control.spec
    rms = _rms(train[spec['odd_columns']].to_numpy(float))
    errors = [_close(control.odd_scale_, rms)]
    odd = np.clip(train[spec['odd_columns']].to_numpy(float) / rms, -8, 8)
    shape = []
    for column in spec['spline_columns']:
        stored = control.splines_[column].bsplines_[0]
        assert stored.k == 2
        errors.append(_close(stored.t, np.linspace(-6, 6, 9)))
        errors.append(_close(stored.c, np.eye(6)))
        basis = BSpline(np.linspace(-6, 6, 9), np.eye(6), 2, extrapolate=False)
        values = odd[:, spec['odd_columns'].index(column)]
        shape.append(.5 * (basis(np.clip(values, -3, 3)) - basis(np.clip(-values, -3, 3))))
    if shape:
        errors.append(_close(control.spline_scale_, _rms(np.column_stack(shape))))
    interactions = []
    for odd_name, context_name in spec['interactions']:
        values = train[context_name].to_numpy(float)
        center, scale = float(values.mean()), max(float(values.std()), 1e-10)
        errors += [_close(control.context_mean_[context_name], center),
                   _close(control.context_scale_[context_name], scale)]
        context = np.clip((values - center) / scale, -3, 3)
        interactions.append(odd[:, spec['odd_columns'].index(odd_name)] * context)
    if interactions:
        errors.append(_close(control.interaction_scale_, _rms(np.column_stack(interactions))))
    return max(errors)


def independent_fingerprint(x, rows, columns):
    meta = rows[META].copy()
    meta['season'] = pd.to_numeric(meta.season).astype(int)
    for name in ('match_id', 'player_a_id', 'player_b_id'):
        meta[name] = meta[name].astype(str)
    meta['tourney_date'] = pd.to_datetime(meta.tourney_date).dt.strftime('%Y-%m-%d')
    h = hashlib.sha256()
    h.update(json.dumps(meta.values.tolist(), separators=(',', ':'), ensure_ascii=False).encode())
    h.update(np.ascontiguousarray(x[columns].to_numpy(dtype='<f8')).tobytes())
    h.update(rows.a_won.to_numpy(dtype='<f8').tobytes())
    h.update(json.dumps(list(columns), separators=(',', ':')).encode())
    return h.hexdigest()


def independent_snapshot(snapshot, rows, design, beta, *, joint):
    """Check vocabularies, Gaussian links, likelihood and gradients independently."""
    keys, dates = independent_nodes(rows, snapshot.structure)
    stored = [(str(player), 'static' if snapshot.structure == 'static' else str(day))
              for (player, _), day in zip(snapshot.keys, snapshot.node_dates)]
    assert stored == keys and list(snapshot.node_dates) == dates
    if snapshot.structure == 'dynamic':
        assert all(str(pd.Period(day, freq='Q')) == tag
                   for (_, tag), day in zip(snapshot.keys, dates))
    else:
        assert all(tag == 'static' for _, tag in snapshot.keys)
    assert tuple(rows.match_id.astype(str)) == snapshot.eligible_match_ids
    prior, gradient, links = independent_state_prior(keys, dates, snapshot.coefficients, snapshot.structure)
    assert len(links) == len(snapshot.prior_links)
    for a, b in zip(links, snapshot.prior_links):
        assert a[:2] == b[:2]
        _close(a[2], b[2])
    latest = {player: i for i, (player, _) in enumerate(keys)}
    assert latest == snapshot.latest_indices
    lookup = {key: i for i, key in enumerate(keys)}
    tags = ['static'] * len(rows) if snapshot.structure == 'static' else [quarter_start(d) for d in rows.tourney_date]
    ia = np.array([lookup[str(p), q] for p, q in zip(rows.player_a_id, tags)])
    ib = np.array([lookup[str(p), q] for p, q in zip(rows.player_b_id, tags)])
    u = np.asarray(snapshot.coefficients)
    eta = design @ beta + u[ia] - u[ib]
    anchor = int(rows.season.max())
    weights = 2. ** (-(anchor - rows.season.to_numpy()) / 3.)
    labels = rows.a_won.to_numpy(float)
    residual = weights * (expit(eta) - labels)
    gradient += np.bincount(ia, weights=residual, minlength=len(keys))
    gradient -= np.bincount(ib, weights=residual, minlength=len(keys))
    objective = float(weights @ (np.logaddexp(0., eta) - labels * eta) + prior)
    if joint:
        objective += .5 * float(beta @ beta) / .01
        gradient = np.r_[design.T @ residual + beta / .01, gradient]
    d = snapshot.diagnostics
    assert d['success'] and int(d['rows']) == len(rows) and d['weight_anchor_season'] == anchor
    assert d['solver_options'] == dict(maxiter=5000, maxcor=30, ftol=0., gtol=1e-10)
    assert d['accepted_gradient_tolerance'] == 1e-8
    assert d['max_eligible_date'] == pd.to_datetime(rows.tourney_date).max().date().isoformat()
    assert d['shared_ridge_constant_omitted'] == (not joint)
    error = _close(objective, d['objective_unscaled'], 2e-7)
    gradient_max = float(np.max(np.abs(gradient)))
    _close(gradient_max, d['max_gradient_unscaled'], 2e-7)
    assert gradient_max / len(rows) <= d['accepted_gradient_tolerance'] * (1 + 1e-6)
    return dict(objective_error=error, max_gradient_unscaled=gradient_max,
                max_gradient_scaled=gradient_max / len(rows), nodes=len(keys))


def independent_prediction(model, features, query):
    shared = independent_design(model.shared_model_, features) @ model.beta_
    residual = np.zeros(len(query))
    cutoffs = (pd.to_datetime(query.tourney_date) - pd.Timedelta(days=21)).dt.strftime('%Y-%m-%d')
    for cutoff in cutoffs.unique():
        state = model.state_snapshots_[cutoff]
        values = {}
        for (player, _), value in zip(state.keys, state.coefficients):
            values[str(player)] = float(value)
        ix = np.flatnonzero(cutoffs.eq(cutoff).to_numpy())
        q = query.iloc[ix]
        residual[ix] = [values.get(str(a), 0.) - values.get(str(b), 0.)
                        for a, b in zip(q.player_a_id, q.player_b_id)]
    return np.clip(expit(shared + residual), 1e-8, 1 - 1e-8), residual


def independent_interval(rows, reference, candidate, *, loss=False):
    y = rows.a_won.to_numpy()
    a, b = np.asarray(reference), np.asarray(candidate)
    if loss:
        gain = (-y * np.log(a) - (1 - y) * np.log1p(-a)
                + y * np.log(b) + (1 - y) * np.log1p(-b))
    else:
        gain = ((b >= .5) == y).astype(int) - ((a >= .5) == y).astype(int)
    grouped = {}
    for day, value in zip(rows.tourney_date, gain):
        total, count = grouped.get(str(day), (0., 0))
        grouped[str(day)] = (total + value, count + 1)
    blocks = np.array([grouped[day] for day in sorted(grouped)])
    rng = np.random.default_rng(158)
    picks = rng.integers(len(blocks), size=(2000, len(blocks)))
    totals = blocks[picks].sum(axis=1)
    return dict(mean=float(np.mean(gain)), ci95=np.quantile(totals[:, 0] / totals[:, 1], [.025, .975]),
                date_blocks=len(blocks), net_correct=int(gain.sum()) if not loss else None)


def _transform_hash(control):
    names = ('spec', 'odd_scale_', 'odd_indices_', 'spline_scale_',
             'context_mean_', 'context_scale_', 'interaction_scale_', 'design_width_')
    state = {key: getattr(control, key) for key in names}
    state['splines_'] = {}
    for name, spline in control.splines_.items():
        state['splines_'][name] = dict(params=spline.get_params(deep=False),
            n_features_in=spline.n_features_in_, n_features_out=spline.n_features_out_,
            bsplines=[{key: getattr(basis, key) for key in ('t', 'c', 'k', 'axis', 'extrapolate')}
                      for basis in spline.bsplines_])
    def numeric_encoding(value):
        if isinstance(value, np.ndarray):
            return dict(shape=list(value.shape), dtype=value.dtype.str, data=value.tobytes(order='C').hex())
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported hash field {type(value)}')
    encoded = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False, default=numeric_encoding)
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_audit(out):
    """Reload every fold/state. No optimization or base-model fitting is performed."""
    from unittest.mock import patch
    from scripts.train_residual_state_round12 import predict_saved

    manifest = json.loads((out / 'run_manifest.json').read_text())
    for name, digest in manifest['artifact_sha256'].items():
        assert sha256(out / name) == digest, name
    registry = json.loads((out / 'search_registry.json').read_text())
    sources = {'protocol_sha256': 'docs/research/round12_residual_state_protocol.md',
               'runner_sha256': 'scripts/train_residual_state_round12.py',
               'engine_sha256': 'scripts/residual_state_round12.py',
               'base_learner_sha256': 'scripts/round2_learning.py',
               'temporal_mask_sha256': 'scripts/train_models.py'}
    for key, path in sources.items():
        assert sha256(ROOT / path) == registry[key], key
    for name, digest in registry['previous_artifact_sha256'].items():
        assert sha256(ROOT / 'results/round5' / name) == digest
    target_path = ROOT / 'results/round3/development_targets.csv'
    assert sha256(target_path) == registry['target_sha256']
    rows = pd.read_csv(target_path)
    x = pd.read_csv(ROOT / 'results/round5/development_features.csv')
    assert len(rows) == 36421 and x.shape == (36421, 326)
    assert list(x) == registry['feature_names']
    assert registry['final_test_evaluated'] is False and registry['no_mixture_search'] is True
    assert set(registry['candidates']) == {*STRUCTURES, REFERENCE}
    oof = pd.read_csv(out / 'oof_predictions.csv')
    valid = pd.read_csv(out / 'validation_predictions.csv')
    old_oof = pd.read_csv(ROOT / 'results/round5/oof_predictions.csv')
    old_valid = pd.read_csv(ROOT / 'results/round5/validation_predictions.csv')
    for expected, saved in [(rows[rows.season.isin([2020, 2021, 2022])], oof),
                            (rows[rows.season.eq(2023)], valid)]:
        assert saved[['match_id', 'season', 'a_won']].equals(
            expected[['match_id', 'season', 'a_won']].reset_index(drop=True))
    assert len(oof) == 6130 and len(valid) == 2588
    selected = independent_selection(oof, oof[[*STRUCTURES, REFERENCE]])
    frozen = json.loads((out / 'frozen_selection.json').read_text())
    freeze_record = json.loads((out / 'pre2023_freeze.json').read_text())
    assert freeze_record['frozen_selection_sha256'] == sha256(out / 'frozen_selection.json')
    assert freeze_record['oof_predictions_sha256'] == sha256(out / 'oof_predictions.csv')
    assert frozen['procedure_gate']['early_selection']['choice'] == selected['early_selected']
    assert frozen['pooled_selection']['choice'] == selected['pooled_selected']
    assert frozen['recommendation'] == selected['recommended']
    assert frozen['procedure_gate']['net_correct'] == selected['additional_correct']
    assert (frozen['procedure_gate']['procedure'] == 'adopt_accuracy_model') == selected['gate_passed']
    maxima = dict(probability=0., old_control=0., symmetry=0., permutation=0., batching=0.,
                  transform=0., objective=0., joint_gradient_unscaled=0., state_gradient_unscaled=0., interval=0.)
    state_count, checks = 0, []
    for year in (2020, 2021, 2022, 2023):
        bundle = joblib.load(out / f'model_{year}.joblib')
        assert bundle['year'] == year and bundle['registry'] == registry
        assert bundle['frozen'] == (frozen if year == 2023 else None)
        fit, cutoff = independent_outer_indices(rows, year)
        mask = rows.season.eq(year)
        query, xq = rows.loc[mask, META], x.loc[mask]
        saved = valid if year == 2023 else oof[oof.season.eq(year)]
        old = old_valid if year == 2023 else old_oof[old_oof.season.eq(year)]
        assert bundle['training_match_ids'] == rows.loc[fit, 'match_id'].tolist()
        assert bundle['query_metadata'].equals(query)
        shared = bundle['shared_model']
        assert shared.spec == registry['base_spec'] and shared.design_width_ == 256
        maxima['transform'] = max(maxima['transform'], audit_transform(shared, x.loc[fit]))
        all_design = independent_design(shared, x)
        pp = np.clip(expit(all_design[mask] @ shared.estimator_.coef_[0]), 1e-8, 1 - 1e-8)
        maxima['probability'] = max(maxima['probability'], _close(pp, saved.shared_only))
        maxima['old_control'] = max(maxima['old_control'], _close(pp, old['all_inputs_spline_C0.01']))
        assert np.array_equal(pp >= .5, old['all_inputs_spline_C0.01'].to_numpy() >= .5)
        _close(saved[REFERENCE], old[REFERENCE])
        binding = independent_fingerprint(x.loc[fit], rows.loc[fit], shared.spec['columns'])
        assert shared.residual_training_fingerprint_ == binding
        assert shared.residual_training_match_ids_ == tuple(rows.loc[fit, 'match_id'])
        assert _transform_hash(shared) == shared.residual_transform_sha256_
        probabilities = predict_saved(bundle, xq, query)
        maxima['probability'] = max(maxima['probability'], _close(probabilities, saved[list(probabilities)]))
        assert np.array_equal(probabilities.to_numpy() >= .5, saved[list(probabilities)].to_numpy() >= .5)
        if year == 2023:
            final_bundle = joblib.load(out / 'frozen_models.joblib')
            assert final_bundle['frozen_selection_sha256'] == freeze_record['frozen_selection_sha256']
            maxima['probability'] = max(maxima['probability'], _close(
                probabilities, predict_saved(final_bundle, xq, query)))
            del final_bundle
        reverse_x, reverse_q = xq.copy(), query.copy()
        odd = [name for name in reverse_x if name.endswith('_diff')]
        reverse_x[odd] *= -1
        reverse_q[['player_a_id', 'player_b_id']] = reverse_q[['player_b_id', 'player_a_id']].to_numpy()
        reverse_p = predict_saved(bundle, reverse_x, reverse_q)
        maxima['symmetry'] = max(maxima['symmetry'], _close(probabilities + reverse_p, np.ones(probabilities.shape)))
        permuted = predict_saved(bundle, xq.iloc[::-1], query.iloc[::-1]).iloc[::-1]
        maxima['permutation'] = max(maxima['permutation'], _close(probabilities, permuted))
        assert np.array_equal(probabilities.to_numpy() >= .5, permuted.to_numpy() >= .5)
        batched = pd.concat([predict_saved(bundle, xq.iloc[ix], query.iloc[ix])
                             for ix in np.array_split(np.arange(len(query)), 4)])
        maxima['batching'] = max(maxima['batching'], _close(probabilities, batched))
        assert np.array_equal(probabilities.to_numpy() >= .5, batched.to_numpy() >= .5)
        for name, model in bundle['models'].items():
            assert model.training_match_ids_ == tuple(rows.loc[fit, 'match_id'])
            assert model.training_fingerprint_ == binding and model.transform_sha256_ == _transform_hash(shared)
            assert model.beta_sha256_ == hashlib.sha256(np.asarray(model.beta_, dtype='<f8').tobytes()).hexdigest()
            assert _transform_hash(model.shared_model_) == model.transform_sha256_
            assert model.training_metadata_['initial_sd'] == .35
            assert model.training_metadata_['innovation_sd'] == .15
            assert model.training_metadata_['C'] == .01 and model.training_metadata_['half_life'] == 3.
            assert model.training_metadata_['match_ids_sha256'] == hashlib.sha256(
                '\n'.join(rows.loc[fit, 'match_id']).encode()).hexdigest()
            assert model.training_state_.diagnostics == model.fit_diagnostics_
            global_check = independent_snapshot(model.training_state_, rows.loc[fit],
                                                all_design[fit], model.beta_, joint=True)
            maxima['objective'] = max(maxima['objective'], global_check['objective_error'])
            maxima['joint_gradient_unscaled'] = max(maxima['joint_gradient_unscaled'], global_check['max_gradient_unscaled'])
            assert model.training_state_.history_sha256 == binding
            expected_cutoffs = (pd.to_datetime(query.tourney_date) - pd.Timedelta(days=21)).dt.strftime('%Y-%m-%d')
            assert set(model.state_snapshots_) == set(expected_cutoffs)
            for key, snapshot in model.state_snapshots_.items():
                index = independent_prefix_indices(rows, key)
                history = rows.loc[index].sort_values(['tourney_date', 'match_id'], kind='stable')
                index = history.index
                assert snapshot.cutoff == key
                assert snapshot.history_sha256 == independent_fingerprint(x.loc[index], history, shared.spec['columns'])
                detail = independent_snapshot(snapshot, history, all_design[index], model.beta_, joint=False)
                maxima['objective'] = max(maxima['objective'], detail['objective_error'])
                maxima['state_gradient_unscaled'] = max(maxima['state_gradient_unscaled'], detail['max_gradient_unscaled'])
                assert snapshot.diagnostics['beta_sha256'] == model.beta_sha256_
                assert snapshot.diagnostics['transform_sha256'] == model.transform_sha256_
                state_count += 1
            manual, residual = independent_prediction(model, xq, query)
            maxima['probability'] = max(maxima['probability'], _close(manual, probabilities[name]))
            assert np.array_equal(manual >= .5, probabilities[name].to_numpy() >= .5)
            # Cached earliest query: make every later outcome and feature unusable.
            # Patch the optimizer to prove this check cannot silently refit.
            first = expected_cutoffs.min()
            first_mask = expected_cutoffs.eq(first)
            poison_rows, poison_x = rows.copy(), x.copy()
            future = pd.to_datetime(rows.tourney_date).gt(pd.Timestamp(first))
            poison_rows.loc[future, 'a_won'] = -999
            poison_x.loc[future, shared.spec['columns']] = np.nan
            with patch('scripts.residual_state_round12._solve', side_effect=AssertionError('Audit must not optimize')):
                replay = model.predict_asof(xq.loc[first_mask], query.loc[first_mask], poison_x, poison_rows)
            _close(replay, probabilities.loc[first_mask, name])
            checks.append(dict(year=year, model=name, global_rows=len(fit), snapshots=len(model.state_snapshots_),
                               max_absolute_residual_logit=float(np.max(np.abs(residual))),
                               mean_absolute_residual_logit=float(np.mean(np.abs(residual))), **global_check))
    assert state_count == 256
    table = pd.read_csv(out / 'validation_comparison.csv')
    for row in table.to_dict('records'):
        metrics = independent_metrics(valid.a_won, valid[row['model']])
        for key, value in metrics.items(): _close(row[key], value, 1e-12)
        _close(row['brier'], np.mean((valid.a_won - valid[row['model']]) ** 2), 1e-12)
    cv = pd.read_csv(out / 'temporal_cv.csv')
    for row in cv.to_dict('records'):
        part = oof[oof.season.eq(row['year'])]
        for key, value in independent_metrics(part.a_won, part[row['model']]).items():
            _close(row['rows' if key == 'matches' else key], value, 1e-12)
    _close(valid['rank_reference'], old_valid['rank_reference'])
    _close(valid['previous_round5_exploration'], old_valid['accuracy_grid_blend'])
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['recommendation'] == selected['recommended']
    assert summary['state_only_solves'] == state_count and summary['global_stages'] == 12
    contrasts = pd.read_csv(out / 'matched_controls.csv')
    for contrast in contrasts.to_dict('records'):
        a, b = contrast['candidate'], contrast['control']
        for period, frame in [('oof', oof), ('validation', valid)]:
            ca = independent_metrics(frame.a_won, frame[a])
            cb = independent_metrics(frame.a_won, frame[b])
            assert contrast[period + '_net_correct'] == ca['correct'] - cb['correct']
            _close(contrast[period + '_loss_change'], ca['mean_log_loss'] - cb['mean_log_loss'], 1e-12)
        recorded = summary['matched_intervals'][a + '_vs_' + b]
        for loss in (False, True):
            computed = independent_interval(valid, valid[b], valid[a], loss=loss)
            expected = recorded['log_loss' if loss else 'accuracy']
            assert computed['date_blocks'] == expected['date_blocks'] and expected['repetitions'] == 2000
            maxima['interval'] = max(maxima['interval'], _close(computed['ci95'], expected['ci95'], 1e-12))
            _close(computed['mean'], expected['mean_log_loss_reduction' if loss else 'accuracy_improvement'], 1e-12)
            if not loss: assert computed['net_correct'] == expected['net_correct']
    result = dict(status='passed', development_rows=36421, oof_rows=6130, validation_rows=2588,
                  global_optimizations_replayed=0, saved_global_models_checked=12,
                  saved_state_snapshots_checked=state_count, independent_selection=selected,
                  maximum_errors=maxima, model_checks=checks, future_label_and_feature_poisoning='passed',
                  final_test_evaluated=False, artifact_sha256=manifest['artifact_sha256'],
                  source_sha256={str(path): sha256(ROOT / path) for path in sources.values()},
                  audit_script_sha256=sha256(Path(__file__)), limits=[
                      'No optimizer trajectory or real-data fit was replayed; saved objectives and gradients were recomputed.',
                      'Old 326 input fields were hash-bound, not reconstructed from raw match sources.',
                      'MAP forecasts do not integrate state uncertainty; reused gate and 2023 remain exploratory.'])
    (out / 'artifact_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round12/residual_state')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        result = run_audit(args.output)
    print(json.dumps({key: result[key] for key in ('status', 'saved_state_snapshots_checked', 'maximum_errors')}))

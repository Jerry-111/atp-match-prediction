"""Joint sparse binomial MAP point skills; an explicit adaptation of Ingram2019.

Every as-of fit jointly optimizes all eligible observations and player effects.
This is neither the old diagonal online update nor full Bayesian NUTS inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logit

from scripts.advanced_features import _date
from scripts.adjusted_point_features import _joined_history
from scripts.baseline import DEFAULT_CACHE
from scripts.multilevel_features import _season_date
from scripts.paper_point_model import match_probability
from scripts.serve_features import _annual_rows, development_entries

LAG = 21
WINDOW = 1460
OFFSET = float(logit(.65))
PRIOR_SD = {'intercept': .5, 'baseline': .2, 'serve': .2, 'return': .15,
            'preference': .08, 'event': .08}


def candidate_specs():
    return [dict(name=name, opponent=opponent, half_life=half, surface=surface, event=event,
                 history_days=WINDOW, prior_sd=dict(PRIOR_SD))
            for name, opponent, half, surface, event in (
                ('serve_only_recent', False, 364., False, False),
                ('joint_static', True, None, False, False),
                ('joint_recent', True, 364., False, False),
                ('joint_surface', True, 364., True, False),
                ('joint_surface_event', True, 364., True, True))]


def event_identity(tourney_id, name, surface):
    parts = str(tourney_id).split('-', 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError('Tournament IDs require a source season prefix')
    normalized = ' '.join(unicodedata.normalize('NFKC', str(name or '')).casefold().split())
    if not normalized:
        return ''  # Unknown metadata has the neutral event effect.
    return json.dumps([parts[1], normalized, str(surface)], ensure_ascii=False, separators=(',', ':'))


def load_point_history(inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Reuse the pinned paired-count join and collect only preknown event metadata."""
    entries = development_entries() if raw_rows is None else []
    raw_source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    eligible = inventory[inventory.source.eq('tour') & pd.to_numeric(inventory.season).between(2006, 2023)]
    required = set(eligible.match_id.astype(str))
    metadata = {}
    def source():
        for season, row in raw_source:
            if not 2006 <= int(season) <= 2023:
                continue  # Do not inspect final-test fixture/source fields.
            key = f"tour:{row['tourney_id']}:{int(row['match_num'])}"
            if key in required:
                metadata[key.removeprefix('tour:')] = dict(
                    event=event_identity(row['tourney_id'], row.get('tourney_name'), row['surface']),
                    best_of=row.get('best_of'), season=int(season))
            yield season, row
    records, audit = _joined_history(inventory, cache_dir, source())
    observations = []
    for day, key, surface, a, b, aw, an, bw, bn in records:
        identifier = key.removeprefix('tour:')
        season = metadata[identifier]['season']
        _season_date(day, season)
        event = metadata[identifier]['event']
        for server, returner, won, total in ((a, b, aw, an), (b, a, bw, bn)):
            observations.append(dict(date=pd.Timestamp(day), match_id=identifier, surface=surface,
                event=event, server=server, returner=returner, won=won, total=total))
    frame = pd.DataFrame(observations, columns=['date','match_id','surface','event','server','returner','won','total'])
    audit.update(source_files=entries, injected_raw_rows=raw_rows is not None,
        service_observations=len(frame), event_identity='year-stripped code + normalized name + surface',
        final_test_evaluated=False)
    return frame, metadata, audit


def attach_metadata(targets, metadata):
    result = targets.copy()
    if result.empty or result.match_id.duplicated().any():
        raise ValueError('Unique nonempty targets required')
    best, events = [], []
    for row in result.itertuples():
        if int(row.season) not in range(2009, 2024):
            raise ValueError('Final holdout locked: development targets only')
        _season_date(_date(row.tourney_date), int(row.season))
        if str(row.player_a_id) == str(row.player_b_id):
            raise ValueError('Distinct players required')
        source = metadata[str(row.match_id)]
        if source['season'] != int(row.season) or int(source['best_of']) not in (3, 5):
            raise ValueError('Invalid preknown format/year source')
        best.append(int(source['best_of']))
        events.append(source['event'])
    result['best_of'], result['event_key'] = best, events
    return result


def terms(server, returner, surface, event, spec):
    result = [('intercept', 1.), (f'serve:{server}', 1.)]
    if surface != 'Hard':
        result.append((f'baseline:{surface}', 1.))
    if spec['opponent']:
        result.append((f'return:{returner}', -1.))
    if spec['surface']:
        result.extend([(f'preference:{server}:{surface}', 1.), (f'preference:{returner}:{surface}', -1.)])
    if spec['event'] and event:
        result.append((f'event:{event}', 1.))
    return result


def design_matrix(observations, spec, keys=None):
    term_rows = [terms(str(r.server), str(r.returner), str(r.surface), str(r.event), spec)
                 for r in observations.itertuples()]
    if keys is None:
        keys = tuple(sorted({'intercept'} | {key for row in term_rows for key, _ in row}))
    lookup = {key: i for i, key in enumerate(keys)}
    ii, jj, vv = [], [], []
    for index, row in enumerate(term_rows):
        for key, value in row:
            if key in lookup:
                ii.append(index); jj.append(lookup[key]); vv.append(value)
    matrix = csr_matrix((vv, (ii, jj)), shape=(len(observations), len(keys)), dtype=float)
    precision = np.array([1/spec['prior_sd'][key.split(':', 1)[0]]**2 for key in keys])
    return matrix, keys, precision


def objective_and_gradient(coefficients, x, won, total, weights, precision):
    eta = OFFSET + x @ coefficients
    scale = max(1., float(len(won)))
    # Remove a coefficient-independent baseline to avoid early termination from
    # cancellation in the much larger absolute binomial negative log likelihood.
    value = np.sum(weights*(total*(np.logaddexp(0., eta)-np.logaddexp(0., OFFSET))
                           -won*(eta-OFFSET))) + .5*np.sum(precision*coefficients**2)
    gradient = x.T @ (weights*(total*expit(eta)-won)) + precision*coefficients
    return float(value/scale), np.asarray(gradient/scale)


@dataclass
class JointPointSnapshot:
    spec: dict
    cutoff: str
    keys: tuple
    coefficients: np.ndarray
    diagnostics: dict

    def predict_serve(self, targets):
        if not set(('player_a_id','player_b_id','surface','event_key')).issubset(targets):
            raise ValueError('Player/surface/preknown-event schema required')
        values = []
        lookup = dict(zip(self.keys, self.coefficients))
        for row in targets.itertuples():
            if str(row.player_a_id) == str(row.player_b_id):
                raise ValueError('Distinct players required')
            pair = []
            for a, b in ((row.player_a_id, row.player_b_id), (row.player_b_id, row.player_a_id)):
                eta = OFFSET + sum(value*lookup.get(key, 0.) for key, value in
                    terms(str(a), str(b), str(row.surface), str(row.event_key), self.spec))
                pair.append(float(expit(eta)))
            values.append(pair)
        return np.asarray(values)

    def predict_proba(self, targets):
        if not targets.empty:
            for row in targets.itertuples():
                if int(row.season) not in range(2009, 2024):
                    raise ValueError('Final holdout locked')
                _season_date(_date(row.tourney_date), int(row.season))
                if (_date(row.tourney_date)-timedelta(days=LAG)).isoformat() != self.cutoff:
                    raise ValueError('Prediction requires the exact T-21 snapshot')
        p = self.predict_serve(targets)
        probability = match_probability(p[:,0], p[:,1], targets.best_of.to_numpy())
        # Equal service probabilities imply an exact tie under the stated rules.
        probability = np.where(p[:,0] == p[:,1], .5, probability)
        return np.clip(probability, 1e-8, 1-1e-8)


def fit_snapshot(history, cutoff, spec, initial=None):
    day = pd.Timestamp(_date(cutoff))
    if not 2006 <= day.year <= 2023:
        raise ValueError('Point state fitting is development-only')
    if initial is not None and (pd.Timestamp(initial.cutoff) > day or initial.spec != spec):
        raise ValueError('Warm starts must be the same model at an earlier/equal cutoff')
    start = day-pd.Timedelta(days=spec['history_days'])
    observed = history.loc[history.date.between(start, day)].sort_values(['date','match_id','server'], kind='stable')
    won, total = observed.won.to_numpy(float), observed.total.to_numpy(float)
    if (not np.isfinite(won).all() or not np.isfinite(total).all() or
            np.any(won < 0) or np.any(won > total) or np.any(total <= 0)):
        raise ValueError('Invalid eligible service counts')
    if observed[['match_id','server']].duplicated().any():
        raise ValueError('Duplicate eligible service observation')
    x, keys, precision = design_matrix(observed, spec)
    age = (day-observed.date).dt.total_seconds().to_numpy()/86400
    weights = np.ones(len(observed)) if spec['half_life'] is None else 2.**(-age/spec['half_life'])
    previous = {} if initial is None else dict(zip(initial.keys, initial.coefficients))
    coefficients = np.array([previous.get(key, 0.) for key in keys])
    started = time.monotonic()
    if len(observed):
        result = minimize(objective_and_gradient, coefficients, args=(x,won,total,weights,precision),
            method='L-BFGS-B', jac=True, options=dict(maxiter=1000, ftol=1e-13, gtol=1e-9, maxls=40))
        if not result.success:
            raise RuntimeError(f"Joint optimizer failed for {spec['name']} {day.date()}: {result.message}")
        coefficients, iterations, evaluations = result.x, int(result.nit), int(result.nfev)
    else:
        coefficients = np.zeros(len(keys))
        iterations = evaluations = 0
    value, gradient = objective_and_gradient(coefficients, x, won, total, weights, precision)
    if not np.isfinite(coefficients).all() or np.max(np.abs(gradient), initial=0.) > 1e-5:
        raise RuntimeError('Joint optimizer did not meet the finite/gradient check')
    diagnostics = dict(observations=len(observed), matches=int(observed.match_id.nunique()),
        parameters=len(keys), nonzeros=int(x.nnz), iterations=iterations, evaluations=evaluations,
        objective=value, max_scaled_gradient=float(np.max(np.abs(gradient), initial=0.)),
        fit_seconds=time.monotonic()-started, oldest_observation=str(observed.date.min()) if len(observed) else None,
        newest_observation=str(observed.date.max()) if len(observed) else None,
        total_points=float(total.sum()), weighted_points=float((weights*total).sum()), converged=True)
    return JointPointSnapshot(spec, day.date().isoformat(), keys, coefficients, diagnostics)


def generate_raw_predictions(targets, history, specs=None, *, keep_states=False, progress=None):
    specs = candidate_specs() if specs is None else specs
    if targets.empty or targets.index.duplicated().any() or not targets.season.isin(range(2009, 2024)).all():
        raise ValueError('Unique development target index required')
    cutoffs = targets.tourney_date.map(lambda d: _date(d)-timedelta(days=LAG))
    probabilities = pd.DataFrame(index=targets.index, columns=[s['name'] for s in specs], dtype=float)
    diagnostics, states, previous = [], {s['name']: {} for s in specs}, {}
    for number, cutoff in enumerate(sorted(cutoffs.unique()), 1):
        query = targets.loc[cutoffs.eq(cutoff)]
        for spec in specs:
            name = spec['name']
            snapshot = fit_snapshot(history, cutoff, spec, previous.get(name))
            previous[name] = snapshot
            probabilities.loc[query.index, name] = snapshot.predict_proba(query)
            diagnostics.append(dict(model=name, cutoff=snapshot.cutoff, queries=len(query), **snapshot.diagnostics))
            if keep_states:
                states[name][snapshot.cutoff] = snapshot
        if progress is not None:
            progress(number, len(cutoffs.unique()), cutoff, diagnostics[-len(specs):])
    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError('Missing/nonfinite raw point predictions')
    return probabilities, pd.DataFrame(diagnostics), states

"""Reproducible chronological model search; never evaluates the final test seasons.

Run as ``python -m scripts.train_models``. All tuning uses 2020--22 expanding
folds. The search registry is written before fitting; 2023 is scored only after
the advanced candidate and ensemble are frozen. See docs/modeling.md.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities

ROOT = Path(__file__).resolve().parents[1]
BASE = ['log_rank_diff', 'overall_rate_diff', 'surface_rate_diff', 'h2h_rate_diff']
FOLDS = (2020, 2021, 2022)
SEED = 158


def mirror_features(x):
    """Odd player-difference columns reverse; symmetric context stays fixed."""
    out = x.copy()
    odd = [c for c in x if c.endswith('_diff')]
    out.loc[:, odd] = -out.loc[:, odd]
    return out


def temporal_masks(rows, year):
    dates = pd.to_datetime(rows.tourney_date)
    validation = rows.season.eq(year)
    if not validation.any():
        raise ValueError(f'No rows in fold {year}')
    last_available = dates[validation].min() - pd.Timedelta(days=21)
    train = rows.season.lt(year) & dates.le(last_available)
    if not train.any():
        raise ValueError(f'No earlier training rows for {year}')
    return train, validation


class SymmetricClassifier:
    """Exchangeable prediction with mirrored training and train-only scaling."""
    def __init__(self, spec):
        self.spec = spec

    def fit(self, x, y, seasons=None):
        x = x[self.spec['columns']]
        xx = pd.concat([x, mirror_features(x)], ignore_index=True)
        yy = np.concatenate([y, 1 - np.asarray(y)])
        kind, params = self.spec['kind'], self.spec['params']
        if kind == 'logistic':
            self.estimator = make_pipeline(StandardScaler(), LogisticRegression(
                C=params['C'], fit_intercept=False, max_iter=2000, random_state=SEED))
        elif kind == 'catboost':
            from catboost import CatBoostClassifier
            self.estimator = CatBoostClassifier(**params, loss_function='Logloss',
                random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False)
        elif kind == 'histboost':
            self.estimator = HistGradientBoostingClassifier(**params, early_stopping=False,
                random_state=SEED)
        else:
            raise ValueError(kind)
        # Half weights preserve original effective sample size under augmentation.
        weights = np.full(len(xx), .5)
        if self.spec.get('half_life') and seasons is not None:
            w = 2.0 ** (-(max(seasons) - np.asarray(seasons)) / self.spec['half_life'])
            weights *= np.tile(w, 2)
        kw = {'logisticregression__sample_weight': weights} if kind == 'logistic' else {'sample_weight': weights}
        self.estimator.fit(xx, yy, **kw)
        return self

    def predict_proba(self, x):
        x = x[self.spec['columns']]
        p = .5 * (self.estimator.predict_proba(x)[:, 1] +
                  1 - self.estimator.predict_proba(mirror_features(x))[:, 1])
        return np.clip(p, 1e-8, 1 - 1e-8)


def mean_loss(y, p):
    p = np.clip(p, 1e-8, 1-1e-8)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def fit_blend(y, predictions):
    """Nonnegative convex weights fitted only on historical OOF predictions."""
    p = np.column_stack(predictions)
    n = p.shape[1]
    result = minimize(lambda w: mean_loss(y, p @ w), np.full(n, 1/n),
        method='SLSQP', bounds=[(0., 1.)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
        options={'ftol': 1e-12, 'maxiter': 1000})
    if not result.success:
        raise RuntimeError(f'Blend optimization failed: {result.message}')
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def specs_for(x):
    specs = []
    def add(name, kind, cols, params, family, **extra):
        specs.append(dict(name=name, kind=kind, columns=list(cols), params=params,
                          family=family, **extra))
    for bits in itertools.product([False, True], repeat=3):
        cols = [BASE[0]] + [c for c, include in zip(BASE[1:], bits) if include]
        family = 'lr_' + ''.join('1' if b else '0' for b in bits)
        for c in [.01, .1, 1., 10.]:
            add(f'{family}_C{c:g}', 'logistic', cols, {'C': c}, family)
    elo = [c for c in x if 'elo' in c and c.endswith('_diff')]
    serve = [c for c in x if c.startswith('points_') and c.endswith('_diff')]
    odd = [c for c in x if c.endswith('_diff')]
    groups = {'elo_rank': [BASE[0]] + elo, 'history_linear': odd,
              'elo_history': list(dict.fromkeys(BASE + elo))}
    if serve:
        groups['elo_serve'] = list(dict.fromkeys(BASE + elo + serve))
    for group, cols in groups.items():
        for c in [.01, .1, 1., 10.]:
            add(f'{group}_C{c:g}', 'logistic', cols, {'C': c}, 'advanced_linear')
    for depth, iterations, l2 in [(3, 400, 5), (4, 500, 10), (5, 600, 15), (6, 600, 20)]:
        for half in [None, 3.]:
            add(f'cat_d{depth}_half{half}', 'catboost', x.columns,
                dict(depth=depth, iterations=iterations, learning_rate=.035, l2_leaf_reg=l2),
                'catboost', half_life=half)
    for leaves, l2 in [(7, 10.), (15, 20.), (15, 50.), (31, 50.)]:
        add(f'hist_l{leaves}_reg{l2:g}', 'histboost', x.columns,
            dict(max_iter=250, max_leaf_nodes=leaves, learning_rate=.04,
                 l2_regularization=l2, min_samples_leaf=80), 'histboost')
    # Ablations retrain with identical tuning; they are explanatory comparisons,
    # not candidates in the pre-validation champion decision.
    for group, cols in [('all_history_reference', odd),
                        ('without_elo', [c for c in odd if 'elo' not in c]),
                        ('without_serve', [c for c in odd if c not in serve]),
                        ('without_rank', [c for c in odd if 'rank' not in c])]:
        for c in [.01, .1, 1.]:
            add(f'{group}_C{c:g}', 'logistic', cols, {'C': c}, group)
    return specs


def cluster_interval(rows, reference, candidate, repeats=2000):
    """Paired date-block bootstrap; conditional on fitted/selected models."""
    y = rows.a_won.to_numpy()
    loss = lambda p: -y*np.log(p)-(1-y)*np.log1p(-p)
    frame = pd.DataFrame({'date': rows.tourney_date.to_numpy(),
                          'gain': loss(reference)-loss(candidate)})
    groups = frame.groupby('date').gain.agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(groups), size=(repeats, len(groups)))
    sample = groups[idx]
    means = sample[:, :, 0].sum(axis=1) / sample[:, :, 1].sum(axis=1)
    return dict(mean_log_loss_reduction=float(frame.gain.mean()),
                ci95=np.quantile(means, [.025, .975]).tolist(),
                date_blocks=len(groups), repetitions=repeats,
                interpretation='Exploratory paired date-block interval; does not account for model search.')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'modeling')
    parser.add_argument('--with-serve', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    target_path = ROOT/'data/processed/model_features.csv'
    inventory_path = ROOT/'data/processed/mens_singles_2006_2025.csv'
    rows = pd.read_csv(target_path)
    if not rows.season.between(2016, 2023).all() or rows.match_id.duplicated().any():
        raise ValueError('Development targets must be unique 2016--2023 rows')
    inventory = pd.read_csv(inventory_path, dtype={'a_won': 'string'})
    # Outcome-bearing final-test records must never enter the feature builder.
    history = inventory.loc[inventory.season.le(2023)].copy()
    del inventory
    from scripts.advanced_features import build_advanced_features, FEATURE_VERSION
    print('Building chronological history features', flush=True)
    advanced = build_advanced_features(rows, history)
    x = pd.concat([rows[BASE], advanced.drop(columns=BASE, errors='ignore')], axis=1)
    serve_metadata = None
    if args.with_serve:
        from scripts.serve_features import build_serve_features
        serve_frame = build_serve_features(rows, history)
        serve_metadata = serve_frame.attrs
        x = pd.concat([x, serve_frame], axis=1)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Features must be unique and finite')
    x.to_csv(out/'development_features.csv', index=False)
    specs = specs_for(x)
    registry = dict(protocol_version=1, seed=SEED, folds=list(FOLDS), features=list(x),
        feature_version=FEATURE_VERSION, target_sha256=sha256(target_path),
        inventory_sha256=sha256(inventory_path), serve_metadata=serve_metadata, candidates=specs,
        selection='Min pooled 2020--22 OOF log loss per family. Advanced champion chosen before 2023.',
        ensemble='Convex OOF blend; use only if better on 2022 when weights fit 2020--21.',
        holdout='2024--2025 unopened; sequential historical updates allowed under frozen settings.')
    (out/'search_registry.json').write_text(json.dumps(registry, indent=2)+'\n')
    oof_mask = rows.season.isin(FOLDS)
    oof_rows = rows[oof_mask]
    y_oof = oof_rows.a_won.to_numpy()
    oof = {}
    cv = []
    for i, spec in enumerate(specs):
        predictions = pd.Series(index=rows.index[oof_mask], dtype=float)
        fold_losses = {}
        for year in FOLDS:
            train, valid = temporal_masks(rows, year)
            model = SymmetricClassifier(spec).fit(x[train], rows.a_won[train], rows.season[train])
            p = model.predict_proba(x[valid])
            predictions.loc[valid[valid].index] = p
            fold_losses[str(year)] = mean_loss(rows.a_won[valid].to_numpy(), p)
        p = predictions.to_numpy()
        oof[spec['name']] = p
        cv.append(dict(name=spec['name'], family=spec['family'],
                       mean_log_loss=mean_loss(y_oof, p), **fold_losses))
        print(f"[{i+1}/{len(specs)}] {spec['name']}: {cv[-1]['mean_log_loss']:.6f}", flush=True)
    # Raw Elo is selected by training-period performance, no learned calibration.
    elo_cols = [f'elo_k{k}_diff' for k in (16, 32, 64)]
    for col in elo_cols:
        p = expit(np.log(10) / 400 * x.loc[oof_mask, col].to_numpy())
        name = 'raw_' + col
        oof[name] = p
        cv.append(dict(name=name, family='raw_elo', mean_log_loss=mean_loss(y_oof, p),
            **{str(year): mean_loss(y_oof[oof_rows.season.eq(year)], p[oof_rows.season.eq(year)]) for year in FOLDS}))
    cv = pd.DataFrame(cv)
    cv.to_csv(out/'temporal_cv.csv', index=False)
    selected = cv.sort_values(['mean_log_loss', 'name']).groupby('family', sort=False).first()['name'].to_dict()
    available = ['advanced_linear', 'catboost', 'histboost', 'raw_elo']
    components = [selected[f] for f in available if f in selected]
    base_champion = min(components, key=lambda n: mean_loss(y_oof, oof[n]))
    # Nested chronological blend gate limits the optimism of fitting weights to OOF.
    meta_train = oof_rows.season.lt(2022).to_numpy()
    meta_valid = ~meta_train
    early_components = [min(cv.loc[cv.family.eq(f), 'name'],
        key=lambda n: (mean_loss(y_oof[meta_train], oof[n][meta_train]), n))
        for f in available if f in selected]
    early_best = min(early_components, key=lambda n: mean_loss(y_oof[meta_train], oof[n][meta_train]))
    early_weights = fit_blend(y_oof[meta_train], [oof[n][meta_train] for n in early_components])
    blend_2022 = np.column_stack([oof[n][meta_valid] for n in early_components]) @ early_weights
    # Hyperparameters and weights for this decision are chosen without 2022.
    # The adopted procedure is subsequently refit with all development folds.
    use_blend = mean_loss(y_oof[meta_valid], blend_2022) < mean_loss(y_oof[meta_valid], oof[early_best][meta_valid])
    weights = fit_blend(y_oof, [oof[n] for n in components])
    champion = 'oof_convex_ensemble' if use_blend else base_champion
    frozen = dict(selected_by_family=selected, advanced_champion=champion,
        best_single=base_champion, ensemble_components=components, ensemble_weights=weights.tolist(),
        ensemble_gate_2022=bool(use_blend), gate_loss=mean_loss(y_oof[meta_valid], blend_2022),
        gate_reference_loss=mean_loss(y_oof[meta_valid], oof[early_best][meta_valid]),
        gate_components=early_components, gate_weights=early_weights.tolist(), gate_reference=early_best,
        caveat='Ensemble OOF loss is meta-training loss. Gate selects the procedure on 2022; 2023 remains validation.',
        final_test_evaluated=False, fit_end_season=2022, history_lag_days=21,
        source_hashes={p.name: sha256(p) for p in [target_path, inventory_path]})
    # Persist the decision before reading any 2023 outcomes for scoring.
    (out/'frozen_selection.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print(f'Frozen advanced candidate: {champion}; now scoring 2023', flush=True)
    train, valid = temporal_masks(rows, 2023)
    vr = rows[valid]
    predictions = {}
    models = {}
    by_name = {s['name']: s for s in specs}
    for name in selected.values():
        if name.startswith('raw_'):
            predictions[name] = expit(np.log(10)/400*x.loc[valid, name[4:]].to_numpy())
        else:
            model = SymmetricClassifier(by_name[name]).fit(x[train], rows.a_won[train], rows.season[train])
            models[name] = model
            predictions[name] = model.predict_proba(x[valid])
    predictions['oof_convex_ensemble'] = np.column_stack([predictions[n] for n in components]) @ weights
    outcomes = list(zip(vr.match_id, vr.a_won))
    table = []
    for name, p in predictions.items():
        scores = score_probabilities(outcomes, zip(vr.match_id, p))
        scores['brier'] = float(np.mean((vr.a_won.to_numpy()-p)**2))
        table.append(dict(model=name, advanced_recommendation=name==champion, **scores))
    result = pd.DataFrame(table).sort_values('mean_log_loss')
    result.to_csv(out/'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), pd.DataFrame(predictions)], axis=1).to_csv(out/'validation_predictions.csv', index=False)
    pd.concat([oof_rows[['match_id','season','a_won']].reset_index(drop=True), pd.DataFrame(oof)], axis=1).to_csv(out/'oof_predictions.csv', index=False)
    lr_names = [n for fam, n in selected.items() if fam.startswith('lr_')]
    subset_table = result[result.model.isin(lr_names)].copy()
    subset_table['feature_count'] = subset_table.model.map(lambda n: len(by_name[n]['columns']))
    subset = subset_table.sort_values(['mean_log_loss', 'feature_count', 'model']).iloc[0].model
    baseline = selected['lr_000']
    rank_correct = ((vr.rank_a < vr.rank_b).to_numpy() == vr.a_won.to_numpy()).sum()
    intervals = {name: cluster_interval(vr, predictions[baseline], p) for name, p in predictions.items()}
    summary = dict(advanced_recommendation=champion, prescribed_subset_selected_on_2023=subset,
        ranking_rule=dict(matches=len(vr), correct=int(rank_correct), accuracy=float(rank_correct/len(vr))),
        paired_vs_rank_logistic=intervals, validation_count=len(vr), final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    joblib.dump(dict(models=models, frozen=frozen, feature_names=list(x), raw_elo_columns=elo_cols), out/'frozen_models.joblib')
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    # Persist classes under their importable module, not transient __main__.
    from scripts.train_models import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()

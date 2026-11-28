"""Chronological, symmetric TabICLv2 inference with matched compact controls.

Run in the isolated .venv-foundation environment. Checkpoint weights are fixed;
the original context rows and transforms are learned only before each fold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.advanced_features import _date
from scripts.multilevel_features import _season_date
from scripts.train_models import FOLDS, ROOT, mean_loss, sha256, temporal_masks
from scripts.train_round5 import accuracy, accuracy_interval, predict_round5, selection_probability
from scripts.train_transfer_round6 import REFERENCE, R5_COMPARATOR, choose_procedure

CHECKPOINT_NAME = 'tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA256 = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
CHECKPOINT_REVISION = '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
R5_LOSS = 'round5_probability_comparator'
CONTEXT_SIZES = (2048, 4096)
MEMBERS = 4
EXTRA_COLUMNS = (
    'points_overall_364_serve_rate_diff', 'points_overall_364_return_rate_diff',
    'points_surface_364_serve_rate_diff', 'points_surface_364_return_rate_diff',
    'points_overall_364_serve_points_log_sum', 'points_surface_364_serve_points_log_sum',
    'profile_age_diff', 'profile_age30_diff', 'profile_age_interval_diff',
    'profile_age_missing_diff', 'profile_age_mean', 'profile_age_missing_sum',
    'format_best_of_five', 'format_log_rank_diff', 'form_90_rate_diff', 'h2h_rate_diff',
)


def feature_columns(available, compact_columns):
    columns = list(compact_columns) + list(EXTRA_COLUMNS)
    if len(columns) != 64 or len(set(columns)) != 64 or not set(columns).issubset(available):
        raise ValueError('Exactly the frozen 48+16 distinct foundation columns are required')
    return columns


def mirror(x):
    result = x.copy()
    odd = [c for c in result if c.endswith('_diff')]
    result.loc[:, odd] *= -1
    return result


def context_indices(rows, year, count):
    if year not in (*FOLDS, 2023) or count not in CONTEXT_SIZES:
        raise ValueError('Only prespecified development folds and context counts are allowed')
    train, _ = temporal_masks(rows, year)
    eligible = rows[train].copy()
    eligible['_date_order'] = pd.to_datetime(eligible.tourney_date)
    ordered = eligible.sort_values(['_date_order', 'match_id'], kind='stable')
    if len(ordered) < count:
        raise ValueError('Insufficient earlier-season context rows')
    return ordered.index[-count:]


def candidate_specs(columns):
    if len(columns) != 64 or len(set(columns)) != 64:
        raise ValueError('Frozen 64-column input required')
    params = {'tabicl': dict(batch_size=1, kv_cache=True, use_amp=False, use_fa3=False,
                            softmax_temperature=.9, average_logits=True,
                            inference_config={'ROW_CONFIG': {'safety_factor': .05},
                                              'COL_CONFIG': {'safety_factor': .05, 'offload': 'cpu'}}),
              'logistic': dict(C=.01, fit_intercept=False, max_iter=2000),
              'catboost': dict(depth=4, iterations=600, learning_rate=.03, l2_leaf_reg=20)}
    return [dict(name=f'{kind}_context{size}', kind=kind, context_rows=size, columns=list(columns),
                 n_estimators=MEMBERS if kind == 'tabicl' else None, parameters=dict(params[kind]))
            for size in CONTEXT_SIZES for kind in ('tabicl', 'logistic', 'catboost')]


def blend_registry(specs):
    return {f"{s['name']}_r4blend{w:g}": dict(model=s['name'], model_weight=w)
            for s in specs if s['kind'] == 'tabicl' for w in (.25, .5, .75)}


def add_blends(predictions, blends):
    result = {n: np.asarray(p, dtype=float) for n, p in predictions.items()}
    for name, setting in blends.items():
        weight = setting['model_weight']
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Mixtures must be convex')
        result[name] = weight * result[setting['model']] + (1 - weight) * result[REFERENCE]
    return result


def select_candidate(y, predictions, mask):
    # Both older Round 5 comparators are descriptive, never eligible selectors.
    from scripts.train_transfer_round6 import select_candidate as select_fixed
    selectable = {n: p for n, p in predictions.items() if n not in (R5_COMPARATOR, R5_LOSS)}
    return select_fixed(y, selectable, mask)


def choose_foundation_procedure(y, predictions, seasons):
    selectable = {n: p for n, p in predictions.items() if n not in (R5_COMPARATOR, R5_LOSS)}
    gate = choose_procedure(y, selectable, seasons)
    if gate['procedure'] == 'adopt_transfer':
        gate['procedure'] = 'adopt_foundation'
    return gate


def make_backend(spec, checkpoint):
    if spec['kind'] == 'tabicl':
        from tabicl import TabICLClassifier
        if sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Frozen foundation checkpoint changed')
        return TabICLClassifier(n_estimators=spec['n_estimators'], **spec['parameters'],
            model_path=str(checkpoint), allow_auto_download=False, checkpoint_version=CHECKPOINT_NAME,
            device='mps', offload_mode='auto', random_state=158, n_jobs=2)
    if spec['kind'] == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(**spec['parameters'], random_state=158))
    if spec['kind'] == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**spec['parameters'],
            loss_function='Logloss', random_seed=158, thread_count=2, verbose=False, allow_writing_files=False)
    raise ValueError('Unknown foundation comparison learner')


class SymmetricContextClassifier:
    """Frozen contextual algorithm or matched control, with exact odd projection."""
    def __init__(self, spec, checkpoint):
        self.spec, self.checkpoint = spec, str(checkpoint)

    def _input(self, x):
        columns = self.spec['columns']
        if x.empty or x.columns.duplicated().any() or not set(columns).issubset(x):
            raise ValueError('Missing, duplicate or empty foundation features')
        result = x[columns].astype(np.float32)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError('Foundation inputs must be finite')
        return result

    def fit(self, x, rows):
        if len(x) != len(rows) or not x.index.equals(rows.index) or not len(rows):
            raise ValueError('Aligned nonempty context rows required')
        if len(rows) > self.spec['context_rows'] or not rows.season.isin(range(2009, 2023)).all():
            raise ValueError('Context must respect its fixed size and exclude 2023/test seasons')
        if rows.match_id.duplicated().any() or not rows.a_won.isin((0, 1)).all():
            raise ValueError('Unique context IDs and binary labels required')
        for date, season in zip(rows.tourney_date, rows.season):
            day = _date(date)
            _season_date(day, int(season))
            if day.year > 2022:
                raise ValueError('Context dates must precede 2023')
        x = self._input(x)
        xx = pd.concat([x, mirror(x)], ignore_index=True)
        yy = np.concatenate([rows.a_won.to_numpy(), 1 - rows.a_won.to_numpy()]).astype(int)
        self.backend_ = make_backend(self.spec, Path(self.checkpoint))
        weights = np.full(len(xx), .5)
        if self.spec['kind'] == 'tabicl':
            self.backend_.fit(xx.to_numpy(), yy)
        elif self.spec['kind'] == 'logistic':
            self.backend_.fit(xx.to_numpy(), yy, logisticregression__sample_weight=weights)
        else:
            self.backend_.fit(xx.to_numpy(), yy, sample_weight=weights)
        self.context_ids_ = rows.match_id.to_list()
        self.context_sha256_ = hashlib.sha256('\n'.join(self.context_ids_).encode()).hexdigest()
        return self

    def predict_proba(self, x, *, chunk_size=256):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('Positive prediction chunk size required')
        x = self._input(x)
        result = []
        for start in range(0, len(x), chunk_size):
            part = x.iloc[start:start + chunk_size]
            combined = pd.concat([part, mirror(part)], ignore_index=True)
            raw = np.asarray(self.backend_.predict_proba(combined.to_numpy())[:, 1], dtype=float)
            result.extend(.5 * (raw[:len(part)] + 1 - raw[len(part):]))
        result = np.asarray(result)
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError('Invalid foundation probabilities')
        return np.clip(result, 1e-8, 1 - 1e-8)


def configure_device():
    import torch
    if importlib.metadata.version('tabicl') != '2.2.0' or importlib.metadata.version('torch') != '2.15.0':
        raise RuntimeError('Use the pinned foundation environment versions')
    if not torch.backends.mps.is_available():
        raise RuntimeError('This frozen experiment requires its smoke-tested MPS device')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.60)


def free_device():
    import torch
    gc.collect()
    torch.mps.empty_cache()


def predict_foundation(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen feature matrix')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    previous = predict_round5(bundle['previous_bundle'], x)
    probabilities = {
        REFERENCE: previous['previous_probability_recommendation'].to_numpy(),
        R5_COMPARATOR: previous[bundle['previous_pooled_choice']].to_numpy(),
        R5_LOSS: previous['all_inputs_spline_C0.01'].to_numpy(),
    }
    probabilities.update({n: m.predict_proba(x) for n, m in bundle['models'].items()})
    result = add_blends(probabilities, bundle['blends'])
    return pd.DataFrame(result, index=features.index)


def score_table(rows, probabilities):
    from scripts.metrics import score_probabilities
    result = []
    for name, p in probabilities.items():
        result.append(dict(model=name, **score_probabilities(zip(rows.match_id, rows.a_won),
                           zip(rows.match_id, p)), brier=float(np.mean((rows.a_won.to_numpy() - p) ** 2))))
    return pd.DataFrame(result).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])


def write_report(out):
    summary = json.loads((out / 'summary.json').read_text())
    frozen = json.loads((out / 'frozen_selection.json').read_text())
    table = pd.read_csv(out / 'validation_comparison.csv')
    gate, pooled = frozen['procedure_gate'], frozen['pooled_selection']
    selected = table.set_index('model').loc[pooled['choice']]
    lines = ['# Round 7: frozen tabular foundation model', '',
        'The complete 2,588-match 2023 Tour cohort and T−21 historical inputs are unchanged. '
        'These are exploratory results after repeated earlier inspection of 2023. Final 2024–2025 outcomes remain sealed.', '',
        f"The frozen branch choice scores **{int(selected.correct)}/2,588 = {selected.accuracy:.3%}**, "
        f"log loss **{selected.mean_log_loss:.6f}**, on 2023. It does not improve the older Round 4 or Round 5 comparisons. "
        'The earlier Round 4 probability recommendation is preserved; this branch does not replace it globally.', '',
        f"Branch-local frozen accuracy choice: **{frozen['recommendation']}**. The early-selected 2022 accuracy gain is "
        f"**{100*gate['accuracy_improvement']:+.3f} percentage points** (required +0.5), "
        f"with log-loss change {gate['loss_increase']:+.6f} (allowed at most +0.005).", '',
        f"The pooled 2020–2022 choice `{pooled['choice']}` was frozen before new 2023 inference. "
        'Prior Round 5 comparators were excluded from every selection/gate. All candidates are retained below.', '',
        '| Model | Correct / 2,588 | Accuracy | Log loss |', '| --- | ---: | ---: | ---: |']
    for row in table.itertuples():
        lines.append(f'| {row.model} | {row.correct} | {100*row.accuracy:.3f}% | {row.mean_log_loss:.6f} |')
    lines += ['',
        'Two recent context sizes (2,048/4,096 original matches, doubled by player-swap mirroring) use identical '
        '64-column inputs for TabICLv2, logistic regression and CatBoost. TabICLv2 uses one frozen synthetic-pretrained '
        'checkpoint, four ensemble members and no gradient fine-tuning. Conventional controls use fixed settings. '
        'Six deterministic TabICLv2/Round 4 mixtures add no fitted parameters.', '',
        'Matched controls in matched_learner_controls.csv compare algorithms without changing rows or columns. '
        'Recent-context truncation can discard useful history; the older rich references use more training data and inputs. '
        'Generic tabular benchmark claims therefore do not determine the tennis result.', '',
        'The source/dependency/checkpoint hashes, context IDs and unscored runtime/invariance smoke results are saved. '
        'MPS float32 kernels introduce small numerical differences across query chunks; the invariance tolerance is '
        '2e−5, and serialized predictions are independently checked. The decision threshold stays 0.5.', '',
        'The reused 2022 gate and conditional date-block intervals do not account for every earlier search. '
        'No final-test superiority or universal SOTA claim follows from this development run.', '',
        'Reproduce: `.venv-foundation/bin/python -m scripts.train_foundation_round7`. '
        'Reload: `.venv-foundation/bin/python -m scripts.train_foundation_round7 --predict-features path/to/full337features.csv`. '
        'Refresh this report without fitting: `.venv-foundation/bin/python -m scripts.train_foundation_round7 --report-only`.', '',
        '[Protocol and primary evidence](../../../docs/research/round7_foundation.md) · '
        '[Frozen selection](frozen_selection.json) · [Runtime smoke](smoke.json)', '']
    (out / 'report_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round7/foundation')
    parser.add_argument('--predict-features', type=Path)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_report(out)
        return
    configure_device()
    checkpoint = out / 'checkpoints' / CHECKPOINT_NAME
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Pinned pretrained model hash mismatch')
    if args.predict_features:
        frame = pd.read_csv(args.predict_features)
        result = predict_foundation(joblib.load(out / 'frozen_models.joblib'), frame)
        if 'match_id' in frame:
            result.insert(0, 'match_id', frame.match_id.to_numpy())
        result.to_csv(out / 'inference_predictions.csv', index=False)
        return
    smoke = json.loads((out / 'smoke.json').read_text())
    if (not smoke['synthetic_only'] or smoke['outcome_scores_computed'] or not smoke['finite_probabilities']
            or smoke['max_chunk_difference'] > 2e-5 or smoke['max_swap_error'] > 2e-6):
        raise ValueError('Unscored device smoke/invariance check is incomplete')
    serialization = json.loads((out / 'serialization_smoke.json').read_text())
    if (not serialization['synthetic_only'] or serialization['outcome_scores_computed']
            or serialization['max_reload_difference'] > 2e-5 or serialization['max_chunk_difference'] > 2e-5
            or serialization['max_swap_error'] > 2e-6):
        raise ValueError('Unscored frozen-model persistence check is incomplete')
    memory_smoke = json.loads((out / 'memory_smoke.json').read_text())
    if (not memory_smoke['synthetic_only'] or memory_smoke['outcome_scores_computed']
            or not memory_smoke['finite_probabilities']
            or memory_smoke['small_context_equivalence_max_difference'] > 2e-5):
        raise ValueError('Full-size synthetic memory scheduling check is incomplete')
    previous = ROOT / 'results/round5'
    audit = json.loads((previous / 'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv'):
        if sha256(previous / name) != audit['artifact_sha256'][name]:
            raise ValueError('Prior audited artifact changed')
    transfer = ROOT / 'results/round6/transfer'
    tr = json.loads((transfer / 'search_registry.json').read_text())
    transfer_audit = json.loads((transfer / 'artifact_audit.json').read_text())
    if sha256(transfer / 'inference_features.csv') != transfer_audit['artifact_sha256']['inference_features.csv']:
        raise ValueError('Audited transfer inference features changed')
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    if sha256(ROOT / 'results/round3/development_targets.csv') != tr['prior_target_sha256']:
        raise ValueError('Frozen target metadata changed')
    full = pd.read_csv(transfer / 'inference_features.csv')
    if len(full) != len(rows) or list(full) != tr['inference_feature_names']:
        raise ValueError('Full inference feature schema changed')
    columns = feature_columns(full, tr['feature_names'])
    x = full[columns]
    specs, blends = candidate_specs(columns), blend_registry(candidate_specs(columns))
    orows = rows[rows.season.isin(FOLDS)]
    old_oof = pd.read_csv(previous / 'oof_predictions.csv')
    if not orows[['match_id', 'season', 'a_won']].reset_index(drop=True).equals(old_oof[['match_id', 'season', 'a_won']]):
        raise ValueError('Prior OOF identities changed')
    old_frozen = json.loads((previous / 'frozen_selection.json').read_text())
    oof = {REFERENCE: old_oof.previous_probability_recommendation.to_numpy(),
           R5_COMPARATOR: selection_probability(old_frozen['pooled_selection'],
                {n: old_oof[n].to_numpy() for n in old_frozen['pooled_selection']['components']}),
           R5_LOSS: old_oof['all_inputs_spline_C0.01'].to_numpy()}
    contexts = {}
    for year in (*FOLDS, 2023):
        for size in CONTEXT_SIZES:
            indices = context_indices(rows, year, size)
            contexts[f'{year}_{size}'] = dict(ids=rows.loc[indices, 'match_id'].to_list(),
                earliest_date=rows.loc[indices, 'tourney_date'].min(), latest_date=rows.loc[indices, 'tourney_date'].max(),
                original_rows=len(indices), mirrored_rows=2*len(indices))
    (out / 'context_rows.json').write_text(json.dumps(contexts, indent=2) + '\n')
    registry = dict(round=7, branch='foundation', candidates=specs, blends=blends, fit_candidates=6,
        selection_entries=13, foundation_features=columns, feature_names=list(full),
        checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_revision=CHECKPOINT_REVISION, n_estimators=MEMBERS,
        device='mps', precision='float32', cpu_threads=2, memory_fraction=.60, query_chunk_size=256,
        target_sha256=sha256(ROOT / 'results/round3/development_targets.csv'),
        feature_sha256=sha256(transfer / 'inference_features.csv'), context_sha256=sha256(out / 'context_rows.json'),
        protocol_sha256=sha256(ROOT / 'docs/research/round7_foundation.md'), source_code_sha256=sha256(Path(__file__)),
        dependencies_sha256=sha256(out / 'dependencies.txt'), smoke_sha256=sha256(out / 'smoke.json'),
        memory_smoke_sha256=sha256(out / 'memory_smoke.json'),
        model_parameter_count=serialization['model_parameter_count'],
        dependency_versions={n:importlib.metadata.version(n) for n in ('tabicl','torch','scikit-learn','numpy','pandas','catboost','joblib')},
        final_test_evaluated=False, threshold=.5, context_policy='latest earlier-season date-purged rows, independent of outcome')
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    x.to_csv(out / 'development_features.csv', index=False)
    y = orows.a_won.to_numpy()
    timing = []
    for i, spec in enumerate(specs, 1):
        p = pd.Series(index=orows.index, dtype=float)
        for year in FOLDS:
            start = time.monotonic()
            indices = context_indices(rows, year, spec['context_rows'])
            valid = rows.season.eq(year)
            model = SymmetricContextClassifier(spec, checkpoint).fit(x.loc[indices], rows.loc[indices])
            fit_seconds = time.monotonic()-start
            pred_start = time.monotonic()
            p.loc[rows.index[valid]] = model.predict_proba(x[valid])
            timing.append(dict(model=spec['name'], fold=year, original_context_rows=len(indices),
                               fit_seconds=fit_seconds, predict_seconds=time.monotonic()-pred_start,
                               context_sha256=model.context_sha256_))
            print(f"[{i}/6] {spec['name']} fold{year}: {time.monotonic()-start:.1f}s", flush=True)
            del model
            free_device()
        oof[spec['name']] = p.to_numpy()
        print(f"CV {spec['name']}: accuracy={accuracy(y,p.to_numpy()):.6f} loss={mean_loss(y,p.to_numpy()):.6f}", flush=True)
        pd.DataFrame(timing).to_csv(out / 'timing.csv', index=False)
    oof = add_blends(oof, blends)
    gate = choose_foundation_procedure(y, oof, orows.season)
    pooled = select_candidate(y, oof, np.ones(len(y), dtype=bool))
    recommendation = pooled['choice'] if gate['procedure'] == 'adopt_foundation' else REFERENCE
    frozen = dict(round=7, branch='foundation', recommendation=recommendation, procedure_gate=gate,
        pooled_selection=pooled, decision_threshold=.5, fit_end_season=2022, history_lag_days=21,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('Frozen foundation selection:', json.dumps(frozen), flush=True)
    cv = score_table(orows, oof)
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        cv[f'{year}_accuracy'] = cv.model.map({n: accuracy(y[mask], p[mask]) for n,p in oof.items()})
        cv[f'{year}_log_loss'] = cv.model.map({n: mean_loss(y[mask], p[mask]) for n,p in oof.items()})
    cv.to_csv(out / 'temporal_cv.csv', index=False)
    pd.concat([orows[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    matched = []
    for size in CONTEXT_SIZES:
        for control in ('logistic', 'catboost'):
            a, b = f'tabicl_context{size}', f'{control}_context{size}'
            for year in (*FOLDS, 'pooled'):
                mask = np.ones(len(y),dtype=bool) if year == 'pooled' else orows.season.eq(year).to_numpy()
                matched.append(dict(context_rows=size, foundation=a, control=b, fold=year,
                    accuracy_improvement=accuracy(y[mask],oof[a][mask])-accuracy(y[mask],oof[b][mask]),
                    log_loss_improvement=mean_loss(y[mask],oof[b][mask])-mean_loss(y[mask],oof[a][mask])))
    pd.DataFrame(matched).to_csv(out/'matched_learner_controls.csv',index=False)
    models = {}
    for spec in specs:
        indices = context_indices(rows, 2023, spec['context_rows'])
        start = time.monotonic()
        models[spec['name']] = SymmetricContextClassifier(spec, checkpoint).fit(x.loc[indices], rows.loc[indices])
        print(f"Final context fitted: {spec['name']} in {time.monotonic()-start:.1f}s",flush=True)
    bundle = dict(models=models, feature_names=list(full), foundation_features=columns,
        previous_bundle=joblib.load(previous/'frozen_models.joblib'),
        previous_pooled_choice=old_frozen['pooled_selection']['choice'],blends=blends,frozen=frozen,
        checkpoint_sha256=CHECKPOINT_SHA256)
    joblib.dump(bundle,out/'frozen_models.joblib')
    vr = rows[rows.season.eq(2023)]
    p = predict_foundation(bundle, full.loc[vr.index])
    table = score_table(vr,{n:p[n].to_numpy() for n in p})
    table['recommended'] = table.model.eq(recommendation)
    table.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    summary = dict(round=7,branch='foundation',recommendation=recommendation,pooled_choice=pooled['choice'],
        fit_candidates=6,selection_entries=13,foundation_features=64,inference_features=full.shape[1],
        validation_matches=len(vr),context_sizes=list(CONTEXT_SIZES),n_estimators=MEMBERS,
        correct_required_for_70=int(np.ceil(.7*len(vr))),
        comparisons_vs_round4={n:accuracy_interval(vr,p[REFERENCE].to_numpy(),p[n].to_numpy()) for n in p},
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out)
    print(table.to_string(index=False),flush=True)


if __name__ == '__main__':
    from scripts.train_foundation_round7 import main as canonical_main
    with threadpool_limits(limits=2):
        canonical_main()

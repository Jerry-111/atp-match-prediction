"""Two fixed stale/fresh information contrasts; three shared logistic fits.

Requires explicit reviewed protocol/qualification hashes. No probability
mixtures, pooled reselection, fresh-history training or threshold tuning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.freshness_round11 import COLUMNS, VERSION, build_freshness_inputs, raw_elo_probability
from scripts.metrics import score_probabilities
from scripts.train_distillation_round7 import validate_rows
from scripts.train_models import ROOT, SEED, SymmetricClassifier, temporal_masks, sha256, mean_loss, cluster_interval, mirror_features
from scripts.train_round5 import REFERENCE, accuracy, accuracy_interval, selection_probability
from scripts.train_transfer_round6 import verify_tour_targets

YEARS = (2021, 2022, 2023)
COUNTS = {2021: 2388, 2022: 2528, 2023: 2588}
ALGORITHMS = ('overall_elo', 'rank_rating_logistic')
PROBABILITY_COLUMNS = [f'{name}_{mode}' for name in ALGORITHMS for mode in ('stale', 'fresh')]
R5_REFERENCE = 'previous_round5_exploration'
ACCURACY_GATE, LOSS_GUARD = .005, .005
SPEC = {'name': 'shared_rank_rating_logistic_C0.01', 'kind': 'logistic',
        'columns': COLUMNS, 'params': {'C': .01}, 'half_life': None}


def _check_probabilities(predictions, n):
    for name in PROBABILITY_COLUMNS:
        value = np.asarray(predictions[name], dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all() or np.any((value <= 0) | (value >= 1)):
            raise ValueError('Invalid aligned probability: ' + name)


def predict_freshness(model, stale, fresh):
    """The same trained weights and scaler see both three-field inputs."""
    if (not stale.index.equals(fresh.index) or list(stale) != COLUMNS or list(fresh) != COLUMNS
            or stale.empty or not np.isfinite(stale.to_numpy()).all()
            or not np.isfinite(fresh.to_numpy()).all()
            or not np.array_equal(stale.log_rank_diff.to_numpy(), fresh.log_rank_diff.to_numpy())):
        raise ValueError('Matched finite stale/fresh inputs and unchanged rank required')
    values = {'overall_elo_stale': raw_elo_probability(stale),
              'overall_elo_fresh': raw_elo_probability(fresh),
              'rank_rating_logistic_stale': model.predict_proba(stale),
              'rank_rating_logistic_fresh': model.predict_proba(fresh)}
    unchanged = np.all(stale.to_numpy() == fresh.to_numpy(), axis=1)
    for algorithm in ALGORITHMS:
        values[algorithm + '_fresh'][unchanged] = values[algorithm + '_stale'][unchanged]
    _check_probabilities(values, len(stale))
    return pd.DataFrame(values, index=stale.index)


def choose_procedure(rows, predictions):
    """Select an information algorithm on 2021, then gate that one on 2022."""
    years = np.asarray(rows.season)
    y = np.asarray(rows.a_won)
    if (not rows.index.is_unique or (isinstance(predictions, pd.DataFrame) and not predictions.index.equals(rows.index))
            or not np.isin(years, (2021, 2022)).all() or set(years) != {2021, 2022}
            or not np.isin(y, (0, 1)).all()):
        raise ValueError('Selection needs only complete 2021/2022 binary cohorts')
    _check_probabilities(predictions, len(rows))
    early, held = years == 2021, years == 2022
    scores = []
    for algorithm in ALGORITHMS:
        stale = np.asarray(predictions[algorithm + '_stale'])[early]
        fresh = np.asarray(predictions[algorithm + '_fresh'])[early]
        net_correct = int(((fresh >= .5) == y[early]).sum() - ((stale >= .5) == y[early]).sum())
        gain = net_correct / int(early.sum())
        loss_delta = mean_loss(y[early], fresh) - mean_loss(y[early], stale)
        scores.append({'algorithm': algorithm, 'net_correct': net_correct, 'accuracy_improvement': gain,
            'log_loss_change': loss_delta, 'eligible': bool(gain > 0 and loss_delta <= LOSS_GUARD + 1e-12)})
    eligible = [s for s in scores if s['eligible']]
    chosen = min(eligible, key=lambda s: (-s['net_correct'], s['log_loss_change'], s['algorithm'])) if eligible else None
    algorithm = chosen['algorithm'] if chosen else None
    gate = {'selected_algorithm': algorithm, 'selection_season': 2021, 'gate_season': 2022,
            'required_accuracy_improvement': ACCURACY_GATE, 'allowed_loss_increase': LOSS_GUARD,
            'gate_passed': False, 'procedure': 'retain_original_history_recommendation'}
    if algorithm:
        stale = np.asarray(predictions[algorithm + '_stale'])[held]
        fresh = np.asarray(predictions[algorithm + '_fresh'])[held]
        before, after = accuracy(y[held], stale), accuracy(y[held], fresh)
        before_loss, after_loss = mean_loss(y[held], stale), mean_loss(y[held], fresh)
        net_correct = int(((fresh >= .5) == y[held]).sum() - ((stale >= .5) == y[held]).sum())
        improvement = net_correct / int(held.sum())
        passed = improvement >= ACCURACY_GATE - 1e-12 and after_loss - before_loss <= LOSS_GUARD + 1e-12
        gate.update(stale_2022_accuracy=before, fresh_2022_accuracy=after,
            stale_2022_loss=before_loss, fresh_2022_loss=after_loss,
            net_correct=net_correct, accuracy_improvement=improvement, log_loss_change=after_loss-before_loss,
            gate_passed=passed,
            procedure='adopt_bounded_freshness_algorithm' if passed else 'retain_original_history_recommendation')
    return {'early_selection': {'selection_season': 2021, 'selection_rows': int(early.sum()),
                'choice': algorithm, 'candidates': scores}, 'procedure_gate': gate,
            'recommendation': algorithm + '_fresh' if gate['gate_passed'] else REFERENCE,
            'pooled_selection_performed': False}


def load_context_predictions(rows, previous):
    """Reuse frozen R4/R5 probabilities; never fit contextual references."""
    frozen = json.loads((previous / 'frozen_selection.json').read_text())
    selection = frozen['pooled_selection']
    out = pd.DataFrame(index=rows.index, columns=[REFERENCE, R5_REFERENCE], dtype=float)
    for filename, years in [('oof_predictions.csv', (2021, 2022)), ('validation_predictions.csv', (2023,))]:
        old = pd.read_csv(previous / filename)
        old = old[old.season.isin(years)].set_index('match_id')
        current = rows[rows.season.isin(years)]
        if (old.index.duplicated().any() or set(old.index) != set(current.match_id)
                or not np.array_equal(old.loc[current.match_id, 'a_won'].to_numpy(), current.a_won.to_numpy())
                or not np.array_equal(old.loc[current.match_id, 'season'].to_numpy(), current.season.to_numpy())):
            raise ValueError('Context predictions are not the same complete target cohort')
        aligned = old.loc[current.match_id]
        out.loc[current.index, REFERENCE] = aligned[REFERENCE].to_numpy()
        out.loc[current.index, R5_REFERENCE] = selection_probability(selection, aligned)
        if years == (2023,) and selection['choice'] in aligned:
            if not np.array_equal(out.loc[current.index, R5_REFERENCE].to_numpy(), aligned[selection['choice']].to_numpy()):
                if np.max(np.abs(out.loc[current.index, R5_REFERENCE].to_numpy() - aligned[selection['choice']].to_numpy())) > 1e-14:
                    raise ValueError('Frozen R5 contextual composition changed')
            # Preserve the stored published values exactly, including CSV rounding.
            out.loc[current.index, R5_REFERENCE] = aligned[selection['choice']].to_numpy()
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Missing contextual reference probability')
    return out


def check_approval(protocol, manifest_path, protocol_hash, manifest_hash, *, root=ROOT):
    if not protocol_hash or not manifest_hash:
        raise ValueError('Explicit reviewed protocol and qualification hashes are required before fitting')
    if sha256(protocol) != protocol_hash or sha256(manifest_path) != manifest_hash:
        raise ValueError('Reviewed protocol or qualification manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('protocol_sha256') != protocol_hash
            or manifest.get('root_limited_experiment_authorized') is not True
            or manifest.get('final_test_outcomes_opened') is not False
            or manifest.get('strict_asof_reconstruction') is not False
            or manifest.get('models_fitted_at_freeze') != 0):
        raise ValueError('Qualification authorization or evidence semantics differ from reviewed protocol')
    bindings = manifest.get('input_sha256')
    if (not isinstance(bindings, dict) or not bindings
            or 'results/round3/development_targets.csv' not in bindings):
        raise ValueError('Qualification must bind all inputs and the complete target cohort')
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Frozen qualification input changed: ' + relative)
    pairs = manifest.get('pairs', manifest.get('qualified_pairs'))
    if not isinstance(pairs, list) or not pairs or any(p.get('admitted_for_prediction') is not True for p in pairs):
        raise ValueError('Manifest must contain explicitly admitted, nonempty qualified pairs')
    if manifest.get('final_test_evaluated', False):
        raise ValueError('Sealed final test cannot enter this experiment')
    return manifest, pairs


def score_table(rows, predictions, recommendation=None):
    records = []
    for name in predictions:
        p = predictions[name].to_numpy()
        score = score_probabilities(zip(rows.match_id, rows.a_won), zip(rows.match_id, p))
        records.append({'model': name, **score, 'brier': float(np.mean((rows.a_won.to_numpy()-p)**2)),
                        'recommended': name == recommendation})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round11/freshness')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/round11/completion/qualified_pairs.json')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/research/round11_freshness_protocol.md')
    parser.add_argument('--approved-protocol-sha256')
    parser.add_argument('--approved-manifest-sha256')
    args = parser.parse_args()
    manifest, pairs = check_approval(args.protocol, args.manifest, args.approved_protocol_sha256, args.approved_manifest_sha256)
    out, previous = args.output, ROOT / 'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'frozen_selection.json').exists():
        raise ValueError('Existing frozen run must not be silently overwritten')
    audit = json.loads((previous / 'artifact_audit.json').read_text())
    for filename in ('development_features.csv', 'oof_predictions.csv', 'validation_predictions.csv',
                     'frozen_selection.json', 'search_registry.json', 'frozen_models.joblib'):
        if sha256(previous / filename) != audit['artifact_sha256'][filename]:
            raise ValueError('Audited prior artifact changed: ' + filename)
    old_registry = json.loads((previous / 'search_registry.json').read_text())
    inventory_path = ROOT / 'data/processed/mens_singles_2006_2025.csv'
    target_path = ROOT / 'results/round3/development_targets.csv'
    if sha256(inventory_path) != old_registry['inventory_sha256']:
        raise ValueError('Pinned inventory changed')
    if sha256(ROOT / 'data/processed/model_features.csv') != old_registry['target_sha256']:
        raise ValueError('Original published target cohort changed')
    rows = pd.read_csv(target_path)
    validate_rows(rows)
    if len(rows) != 36421 or not rows.index.equals(pd.RangeIndex(36421)):
        raise ValueError('The complete development cohort is fixed')
    for year, count in COUNTS.items():
        if int(rows.season.eq(year).sum()) != count:
            raise ValueError('A fixed prediction-year denominator changed')
    inventory = pd.read_csv(inventory_path, dtype={'a_won': 'string'})
    inventory = inventory[pd.to_numeric(inventory.season).le(2023)].copy()
    verify_tour_targets(rows, inventory)
    base = pd.read_csv(previous / 'development_features.csv')
    if base.shape != (36421, 326):
        raise ValueError('Expected the unchanged 326-column development matrix')
    stale = base[COLUMNS].copy()
    print('Replaying old Elo on prefix plus fixed certified whole editions', flush=True)
    fresh, coverage, metadata = build_freshness_inputs(rows, inventory, stale, pairs)
    development = pd.concat([stale.add_prefix('stale_'), fresh.add_prefix('fresh_')], axis=1)
    development.to_csv(out / 'development_features.csv', index=False)
    coverage.to_csv(out / 'coverage_by_match.csv', index=False)
    (out / 'feature_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    registry = {'round': 11, 'branch': 'bounded_completed_event_freshness', 'seed': SEED,
        'prediction_years': list(YEARS), 'fit_configurations': 1, 'total_fits': 3,
        'paired_algorithms': list(ALGORITHMS), 'probability_columns': PROBABILITY_COLUMNS,
        'spec': SPEC, 'stale_training_only': True, 'new_mixtures': False, 'pooled_reselection': False,
        'protocol_sha256': sha256(args.protocol), 'qualified_manifest_sha256': sha256(args.manifest),
        'runner_sha256': sha256(Path(__file__)), 'feature_code_sha256': sha256(ROOT / 'scripts/freshness_round11.py'),
        'elo_engine_sha256': sha256(ROOT / 'scripts/advanced_features.py'),
        'classifier_code_sha256': sha256(ROOT / 'scripts/train_models.py'),
        'inventory_sha256': sha256(inventory_path), 'expanded_target_sha256': sha256(target_path),
        'previous_feature_sha256': sha256(previous / 'development_features.csv'),
        'previous_oof_sha256': sha256(previous / 'oof_predictions.csv'),
        'previous_validation_sha256': sha256(previous / 'validation_predictions.csv'),
        'previous_frozen_selection_sha256': sha256(previous / 'frozen_selection.json'),
        'development_features_sha256': sha256(out / 'development_features.csv'),
        'qualified_pairs': len(pairs), 'feature_metadata': metadata,
        'context_references_selection_eligible': False, 'decision_threshold': .5,
        'exploratory_after_2023': True, 'final_test_evaluated': False,
        'information_regime': metadata['information_regime']}
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    all_rows = rows[rows.season.isin(YEARS)]
    oof_rows = rows[rows.season.isin((2021, 2022))]
    values = pd.DataFrame(index=all_rows.index, columns=PROBABILITY_COLUMNS, dtype=float)
    fit_records, models, checks = [], {}, {}
    frozen = None
    for year in YEARS:
        if year == 2023:
            decision = choose_procedure(oof_rows, values.loc[oof_rows.index])
            frozen = {'round': 11, 'branch': 'bounded_completed_event_freshness', **decision,
                'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'decision_threshold': .5,
                'fit_end_season': 2022, 'protocol_sha256': registry['protocol_sha256'],
                'qualified_manifest_sha256': registry['qualified_manifest_sha256'],
                'information_regime': metadata['information_regime'], 'exploratory_after_2023': True,
                'final_test_evaluated': False, 'context_references_selection_eligible': False}
            (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
            print('Frozen before 2023 fit/scoring: ' + json.dumps(decision), flush=True)
        train, valid = temporal_masks(rows, year)
        validate_rows(rows[train], fitting=True)
        model = SymmetricClassifier(SPEC).fit(stale[train], rows.a_won[train].to_numpy(), rows.season[train].to_numpy())
        models[year] = model
        fit_records.append({'validation_season': year, 'train_rows': int(train.sum()),
            'train_max_date': str(pd.to_datetime(rows.loc[train, 'tourney_date']).max()),
            'validation_min_date': str(pd.to_datetime(rows.loc[valid, 'tourney_date']).min()),
            'training_match_ids_sha256': hashlib.sha256('\n'.join(rows.loc[train, 'match_id']).encode()).hexdigest(),
            'training_input_sha256': hashlib.sha256(stale.loc[train].to_numpy().tobytes()).hexdigest(),
            'training_inputs': 'stale only', 'spec': SPEC})
        bundle = {'model': model, 'prediction_year': year, 'columns': COLUMNS,
                  'training_metadata': fit_records[-1], 'registry': registry, 'frozen': frozen}
        filename = 'frozen_models.joblib' if year == 2023 else f'model_{year}.joblib'
        joblib.dump(bundle, out / filename)
        p = predict_freshness(model, stale[valid], fresh[valid])
        values.loc[rows.index[valid], PROBABILITY_COLUMNS] = p
        reloaded = predict_freshness(joblib.load(out / filename)['model'], stale[valid], fresh[valid])
        swapped = predict_freshness(model, mirror_features(stale[valid]), mirror_features(fresh[valid]))
        no_extra = ~coverage.loc[valid, 'has_extra_history']
        checks[year] = {'reload_max_error': float(np.max(np.abs(p.to_numpy()-reloaded.to_numpy()))),
            'swap_max_error': float(np.max(np.abs(p.to_numpy()+swapped.to_numpy()-1))),
            'no_extra_rows': int(no_extra.sum()), 'exact_fallback': all(np.array_equal(
                p.loc[no_extra, algorithm+'_stale'].to_numpy(), p.loc[no_extra, algorithm+'_fresh'].to_numpy()) for algorithm in ALGORITHMS)}
        if checks[year]['reload_max_error'] != 0 or checks[year]['swap_max_error'] > 1e-12 or not checks[year]['exact_fallback']:
            raise ValueError('Reload, exchange or fallback check failed')
        print(f'Finished single stale fit for {year}: {int(train.sum())} training rows; {int(valid.sum())} predictions', flush=True)
    context = load_context_predictions(all_rows, previous)
    values = pd.concat([values, context], axis=1)
    identity = ['match_id', 'season', 'tourney_date', 'tourney_id', 'a_won']
    all_predictions = pd.concat([all_rows[identity], values], axis=1)
    all_predictions.to_csv(out / 'all_predictions.csv', index=False)
    all_predictions[all_predictions.season.isin((2021, 2022))].to_csv(out / 'oof_predictions.csv', index=False)
    all_predictions[all_predictions.season.eq(2023)].to_csv(out / 'validation_predictions.csv', index=False)
    validation_rows = rows[rows.season.eq(2023)]
    comparison = score_table(validation_rows, values.loc[validation_rows.index], frozen['recommendation'])
    comparison.sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True]).to_csv(out / 'validation_comparison.csv', index=False)
    cv = []
    for year in (2021, 2022):
        yr = rows[rows.season.eq(year)]
        score = score_table(yr, values.loc[yr.index]);score.insert(0, 'season', year);cv.append(score)
    pd.concat(cv, ignore_index=True).to_csv(out / 'temporal_cv.csv', index=False)
    matched, intervals = [], {}
    for year in YEARS:
        yr = rows[rows.season.eq(year)]; yp = values.loc[yr.index]; yc = coverage.loc[yr.index]
        intervals[year] = {}
        for algorithm in ALGORITHMS:
            stale_p, fresh_p = yp[algorithm+'_stale'].to_numpy(), yp[algorithm+'_fresh'].to_numpy()
            info = {'season': year, 'algorithm': algorithm, 'matches': len(yr),
                'stale_correct': int(((stale_p >= .5) == yr.a_won.to_numpy()).sum()),
                'fresh_correct': int(((fresh_p >= .5) == yr.a_won.to_numpy()).sum()),
                'accuracy_improvement': accuracy(yr.a_won, fresh_p)-accuracy(yr.a_won, stale_p),
                'log_loss_change': mean_loss(yr.a_won.to_numpy(), fresh_p)-mean_loss(yr.a_won.to_numpy(), stale_p),
                'edition_covered_rows': int(yc.has_extra_history.sum()),
                'direct_player_covered_rows': int(yc.direct_player_extra_matches.gt(0).sum()),
                'rating_changed_rows': int(yc.rating_input_changed.sum()),
                'probability_changed_rows': int((stale_p != fresh_p).sum()),
                'probability_changed_over_1e12_rows': int((np.abs(stale_p-fresh_p) > 1e-12).sum()),
                'classification_changed_rows': int(((stale_p >= .5) != (fresh_p >= .5)).sum()),
                'exact_fallback_rows': int((~yc.has_extra_history).sum())}
            matched.append(info)
            intervals[year][algorithm] = {'accuracy': accuracy_interval(yr, stale_p, fresh_p),
                                         'log_loss': cluster_interval(yr, stale_p, fresh_p)}
    pd.DataFrame(matched).to_csv(out / 'matched_controls.csv', index=False)
    (out / 'fit_records.json').write_text(json.dumps(fit_records, indent=2) + '\n')
    summary = {'round': 11, 'branch': 'bounded_completed_event_freshness',
        'recommendation': frozen['recommendation'], 'procedure_gate': frozen['procedure_gate'],
        'total_fits': 3, 'paired_algorithms': 2, 'new_probability_columns': 4,
        'all_prediction_rows': len(all_rows), 'oof_matches': len(oof_rows), 'validation_matches': len(validation_rows),
        'qualified_pairs': len(pairs), 'feature_metadata': metadata, 'comparisons': intervals,
        'matched_controls': matched, 'correct_required_for_70': 1812,
        'exploratory_after_2023': True, 'final_test_evaluated': False,
        'pooled_selection_performed': False, 'information_regime': metadata['information_regime'],
        'context_reference_caveat': 'R4/R5 were selected in prior exploration; their earlier-year reference rows are descriptive and excluded from this mechanism selector.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'self_check.json').write_text(json.dumps({'year_checks': checks, 'old_feature_alignment': metadata['old_elo_rebuild_max_abs_error'],
        'all_prediction_rows': len(all_rows), '2023_scoring_after_freeze': True,
        'artifact_sha256': {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'self_check.json'}}, indent=2) + '\n')
    print(comparison.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_freshness_round11 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()

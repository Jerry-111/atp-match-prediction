"""Fixed conditional binary player states, with all choices frozen before 2023."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.residual_state_round12 import ResidualStateModel, fit_shared_control
from scripts.train_distillation_round7 import validate_rows
from scripts.train_models import ROOT, FOLDS, sha256, temporal_masks, mean_loss, cluster_interval
from scripts.train_round5 import REFERENCE, accuracy, accuracy_interval

BASE = 'all_inputs_spline_C0.01'
SHARED = 'shared_only'
STRUCTURES = {'shared_static_residual': 'static', 'shared_dynamic_residual': 'dynamic'}
CANDIDATES = [SHARED, *STRUCTURES, REFERENCE]
META = ['match_id', 'season', 'tourney_date', 'player_a_id', 'player_b_id']
PROTOCOL = ROOT / 'docs/research/round12_residual_state_protocol.md'


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def correct(y, p):
    return int(np.count_nonzero((np.asarray(p) >= .5) == np.asarray(y)))


def select_single(y, predictions, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if (y.ndim != 1 or mask.shape != y.shape or not mask.any()
            or not np.isin(y, [0, 1]).all() or set(predictions) != set(CANDIDATES)):
        raise ValueError('Exact candidate set and aligned binary selection rows required')
    values = {name: np.asarray(p) for name, p in predictions.items()}
    if any(p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
           for p in values.values()):
        raise ValueError('Invalid selection probability')
    losses = {name: mean_loss(y[mask], p[mask]) for name, p in values.items()}
    counts = {name: correct(y[mask], p[mask]) for name, p in values.items()}
    limit = losses[REFERENCE] + .005
    eligible = sorted(name for name in values if losses[name] <= limit)
    choice = min(eligible, key=lambda name: (-counts[name], losses[name], name))
    return dict(choice=choice, correct=counts[choice], selection_rows=int(mask.sum()),
                accuracy=counts[choice] / int(mask.sum()), mean_log_loss=losses[choice],
                loss_limit=limit, eligible_models=eligible, mixture_search=False)


def choose_procedure(y, predictions, seasons):
    y, seasons = np.asarray(y), np.asarray(seasons)
    if seasons.shape != y.shape or set(np.unique(seasons)) != set(FOLDS):
        raise ValueError('Exactly the three forward OOF years required')
    early = select_single(y, predictions, np.isin(seasons, [2020, 2021]))
    held = seasons == 2022
    if int(held.sum()) != 2528:
        raise ValueError('The original full 2022 gate must contain 2,528 rows')
    ref = np.asarray(predictions[REFERENCE])[held]
    candidate = np.asarray(predictions[early['choice']])[held]
    gain = correct(y[held], candidate) - correct(y[held], ref)
    ref_loss, loss = mean_loss(y[held], ref), mean_loss(y[held], candidate)
    passed = gain >= 13 and loss <= ref_loss + .005
    return dict(procedure='adopt_accuracy_model' if passed else 'retain_round4',
                early_selection=early, gate_season=2022, gate_rows=2528,
                reference_correct=correct(y[held], ref), candidate_correct=correct(y[held], candidate),
                net_correct=gain, required_net_correct=13,
                reference_log_loss=ref_loss, candidate_log_loss=loss, allowed_loss_increase=.005)


def check_control(y, current, previous):
    y, current, previous = np.asarray(y), np.asarray(current), np.asarray(previous)
    if (y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all()
            or current.shape != y.shape or previous.shape != y.shape
            or not np.isfinite(current).all() or not np.isfinite(previous).all()
            or np.any((current < 0) | (current > 1))
            or np.any((previous < 0) | (previous > 1))):
        raise ValueError('Misaligned control prediction')
    error = float(np.max(np.abs(current - previous)))
    changes = int(np.count_nonzero((current >= .5) != (previous >= .5)))
    if error > 1e-10 or changes:
        raise ValueError(f'Old shared control failed reproduction: {error}, {changes} class changes')
    return dict(rows=len(current), max_absolute_error=error, classification_changes=changes,
                correct=correct(y, current))


def predict_saved(bundle, x_query, query_metadata):
    """Infer from saved snapshots only; no outcomes or optimizer required."""
    if 'a_won' in query_metadata or not set(META).issubset(query_metadata):
        raise ValueError('Label-free query identity metadata required')
    values = {SHARED: bundle['shared_model'].predict_proba(x_query)}
    values.update({name: model.predict_from_snapshots(x_query, query_metadata)
                   for name, model in bundle['models'].items()})
    if any(np.asarray(p).shape != (len(x_query),) or not np.isfinite(p).all()
           or np.any((p < 0) | (p > 1)) for p in values.values()):
        raise ValueError('Invalid saved prediction')
    return pd.DataFrame(values, index=x_query.index)


def load_inputs():
    previous = ROOT / 'results/round5'
    audited = json.loads((previous / 'artifact_audit.json').read_text())['artifact_sha256']
    input_names = ['development_features.csv', 'search_registry.json', 'oof_predictions.csv',
                   'validation_predictions.csv', 'frozen_selection.json']
    for name in input_names:
        if sha256(previous / name) != audited[name]:
            raise ValueError(f'Pinned R5 input changed: {name}')
    old_registry = json.loads((previous / 'search_registry.json').read_text())
    specs = [s for s in old_registry['candidates'] if s['name'] == BASE]
    if len(specs) != 1:
        raise ValueError('Exactly one original shared model specification required')
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    validate_rows(rows)
    if len(rows) != 36421 or int(rows.season.eq(2023).sum()) != 2588:
        raise ValueError('Original full cohort required')
    x = pd.read_csv(previous / 'development_features.csv')
    if x.shape != (len(rows), 326) or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Original finite historical features required')
    old_oof = pd.read_csv(previous / 'oof_predictions.csv')
    old_valid = pd.read_csv(previous / 'validation_predictions.csv')
    identity = ['match_id', 'season', 'a_won']
    for mask, old in [(rows.season.isin(FOLDS), old_oof), (rows.season.eq(2023), old_valid)]:
        if not rows.loc[mask, identity].reset_index(drop=True).equals(old[identity]):
            raise ValueError('Original target identities changed')
    # Bind this historical target snapshot without opening the combined final-test inventory.
    older_registry = json.loads((ROOT / 'results/round9/matchup/search_registry.json').read_text())
    target_sha = sha256(ROOT / 'results/round3/development_targets.csv')
    if target_sha != older_registry['target_sha256']:
        raise ValueError('Pinned development target table changed')
    return rows, x, specs[0], old_oof, old_valid, {n: audited[n] for n in input_names}, target_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round12/residual_state')
    parser.add_argument('--approved-protocol-sha256')
    parser.add_argument('--predict-features', type=Path)
    parser.add_argument('--query-metadata', type=Path)
    args = parser.parse_args()
    out = args.output
    if args.predict_features:
        if args.query_metadata is None:
            parser.error('--query-metadata is required for state inference')
        xq, rq = pd.read_csv(args.predict_features), pd.read_csv(args.query_metadata)
        bundle = joblib.load(out / 'frozen_models.joblib')
        p = predict_saved(bundle, xq, rq)
        p.insert(0, 'match_id', rq.match_id.to_numpy())
        p.to_csv(out / 'inference_predictions.csv', index=False)
        return
    if not args.approved_protocol_sha256 or sha256(PROTOCOL) != args.approved_protocol_sha256:
        parser.error('Exact reviewed protocol hash required before empirical fitting')
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new empty output directory; original results must not be overwritten')
    rows, x, spec, old_oof, old_valid, source_hashes, target_sha = load_inputs()
    out.mkdir(parents=True, exist_ok=True)
    registry = dict(round=12, branch='conditional_binary_player_residuals',
        created_at_utc=datetime.now(timezone.utc).isoformat(), candidates=CANDIDATES,
        structures=STRUCTURES, base_spec=spec, feature_names=list(x), folds=list(FOLDS),
        global_optimization_stages=12, maximum_state_only_solves=256,
        protocol_sha256=sha256(PROTOCOL), runner_sha256=sha256(Path(__file__)),
        engine_sha256=sha256(ROOT / 'scripts/residual_state_round12.py'),
        base_learner_sha256=sha256(ROOT / 'scripts/round2_learning.py'),
        temporal_mask_sha256=sha256(ROOT / 'scripts/train_models.py'),
        previous_artifact_sha256=source_hashes, target_sha256=target_sha,
        decision_threshold=.5, historical_lag_days=21, history_first_season=2009,
        initial_residual_sd=.35, annual_innovation_sd=.15, final_test_evaluated=False,
        exploratory_after_2023=True, no_mixture_search=True)
    dump_json(out / 'search_registry.json', registry)
    predictions, control_checks, populations = [], {}, []
    optimizer_records, residual_records = [], []
    frozen, frozen_sha = None, None
    for year in (*FOLDS, 2023):
        if year == 2023:
            oof_frame = pd.concat(predictions, ignore_index=True)
            oof_frame.to_csv(out / 'oof_predictions.csv', index=False)
            ys = oof_frame.a_won.to_numpy()
            op = {name: oof_frame[name].to_numpy() for name in CANDIDATES}
            gate = choose_procedure(ys, op, oof_frame.season)
            pooled = select_single(ys, op, np.ones(len(ys), dtype=bool))
            frozen = dict(round=12, procedure_gate=gate, pooled_selection=pooled,
                recommendation=pooled['choice'] if gate['procedure'] == 'adopt_accuracy_model' else REFERENCE,
                frozen_at_utc=datetime.now(timezone.utc).isoformat(), decision_threshold=.5,
                protocol_sha256=registry['protocol_sha256'], historical_lag_days=21,
                exploratory_after_2023=True, final_test_evaluated=False)
            dump_json(out / 'frozen_selection.json', frozen)
            frozen_sha = sha256(out / 'frozen_selection.json')
            dump_json(out / 'pre2023_freeze.json', dict(frozen_selection_sha256=frozen_sha,
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                oof_predictions_sha256=sha256(out / 'oof_predictions.csv')))
            print('FROZEN', json.dumps(frozen), flush=True)
        train, valid = temporal_masks(rows, year)
        vr, xq = rows[valid], x[valid]
        query = vr[META].copy()
        old = old_valid if year == 2023 else old_oof.loc[old_oof.season.eq(year)]
        shared = fit_shared_control(spec, x[train], rows[train])
        p = {SHARED: shared.predict_proba(xq), REFERENCE: old[REFERENCE].to_numpy()}
        control_checks[str(year)] = check_control(vr.a_won, p[SHARED], old[BASE].to_numpy())
        print(f'{year} shared control reproduced: {control_checks[str(year)]}', flush=True)
        models = {}
        for name, structure in STRUCTURES.items():
            model = ResidualStateModel(spec, structure=structure).fit(x[train], rows[train], shared_model=shared)
            # The API receives no target labels and filters historical rows before consuming outcomes.
            p[name] = model.predict_asof(xq, query, x, rows)
            models[name] = model
            optimizer_records.append(dict(year=year, model=name, stage='global', cutoff='',
                                          **model.fit_diagnostics_))
            for cutoff, snapshot in model.state_snapshots_.items():
                optimizer_records.append(dict(year=year, model=name, stage='state_only', cutoff=cutoff,
                                              **snapshot.diagnostics))
            state_effect = np.zeros(len(query))
            cutoffs = (pd.to_datetime(query.tourney_date)-pd.Timedelta(days=21)).dt.strftime('%Y-%m-%d')
            for cutoff, snapshot in model.state_snapshots_.items():
                indices = np.flatnonzero(cutoffs.eq(cutoff).to_numpy())
                state_effect[indices] = snapshot.differences(query.iloc[indices])
            shared_logit = model._design(xq) @ model.beta_
            residual_records.append(dict(year=year, model=name,
                mean_absolute_state_logit=float(np.mean(np.abs(state_effect))),
                rms_state_logit=float(np.sqrt(np.mean(state_effect**2))),
                max_absolute_state_logit=float(np.max(np.abs(state_effect))),
                classifications_changed_by_state=int(np.count_nonzero((shared_logit>=0)!=(shared_logit+state_effect>=0))),
                global_state_rms=float(np.sqrt(np.mean(model.training_state_.coefficients**2)))))
        if year == 2023 and sha256(out / 'frozen_selection.json') != frozen_sha:
            raise ValueError('The pre-2023 frozen choice changed')
        for name in STRUCTURES:
            print(f'{year} {name}: {correct(vr.a_won,p[name])}/{len(vr)}, '
                  f'loss={mean_loss(vr.a_won.to_numpy(),p[name]):.6f}', flush=True)
        bundle = dict(year=year, shared_model=shared, models=models, base_spec=spec,
                      feature_names=list(x), query_metadata=query, frozen=frozen,
                      registry=registry, frozen_selection_sha256=frozen_sha,
                      training_match_ids=rows.match_id[train].tolist())
        joblib.dump(bundle, out / f'model_{year}.joblib')
        reload_p = predict_saved(joblib.load(out / f'model_{year}.joblib'), xq, query)
        for name in [SHARED, *STRUCTURES]:
            if not np.array_equal(reload_p[name].to_numpy(), p[name]):
                raise ValueError(f'Saved inference failed exact reproduction: {year}/{name}')
        frame = pd.concat([vr.reset_index(drop=True), pd.DataFrame(p)], axis=1)
        populations.append(dict(year=year, training_rows=int(train.sum()), target_rows=int(valid.sum()),
            training_latest_date=str(pd.to_datetime(rows.tourney_date[train]).max()),
            earliest_query_cutoff=str(pd.to_datetime(vr.tourney_date).min()-pd.Timedelta(days=21)),
            query_dates=int(vr.tourney_date.nunique()), design_width=int(shared.design_width_),
            shared_optimizer_iterations=int(shared.estimator_.n_iter_[0])))
        if year != 2023:
            predictions.append(frame)
        else:
            joblib.dump(bundle, out / 'frozen_models.joblib')
            frame['rank_reference'] = old_valid.rank_reference.to_numpy()
            frame['previous_round5_exploration'] = old_valid.accuracy_grid_blend.to_numpy()
            frame.to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame(populations).to_csv(out / 'fit_populations.csv', index=False)
    pd.DataFrame(optimizer_records).to_csv(out / 'optimizer_diagnostics.csv', index=False)
    pd.DataFrame(residual_records).to_csv(out / 'residual_contributions.csv', index=False)
    dump_json(out / 'control_reproduction.json', control_checks)
    table = []
    for name in [*CANDIDATES, 'rank_reference', 'previous_round5_exploration']:
        scores = score_probabilities(zip(frame.match_id, frame.a_won), zip(frame.match_id, frame[name]))
        table.append(dict(model=name, recommended=name == frozen['recommendation'], **scores,
                          brier=float(np.mean((frame.a_won-frame[name])**2))))
    table = pd.DataFrame(table).sort_values(['accuracy','mean_log_loss'], ascending=[False, True])
    table.to_csv(out / 'validation_comparison.csv', index=False)
    cv = []
    for year in FOLDS:
        part = oof_frame.loc[oof_frame.season.eq(year)]
        for name in CANDIDATES:
            cv.append(dict(year=year, model=name, rows=len(part), correct=correct(part.a_won, part[name]),
                accuracy=accuracy(part.a_won, part[name]), mean_log_loss=mean_loss(part.a_won.to_numpy(),part[name].to_numpy())))
    pd.DataFrame(cv).to_csv(out / 'temporal_cv.csv', index=False)
    contrasts, intervals = [], {}
    for a,b in [('shared_static_residual',SHARED),('shared_dynamic_residual','shared_static_residual'),
                ('shared_dynamic_residual',SHARED)]:
        contrasts.append(dict(candidate=a, control=b,
            oof_net_correct=correct(oof_frame.a_won,oof_frame[a])-correct(oof_frame.a_won,oof_frame[b]),
            oof_loss_change=mean_loss(oof_frame.a_won.to_numpy(),oof_frame[a].to_numpy())-mean_loss(oof_frame.a_won.to_numpy(),oof_frame[b].to_numpy()),
            validation_net_correct=correct(frame.a_won,frame[a])-correct(frame.a_won,frame[b]),
            validation_loss_change=mean_loss(frame.a_won.to_numpy(),frame[a].to_numpy())-mean_loss(frame.a_won.to_numpy(),frame[b].to_numpy())))
        intervals[a+'_vs_'+b] = dict(accuracy=accuracy_interval(frame,frame[b].to_numpy(),frame[a].to_numpy()),
                                   log_loss=cluster_interval(frame,frame[b].to_numpy(),frame[a].to_numpy()))
    pd.DataFrame(contrasts).to_csv(out / 'matched_controls.csv', index=False)
    summary = dict(round=12, rows=2588, oof_rows=6130, target_correct=1812,
                   recommendation=frozen['recommendation'], frozen_selection_sha256=frozen_sha,
                   matched_intervals=intervals,
                   control_reproduction=control_checks, global_stages=12,
                   state_only_solves=sum(v['stage']=='state_only' for v in optimizer_records),
                   final_test_evaluated=False, exploratory_after_2023=True)
    dump_json(out / 'summary.json', summary)
    files = sorted(p for p in out.iterdir() if p.is_file())
    dump_json(out / 'run_manifest.json', dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={p.name:sha256(p) for p in files}, source_registry_sha256=sha256(out/'search_registry.json')))
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

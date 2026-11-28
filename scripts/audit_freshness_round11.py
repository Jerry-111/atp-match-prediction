"""Independent Round11 availability, state and saved-estimator audit.

No optimization is performed. The independent rating implementation below
does not import HistoricalState, freshness_round11 or the qualification helper.
Human source/publication assumptions are checked as recorded, not re-adjudicated.
"""
from __future__ import annotations

from collections import defaultdict
import ast
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['log_rank_diff', 'elo_k32_diff', 'shrunk_surface_elo_k32_diff']
TARGET_SHA = '9bcb3ad4b0fbe745b23b776a750256f92ab9acc9fd1cb57aed0a16cef86435d2'
FRAME_SHA = ['5ab34f830540fa79feec12eeee8e6f53b0a9d2acd09ad68bc0c4cc89c5ee507f',
             'ea7922107d77f113251017d01ecd96144d4acc49462668dd39066da6344b09a6']


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def error(left, right):
    a, b = np.asarray(left), np.asarray(right)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def same(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right), (set(left), set(right))
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert abs(left-right) <= 1e-12, (left, right)
    else:
        assert left == right, (left, right)


def ordinal(value):
    return date.fromisoformat(str(value)).toordinal()


def read_eligible_history(path):
    """Parse source/season first, then skip sealed rows before reading labels."""
    meta = pd.read_csv(path, usecols=['source', 'season'])
    keep = meta.source.eq('tour') & meta.season.between(2006, 2023)
    skipped = (np.flatnonzero(~keep.to_numpy()) + 1).tolist()
    columns = ['source', 'season', 'match_id', 'tourney_id', 'tourney_date',
               'surface', 'player_a_id', 'player_b_id', 'a_won']
    history = pd.read_csv(path, skiprows=skipped, usecols=columns,
                          dtype={'player_a_id': str, 'player_b_id': str})
    assert len(history) == int(keep.sum()) and history.match_id.is_unique
    assert history.source.eq('tour').all() and history.season.between(2006, 2023).all()
    assert history.match_id.str.startswith('tour:').all()
    history['inventory_match_id'] = history.match_id
    history['match_id'] = history.match_id.str.slice(5)
    assert history.match_id.is_unique
    assert history.a_won.isin([0, 1]).all()
    assert history.surface.isin(['Hard', 'Clay', 'Grass']).all()
    for r in history.itertuples(index=False):
        d = date.fromisoformat(str(r.tourney_date))
        assert d.year == r.season or (d.year == r.season-1 and d.month == 12 and d.day >= 25)
        assert d.year <= 2023 and r.player_a_id != r.player_b_id
        assert r.match_id.startswith(r.tourney_id + ':')
        assert int(r.tourney_id.split('-')[0]) == r.season
    return history


def independent_availability(ledger, rows):
    """Recompute all original pairs from metadata and recorded evidence dates."""
    assert ledger['strict_asof_reconstruction'] is False
    assert ledger['models_fitted_at_freeze'] == 0
    assert ledger['root_limited_experiment_authorized'] is True
    for relative, expected in ledger['input_sha256'].items():
        assert digest(ROOT/relative) == expected, relative
    frame_paths = [ROOT/'results/round10/completion/source_frame.csv',
                   ROOT/'results/round10/completion/source_frame_2023_unqualified.csv']
    assert [digest(p) for p in frame_paths] == FRAME_SHA
    frame = pd.concat([pd.read_csv(p) for p in frame_paths], ignore_index=True)
    assert len(frame) == 763 and not frame.duplicated(['source_tourney_id', 'target_tourney_id']).any()
    sources, targets = {}, defaultdict(list)
    for year in (2021, 2022, 2023):
        probe = json.loads((ROOT/f'results/round11/completion/probe_{year}_first10.json').read_text())
        ordered = frame[frame.season.eq(year)][['source_tourney_id', 'source_nominal_date']]
        ordered = ordered.drop_duplicates().sort_values(['source_nominal_date', 'source_tourney_id'])
        expected = ordered.source_tourney_id.tolist()
        assert [v['source_tourney_id'] for v in probe['cases']] == expected[:10]
        assert [v['source_tourney_id'] for v in probe['worklist']] == expected
        for c in probe['cases']:
            full = [e for e in c.get('completion_evidence', [c]) if e.get('inspection') == 'full_text']
            if full and c.get('completed_by_day') and c.get('displayed_publication_day'):
                sources[c['source_tourney_id']] = max(ordinal(c['completed_by_day']),
                                                       ordinal(c['displayed_publication_day']))
            for e in c.get('target_start_evidence', []):
                day = e.get('announced_main_start_day') or e.get('reported_main_start_day')
                if e.get('inspection') == 'full_text' and day and e.get('start_semantics', '').startswith('explicit_'):
                    targets[c['source_tourney_id']].append(ordinal(day))
        for t in probe.get('target_start_evidence', []):
            if t.get('inspection') == 'full_text' and t.get('target_lower_day'):
                targets[t['target_tourney_id']].append(ordinal(t['target_lower_day']))
        for t in probe.get('authorized_target_start_followup', {}).get('targets', []):
            if t.get('start_supported'):
                for e in t.get('target_start_evidence', []):
                    if e.get('inspection') == 'full_text' and e.get('announced_main_start_day'):
                        targets[t['target_tourney_id']].append(ordinal(e['announced_main_start_day']))
    targets = {key: min(value) for key, value in targets.items()}
    assert set(sources) == set(ledger['sources']) and set(targets) == set(ledger['targets'])
    assert all(value == ordinal(ledger['sources'][key]['upper_day']) for key, value in sources.items())
    assert all(value == ordinal(ledger['targets'][key]['lower_day']) for key, value in targets.items())
    saved = {(c['source_tourney_id'], c['target_tourney_id']): c for c in ledger['cases']}
    assert len(saved) == len(ledger['cases']) == len(frame)
    groups = {key: g for key, g in rows.groupby('tourney_id')}
    accepted, by_target, details = [], defaultdict(set), []
    for f in frame.to_dict('records'):
        sid, tid = f['source_tourney_id'], f['target_tourney_id']
        source, target = groups[sid], groups[tid]
        c = saved[sid, tid]
        assert source.match_id.tolist() == json.loads(f['source_match_ids']) == c['source_match_ids']
        assert target.match_id.tolist() == json.loads(f['target_match_ids']) == c['target_match_ids']
        sd, td = ordinal(source.tourney_date.iloc[0]), ordinal(target.tourney_date.iloc[0])
        assert source.tourney_date.nunique() == target.tourney_date.nunique() == 1
        assert td-21 < sd <= td and sid != tid
        assert sd == ordinal(f['source_nominal_date']) and td == ordinal(f['target_nominal_date'])
        assert ordinal(f['original_T_minus_21_date']) == td-21
        assert c['season'] == int(tid[:4]) == int(target.season.iloc[0])
        passed = sid in sources and tid in targets and 24*(targets[tid]-sources[sid])-50 > 0
        assert c['admitted_for_prediction'] is bool(passed)
        people = set(source.player_a_id.astype(str)) | set(source.player_b_id.astype(str))
        direct = target.player_a_id.astype(str).isin(people) | target.player_b_id.astype(str).isin(people)
        assert c['direct_participant_target_ids'] == target.loc[direct, 'match_id'].tolist()
        if sid in sources and tid in targets:
            assert c['clearance_hours'] == 24*(targets[tid]-sources[sid])-50
            upper = datetime.combine(date.fromordinal(sources[sid]), datetime.min.time(), timezone.utc) + timedelta(hours=36)
            lower = datetime.combine(date.fromordinal(targets[tid]), datetime.min.time(), timezone.utc) - timedelta(hours=14)
            assert c['source_upper_utc'] == upper.isoformat() and c['target_lower_utc'] == lower.isoformat()
        if passed:
            accepted.append(c)
            by_target[tid].update(c['source_match_ids'])
    same(accepted, ledger['qualified_pairs'])
    for year in (2021, 2022, 2023):
        cases = [c for c in accepted if c['season'] == year]
        summary = dict(season=year, total_target_matches=int(rows.season.eq(year).sum()),
            qualified_pairs=len(cases), qualified_source_editions=len({c['source_tourney_id'] for c in cases}),
            qualified_target_editions=len({c['target_tourney_id'] for c in cases}),
            qualified_source_match_ids=len({i for c in cases for i in c['source_match_ids']}),
            target_edition_rows=len({i for c in cases for i in c['target_match_ids']}),
            direct_participant_target_rows=len({i for c in cases for i in c['direct_participant_target_ids']}),
            direct_participant_target_ids=sorted({i for c in cases for i in c['direct_participant_target_ids']}),
            model_probability_changes_not_yet_measured=True)
        same(summary, next(s for s in ledger['summary'] if s['season'] == year))
        details.append({key: value for key, value in summary.items() if not key.endswith('_ids')})
    return dict(by_target), details


def independent_rating_inputs(rows, history, extras):
    """Own K32 dictionaries, prefix sweep and extra-date batch updates."""
    records = sorted((ordinal(r.tourney_date), str(r.match_id), str(r.player_a_id),
                      str(r.player_b_id), float(r.a_won), str(r.surface), str(r.tourney_id))
                     for r in history.itertuples(index=False))
    by_id = {r[1]: r for r in records}
    groups = defaultdict(list)
    for r in records:
        groups[r[0]].append(r)
    dates = sorted(groups)
    def empty():
        return {}, {}, {}
    def copy(state):
        return tuple(dict(v) for v in state)
    def update(state, batch):
        overall, surface, count = state
        delta, surface_delta = defaultdict(float), defaultdict(float)
        for _, _, a, b, won, court, _ in batch:
            ra, rb = overall.get(a, 1500.), overall.get(b, 1500.)
            change = 32*(won-1/(1+10**((rb-ra)/400)))
            delta[a] += change; delta[b] -= change
            sa, sb = surface.get((a, court), 1500.), surface.get((b, court), 1500.)
            change = 32*(won-1/(1+10**((sb-sa)/400)))
            surface_delta[a, court] += change; surface_delta[b, court] -= change
            count[a, court] = count.get((a, court), 0)+1
            count[b, court] = count.get((b, court), 0)+1
        for p, v in delta.items():
            overall[p] = overall.get(p, 1500.)+v
        for p, v in surface_delta.items():
            surface[p] = surface.get(p, 1500.)+v
    def query(state, row):
        overall, surface, count = state
        values = []
        for p in (str(row.player_a_id), str(row.player_b_id)):
            r = overall.get(p, 1500.); n = count.get((p, row.surface), 0)
            s = surface.get((p, row.surface), 1500.)
            values.append((r, r+n/(n+20.)*(s-r)))
        return [values[0][0]-values[1][0], values[0][1]-values[1][1]]
    needed = {ordinal(g.tourney_date.iloc[0])-21 for key, g in rows.groupby('tourney_id') if key in extras}
    snapshots, stale = {}, np.zeros((len(rows), 2))
    cutoffs = np.array([ordinal(v)-21 for v in rows.tourney_date])
    state, cursor = empty(), 0
    for cutoff in sorted(set(cutoffs)):
        while cursor < len(dates) and dates[cursor] <= cutoff:
            update(state, groups[dates[cursor]]); cursor += 1
        if cutoff in needed:
            snapshots[cutoff] = copy(state)
        for i in np.flatnonzero(cutoffs == cutoff):
            stale[i] = query(state, rows.iloc[i])
    fresh = stale.copy(); direct_counts = np.zeros(len(rows), dtype=int)
    extra_counts = np.zeros(len(rows), dtype=int); edition_counts = np.zeros(len(rows), dtype=int)
    replay_metadata = []
    for tid, ids in sorted(extras.items()):
        indices = np.flatnonzero(rows.tourney_id.eq(tid).to_numpy())
        cutoff = cutoffs[indices[0]]; assert len(set(cutoffs[indices])) == 1
        batches = defaultdict(list)
        assert ids <= set(by_id)
        for identifier in sorted(ids):
            r = by_id[identifier]; assert cutoff < r[0] <= cutoff+21 and r[-1] != tid
            batches[r[0]].append(r)
        state = copy(snapshots[cutoff])
        for day in sorted(batches):
            update(state, sorted(batches[day]))
        for i in indices:
            row = rows.iloc[i]; fresh[i] = query(state, row)
            players = {str(row.player_a_id), str(row.player_b_id)}
            direct_counts[i] = sum(bool(players.intersection(by_id[j][2:4])) for j in ids)
            extra_counts[i] = len(ids); edition_counts[i] = len({by_id[j][-1] for j in ids})
        included = [r[1] for r in records if r[0] <= cutoff or r[1] in ids]
        replay_metadata.append(dict(target_tourney_id=tid, baseline_cutoff_day=str(date.fromordinal(cutoff)),
            last_included_nominal_day=str(date.fromordinal(max(batches))), extra_source_matches=len(ids),
            extra_source_editions=len({by_id[j][-1] for j in ids}), history_rows=len(included),
            history_match_ids_sha256=sha256('\n'.join(sorted(included)).encode()).hexdigest()))
    return stale, fresh, dict(extra_counts=extra_counts, edition_counts=edition_counts,
                               direct_counts=direct_counts, replays=replay_metadata)


def manual_symmetric_logistic(model, values):
    scaler, estimator = model.estimator.steps[0][1], model.estimator.steps[1][1]
    x = np.asarray(values, dtype=float)
    a = expit(((x-scaler.mean_)/scaler.scale_)@estimator.coef_[0])
    b = expit(((-x-scaler.mean_)/scaler.scale_)@estimator.coef_[0])
    return np.clip(.5*(a+1-b), 1e-8, 1-1e-8)


def independent_metrics(rows, probability):
    y = rows.a_won.to_numpy(dtype=int); p = np.asarray(probability)
    shared = score_probabilities(zip(rows.match_id, y), zip(rows.match_id, p))
    assert abs(shared['mean_log_loss']-np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))) < 1e-12
    assert shared['correct'] == int(np.count_nonzero((p >= .5) == y))
    return dict(**shared, brier_score=float(np.mean((p-y)**2)))


def independent_delta(rows, stale, fresh):
    y = rows.a_won.to_numpy(dtype=int)
    accuracy_gain = ((fresh >= .5) == y).astype(int)-((stale >= .5) == y).astype(int)
    def losses(p):
        return -y*np.log(p)-(1-y)*np.log1p(-p)
    loss_delta = losses(fresh)-losses(stale)
    return dict(accuracy_gain=float(accuracy_gain.mean()), net_correct=int(accuracy_gain.sum()),
                log_loss_delta=float(loss_delta.mean()), corrected=int(np.count_nonzero(accuracy_gain == 1)),
                newly_wrong=int(np.count_nonzero(accuracy_gain == -1)))


def independent_intervals(rows, stale, fresh):
    y = rows.a_won.to_numpy(dtype=int)
    a = ((fresh >= .5) == y).astype(int)-((stale >= .5) == y).astype(int)
    def loss(p):
        return -y*np.log(p)-(1-y)*np.log1p(-p)
    l = loss(stale)-loss(fresh)
    blocks = sorted(set(rows.tourney_date))
    counts = np.array([(rows.tourney_date == d).sum() for d in blocks])
    sums = np.array([[a[rows.tourney_date == d].sum(), l[rows.tourney_date == d].sum()] for d in blocks])
    draw = np.random.default_rng(158).integers(len(blocks), size=(2000, len(blocks)))
    distribution = sums[draw].sum(axis=1)/counts[draw].sum(axis=1)[:, None]
    return dict(accuracy_ci95=np.quantile(distribution[:, 0], [.025, .975]),
                log_loss_ci95=np.quantile(distribution[:, 1], [.025, .975]), date_blocks=len(blocks))


def check_runner_revision(out, expected):
    """Keep the actual fit source distinct from the corrected future runner."""
    correction = json.loads((out/'selection_correction.json').read_text())
    original = out/correction['original_source_snapshot']
    current = ROOT/correction['corrected_runner_path']
    assert digest(original) == correction['original_runner_sha256'] == correction['registry_runner_sha256'] == expected
    assert digest(current) == correction['corrected_runner_sha256']
    old_text, new_text = original.read_text(), current.read_text()
    reversed_text = new_text
    for item in correction['exact_reversible_replacements']:
        assert reversed_text.count(item['corrected']) == 1
        reversed_text = reversed_text.replace(item['corrected'], item['original'])
    assert reversed_text == old_text
    trees = [ast.parse(value) for value in (old_text, new_text)]
    originals = []
    for tree in trees:
        selection = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'choose_procedure']
        assert len(selection) == 1
        originals.append(selection[0])
        tree.body.remove(selection[0])
    assert ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False)
    assert ast.dump(originals[0], include_attributes=False) != ast.dump(originals[1], include_attributes=False)
    for name, value in correction['original_artifact_hashes_verified'].items():
        assert digest(out/name) == value
    assert digest(out/'self_check.json') == correction['self_check_sha256']
    return correction, originals[0]


def main():
    from scripts.train_freshness_round11 import predict_freshness
    out = ROOT/'results/round11/freshness'
    registry = json.loads((out/'search_registry.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    frozen = json.loads((out/'frozen_selection.json').read_text())
    records = json.loads((out/'fit_records.json').read_text())
    metadata = json.loads((out/'feature_metadata.json').read_text())
    self_check = json.loads((out/'self_check.json').read_text())
    for name, expected in self_check['artifact_sha256'].items():
        assert digest(out/name) == expected, name
    paths = {'protocol_sha256': 'docs/research/round11_freshness_protocol.md',
        'qualified_manifest_sha256': 'results/round11/completion/qualified_pairs.json',
        'runner_sha256': 'scripts/train_freshness_round11.py',
        'feature_code_sha256': 'scripts/freshness_round11.py',
        'elo_engine_sha256': 'scripts/advanced_features.py',
        'classifier_code_sha256': 'scripts/train_models.py',
        'inventory_sha256': 'data/processed/mens_singles_2006_2025.csv',
        'expanded_target_sha256': 'results/round3/development_targets.csv',
        'previous_feature_sha256': 'results/round5/development_features.csv',
        'previous_oof_sha256': 'results/round5/oof_predictions.csv',
        'previous_validation_sha256': 'results/round5/validation_predictions.csv',
        'previous_frozen_selection_sha256': 'results/round5/frozen_selection.json',
        'development_features_sha256': 'results/round11/freshness/development_features.csv'}
    for key, relative in paths.items():
        if key == 'runner_sha256':
            continue
        assert digest(ROOT/relative) == registry[key], key
    correction, original_selector = check_runner_revision(out, registry['runner_sha256'])
    assert registry['expanded_target_sha256'] == TARGET_SHA
    assert registry['total_fits'] == 3 and registry['paired_algorithms'] == ['overall_elo', 'rank_rating_logistic']
    assert registry['stale_training_only'] and not registry['new_mixtures'] and not registry['pooled_reselection']
    assert not registry['context_references_selection_eligible'] and registry['decision_threshold'] == .5
    assert not registry['final_test_evaluated'] and registry['prediction_years'] == [2021, 2022, 2023]
    names = [f'{a}_{v}' for a in ('overall_elo', 'rank_rating_logistic') for v in ('stale', 'fresh')]
    assert registry['probability_columns'] == names
    rows = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    assert rows.shape[0] == 36421 and rows.match_id.is_unique and rows.season.between(2009, 2023).all()
    base = pd.read_csv(ROOT/'results/round5/development_features.csv')
    assert base.shape == (36421, 326)
    development = pd.read_csv(out/'development_features.csv')
    assert development.shape == (36421, 6)
    assert list(development) == ['stale_'+c for c in COLUMNS]+['fresh_'+c for c in COLUMNS]
    stale = development[['stale_'+c for c in COLUMNS]].set_axis(COLUMNS, axis=1)
    fresh = development[['fresh_'+c for c in COLUMNS]].set_axis(COLUMNS, axis=1)
    assert error(stale, base[COLUMNS]) < 1e-12
    assert np.array_equal(stale.log_rank_diff, fresh.log_rank_diff)
    rank_error = error(stale.log_rank_diff, np.log(rows.rank_b)-np.log(rows.rank_a))
    assert rank_error < 1e-12
    ledger = json.loads((ROOT/paths['qualified_manifest_sha256']).read_text())
    assert ledger['protocol_sha256'] == registry['protocol_sha256']
    extras, qualification = independent_availability(ledger, rows)
    history = read_eligible_history(ROOT/paths['inventory_sha256'])
    inventory_lookup = history.set_index('match_id')
    expected_ids = set(history.loc[history.season.ge(2009), 'match_id'])
    assert set(rows.match_id) == expected_ids
    for name in ('player_a_id', 'player_b_id', 'surface', 'season', 'a_won'):
        assert np.array_equal(inventory_lookup.loc[rows.match_id, name].astype(str), rows[name].astype(str)), name
    source_groups = {k: set(g.match_id) for k, g in history.groupby('tourney_id')}
    for pair in ledger['qualified_pairs']:
        assert set(pair['source_match_ids']) == source_groups[pair['source_tourney_id']]
    old_ratings, new_ratings, independent_coverage = independent_rating_inputs(rows, history, extras)
    stale_error = error(old_ratings, stale[COLUMNS[1:]])
    fresh_error = error(new_ratings, fresh[COLUMNS[1:]])
    assert max(stale_error, fresh_error) < 1e-9
    same(independent_coverage['replays'], metadata['replays'])
    coverage = pd.read_csv(out/'coverage_by_match.csv')
    for column in ('match_id', 'season', 'tourney_id'):
        assert np.array_equal(coverage[column], rows[column])
    for name, key in [('extra_source_matches', 'extra_counts'),
                      ('extra_source_editions', 'edition_counts'),
                      ('direct_player_extra_matches', 'direct_counts')]:
        assert np.array_equal(coverage[name], independent_coverage[key]), name
    assert np.array_equal(coverage.has_extra_history, independent_coverage['extra_counts'] > 0)
    assert np.array_equal(coverage.rating_input_changed, np.any(stale.to_numpy() != fresh.to_numpy(), axis=1))
    assert error(stale[~coverage.has_extra_history], fresh[~coverage.has_extra_history]) == 0
    assert metadata['full_target_count'] == 36421 and metadata['qualified_pairs'] == len(ledger['qualified_pairs'])
    assert metadata['extra_unique_match_ids'] == len(set().union(*extras.values()))
    assert metadata['pair_identity_sha256'] == sha256(json.dumps(ledger['qualified_pairs'], sort_keys=True).encode()).hexdigest()
    # Additional adversarial builder check, separate from the independent math
    # above: all target outcomes poisoned, A/B exchanged, sealed rows invalid.
    from scripts.freshness_round11 import build_freshness_inputs
    poisoned = rows.copy()
    poisoned['a_won'] = 'UNREAD_TARGET_LABEL'
    for a, b in [('player_a_id', 'player_b_id'), ('rank_a', 'rank_b')]:
        poisoned[a], poisoned[b] = rows[b].copy(), rows[a].copy()
    if 'log_rank_diff' in poisoned:
        poisoned['log_rank_diff'] = -poisoned.log_rank_diff
    original_namespace = history.drop(columns=['inventory_match_id']).copy()
    original_namespace['match_id'] = history.inventory_match_id
    sentinels = pd.DataFrame([dict(source='tour', season=y, match_id='UNREAD_FINAL_ID',
        tourney_id='UNREAD_FINAL_EDITION', tourney_date='UNREAD_FINAL_DATE',
        surface='UNREAD_FINAL_SURFACE', player_a_id='UNREAD_FINAL_A',
        player_b_id='UNREAD_FINAL_B', a_won='UNREAD_FINAL_LABEL') for y in (2024, 2025)])
    guarded_inventory = pd.concat([original_namespace, sentinels], ignore_index=True)
    adversarial, adversarial_coverage, _ = build_freshness_inputs(
        poisoned, guarded_inventory, -base[COLUMNS], ledger['qualified_pairs'])
    feature_swap_error = error(adversarial, -fresh)
    assert feature_swap_error < 1e-9
    for column in coverage:
        assert np.array_equal(adversarial_coverage[column], coverage[column]), column
    expected_rows = rows[rows.season.isin((2021, 2022, 2023))]
    all_predictions = pd.read_csv(out/'all_predictions.csv')
    assert len(all_predictions) == 7504
    for column in ('match_id', 'season', 'tourney_date', 'tourney_id', 'a_won'):
        assert np.array_equal(all_predictions[column], expected_rows[column]), column
    all_predictions.index = expected_rows.index
    for filename, years in [('oof_predictions.csv', (2021, 2022)), ('validation_predictions.csv', (2023,))]:
        saved = pd.read_csv(out/filename)
        subset = all_predictions[all_predictions.season.isin(years)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, subset)
    assert len(records) == 3
    fit_checks, reload_error, head_error, swap_error, permutation_error = [], 0., 0., 0., 0.
    metrics_error, interval_error = 0., 0.
    cv = pd.read_csv(out/'temporal_cv.csv')
    comparison = pd.read_csv(out/'validation_comparison.csv')
    matched_saved = pd.read_csv(out/'matched_controls.csv')
    dates = pd.to_datetime(rows.tourney_date)
    for year, count in [(2021, 2388), (2022, 2528), (2023, 2588)]:
        valid = rows.season.eq(year)
        fit = rows.season.lt(year) & dates.le(dates[valid].min()-pd.Timedelta(days=21))
        yr = rows[valid]; assert len(yr) == count
        bundle = joblib.load(out/('frozen_models.joblib' if year == 2023 else f'model_{year}.joblib'))
        assert bundle['prediction_year'] == year and bundle['columns'] == COLUMNS
        same(bundle['registry'], registry)
        fit_metadata = next(r for r in records if r['validation_season'] == year)
        same(bundle['training_metadata'], fit_metadata)
        assert fit_metadata['train_rows'] == int(fit.sum())
        assert fit_metadata['train_max_date'] == str(dates[fit].max())
        assert fit_metadata['validation_min_date'] == str(dates[valid].min())
        assert fit_metadata['training_match_ids_sha256'] == sha256('\n'.join(rows.loc[fit, 'match_id']).encode()).hexdigest()
        training = base.loc[fit, COLUMNS]
        assert fit_metadata['training_input_sha256'] == sha256(training.to_numpy().tobytes()).hexdigest()
        assert fit_metadata['training_inputs'] == 'stale only' and int(rows.loc[fit, 'season'].max()) == year-1
        model = bundle['model']; same(model.spec, registry['spec'])
        assert model.spec['columns'] == COLUMNS and model.spec.get('half_life') is None
        scaler, linear = model.estimator.steps[0][1], model.estimator.steps[1][1]
        assert linear.C == .01 and not linear.fit_intercept and linear.n_features_in_ == 3
        augmented = np.concatenate([training.to_numpy(), -training.to_numpy()])
        scale_error = max(error(scaler.mean_, augmented.mean(axis=0)),
                          error(scaler.scale_, augmented.std(axis=0)))
        assert scale_error < 1e-9 and scaler.n_samples_seen_ == 2*int(fit.sum())
        assert not np.any(scaler.scale_ <= 0)
        loaded = predict_freshness(model, stale[valid], fresh[valid])
        saved = all_predictions.loc[yr.index, names]
        current_reload = error(loaded, saved); reload_error = max(reload_error, current_reload)
        assert current_reload < 1e-12
        manual = pd.DataFrame(index=yr.index)
        for mode, x in [('stale', stale[valid]), ('fresh', fresh[valid])]:
            manual['overall_elo_'+mode] = 1/(1+10**(-x.elo_k32_diff.to_numpy()/400))
            manual['rank_rating_logistic_'+mode] = manual_symmetric_logistic(model, x)
        current_head = error(manual[names], saved); head_error = max(head_error, current_head)
        assert current_head < 1e-12
        reverse = predict_freshness(model, -stale[valid], -fresh[valid])
        current_swap = error(loaded.to_numpy()+reverse.to_numpy(), np.ones(loaded.shape))
        swap_error = max(swap_error, current_swap); assert current_swap < 1e-12
        perm = np.random.default_rng(158+year).permutation(len(yr))
        changed = predict_freshness(model, stale[valid].iloc[perm], fresh[valid].iloc[perm]).loc[yr.index]
        current_permutation = error(loaded, changed); permutation_error = max(permutation_error, current_permutation)
        assert current_permutation < 1e-12
        assert np.array_equal(loaded.to_numpy() >= .5, saved.to_numpy() >= .5)
        assert np.array_equal(loaded.to_numpy() >= .5, changed.to_numpy() >= .5)
        no_extra = ~coverage.loc[yr.index, 'has_extra_history']
        unchanged = ~coverage.loc[yr.index, 'rating_input_changed']
        for algorithm in ('overall_elo', 'rank_rating_logistic'):
            before = saved[algorithm+'_stale'].to_numpy(); after = saved[algorithm+'_fresh'].to_numpy()
            assert np.array_equal(before[no_extra], after[no_extra])
            assert np.array_equal(before[unchanged], after[unchanged])
            delta = independent_delta(yr, before, after)
            m = matched_saved[(matched_saved.season == year)&(matched_saved.algorithm == algorithm)].iloc[0]
            assert m.matches == len(yr) and m.fresh_correct-m.stale_correct == delta['net_correct']
            assert abs(m.accuracy_improvement-delta['accuracy_gain']) < 1e-12
            assert abs(m.log_loss_change-delta['log_loss_delta']) < 1e-12
            assert m.edition_covered_rows == int((~no_extra).sum()) and m.exact_fallback_rows == int(no_extra.sum())
            assert m.direct_player_covered_rows == int(coverage.loc[yr.index, 'direct_player_extra_matches'].gt(0).sum())
            assert m.rating_changed_rows == int((~unchanged).sum())
            assert m.classification_changed_rows == int(np.count_nonzero((before >= .5) != (after >= .5)))
            block = independent_intervals(yr, before, after)
            original = summary['comparisons'][str(year)][algorithm]
            interval_error = max(interval_error, error(block['accuracy_ci95'], original['accuracy']['ci95']),
                                 error(block['log_loss_ci95'], original['log_loss']['ci95']))
            assert original['accuracy']['net_correct'] == delta['net_correct']
            assert original['accuracy']['corrected'] == delta['corrected']
            assert original['accuracy']['newly_wrong'] == delta['newly_wrong']
            assert block['date_blocks'] == original['accuracy']['date_blocks'] == original['log_loss']['date_blocks']
        table = comparison if year == 2023 else cv[cv.season.eq(year)]
        probability_columns = names+['previous_probability_recommendation', 'previous_round5_exploration']
        for name in probability_columns:
            score = independent_metrics(yr, all_predictions.loc[yr.index, name])
            saved_score = table[table.model.eq(name)].iloc[0]
            assert saved_score.matches == score['matches'] and saved_score.correct == score['correct']
            metrics_error = max(metrics_error, abs(saved_score.accuracy-score['accuracy']),
                abs(saved_score.mean_log_loss-score['mean_log_loss']), abs(saved_score.brier-score['brier_score']))
        fit_checks.append(dict(prediction_year=year, training_rows=int(fit.sum()), validation_rows=len(yr),
            scaler_max_error=scale_error, reload_max_error=current_reload, manual_head_max_error=current_head,
            swap_max_error=current_swap, permutation_max_error=current_permutation,
            no_extra_exact_fallback_rows=int(no_extra.sum()), unchanged_input_exact_fallback_rows=int(unchanged.sum())))
        if year == 2023:
            same(bundle['frozen'], frozen)
        else:
            assert bundle['frozen'] is None
    assert max(metrics_error, interval_error) < 1e-12
    early_rows = rows[rows.season.eq(2021)]; eligible = []
    for algorithm in ('overall_elo', 'rank_rating_logistic'):
        p = all_predictions.loc[early_rows.index]
        delta = independent_delta(early_rows, p[algorithm+'_stale'].to_numpy(), p[algorithm+'_fresh'].to_numpy())
        chosen = next(v for v in frozen['early_selection']['candidates'] if v['algorithm'] == algorithm)
        assert abs(chosen['accuracy_improvement']-delta['accuracy_gain']) < 1e-12
        assert abs(chosen['log_loss_change']-delta['log_loss_delta']) < 1e-12
        allowed = delta['net_correct'] > 0 and delta['log_loss_delta'] <= .005+1e-12
        assert chosen['eligible'] is allowed
        if allowed:
            eligible.append((-delta['net_correct'], delta['log_loss_delta'], algorithm))
    choice = min(eligible)[2] if eligible else None
    assert frozen['early_selection']['choice'] == choice
    assert frozen['early_selection']['selection_rows'] == 2388
    assert frozen['early_selection']['selection_season'] == 2021
    gate = frozen['procedure_gate']; assert gate['selected_algorithm'] == choice
    passed = False
    if choice:
        gate_rows = rows[rows.season.eq(2022)]; p = all_predictions.loc[gate_rows.index]
        delta = independent_delta(gate_rows, p[choice+'_stale'].to_numpy(), p[choice+'_fresh'].to_numpy())
        passed = delta['net_correct']/len(gate_rows) >= .005-1e-12 and delta['log_loss_delta'] <= .005+1e-12
        assert abs(gate['accuracy_improvement']-delta['accuracy_gain']) < 1e-12
        assert abs(gate['log_loss_change']-delta['log_loss_delta']) < 1e-12
    assert gate['gate_passed'] is bool(passed)
    assert frozen['recommendation'] == (choice+'_fresh' if passed else 'previous_probability_recommendation')
    assert not frozen['pooled_selection_performed'] and not summary['pooled_selection_performed']
    same(summary['procedure_gate'], gate)
    # Re-evaluate only selection functions, never main/fit, in both source versions.
    from scripts import train_freshness_round11 as runner
    namespace = dict(vars(runner))
    exec(compile(ast.Module(body=[original_selector], type_ignores=[]),
                 str(out/'original_train_freshness_round11.py'), 'exec'), namespace)
    oof_rows = rows[rows.season.isin((2021, 2022))]
    p = all_predictions.loc[oof_rows.index, names]
    original_choice = namespace['choose_procedure'](oof_rows, p)
    corrected_choice = runner.choose_procedure(oof_rows, p)
    same(original_choice, {key: frozen[key] for key in original_choice})
    same(original_choice, correction['original_selection_recomputed'])
    same(corrected_choice, correction['corrected_selection_recomputed'])
    assert original_choice['early_selection']['choice'] == corrected_choice['early_selection']['choice'] == choice
    same(original_choice['procedure_gate'], corrected_choice['procedure_gate'])
    assert original_choice['recommendation'] == corrected_choice['recommendation'] == frozen['recommendation']
    # Contexts are only descriptive; independently reproduce the inherited blend.
    old_choice = json.loads((ROOT/'results/round5/frozen_selection.json').read_text())['pooled_selection']
    contextual_error = 0.
    for filename, years in [('oof_predictions.csv', (2021, 2022)), ('validation_predictions.csv', (2023,))]:
        old = pd.read_csv(ROOT/'results/round5'/filename).set_index('match_id')
        ids = rows.loc[rows.season.isin(years), 'match_id']
        current = all_predictions.loc[rows.index[rows.season.isin(years)]]
        aligned = old.loc[ids]
        r5 = np.sum(aligned[old_choice['components']].to_numpy()*np.asarray(old_choice['weights']), axis=1)
        if years == (2023,):
            assert error(r5, aligned[old_choice['choice']]) < 1e-14
            r5 = aligned[old_choice['choice']].to_numpy()
        contextual_error = max(contextual_error, error(r5, current.previous_round5_exploration),
                               error(aligned.previous_probability_recommendation, current.previous_probability_recommendation))
    assert contextual_error < 1e-14
    audit = dict(status='passed', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        optimizer_calls_repeated=0, qualification_candidate_pairs_checked=763,
        qualified_pair_count=len(ledger['qualified_pairs']), qualification_by_year=qualification,
        independent_history_rows=len(history), inventory_namespace='tour: prefix verified then removed for ledger identity',
        independent_stale_rating_values=int(old_ratings.size), independent_fresh_rating_values=int(new_ratings.size),
        stale_rating_max_error=stale_error, fresh_rating_max_error=fresh_error, rank_formula_max_error=rank_error,
        all_target_feature_swap_max_error=feature_swap_error,
        all_target_labels_poisoned_and_sealed_year_sentinels_ignored=True,
        old_selected_three_column_max_error=error(stale, base[COLUMNS]),
        all_original_326_columns_recomputed=False, saved_fits_checked=3, fit_checks=fit_checks,
        probability_rows_checked=7504, new_probability_columns_checked=4,
        contextual_reference_columns_checked=2, probability_reload_max_error=reload_error,
        independent_linear_head_max_error=head_error, swap_max_error=swap_error,
        permutation_max_error=permutation_error, classifications_unchanged_on_reload_and_permutation=True,
        shared_metric_max_error=metrics_error, independently_recomputed_interval_max_error=interval_error,
        contextual_reference_max_error=contextual_error, early_choice=choice, gate_passed=bool(passed),
        post_run_selection_correction=dict(original_fit_runner_sha256=registry['runner_sha256'],
            current_runner_sha256=digest(ROOT/'scripts/train_freshness_round11.py'),
            exact_original_snapshot_verified=True, only_choose_procedure_changed=True,
            both_versions_recomputed_on_real_oof=True, actual_choice_gate_recommendation_unchanged=True,
            original_run_used_corrected_selector=False, refitting_performed=False),
        recommendation=frozen['recommendation'], pooled_reselection=False,
        final_year_outcomes_parsed_by_this_audit=False,
        limits=['No optimizer is rerun and no training optimization trajectory is replayed.',
                'The independent Elo implementation uses the same published eligible inventory and canonical event dates; it does not re-adjudicate all original match labels.',
                'Eligibility arithmetic and recorded evidence normalization are independently recomputed; public-result availability and schedule-consistency remain Tier B assumptions.',
                'Only the three selected old input fields are rebuilt/checked, not all326 prior features.',
                'No new network source verification is performed by this script; single-reviewer Pune/Acapulco evidence limits remain.',
                'Intervals are conditional exploratory date-block intervals without correction for repeated model search.'],
        audit_code_sha256=digest(Path(__file__)),
        artifact_sha256={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifact_audit.json'},
        source_hashes={relative: digest(ROOT/relative) for relative in paths.values()})
    (out/'artifact_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: audit[k] for k in ('status', 'probability_reload_max_error', 'stale_rating_max_error',
        'fresh_rating_max_error', 'shared_metric_max_error', 'early_choice', 'gate_passed')}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

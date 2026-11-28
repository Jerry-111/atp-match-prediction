"""Read-only timing audit of frozen R8 edges; no odds, outcomes, fitting or scoring.

The source UTC field is provider-reported match time, not independently verified
actual start/completion time. This audit checks the narrower nominal-date guard.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.train_models import ROOT, sha256

TARGET_COLUMNS = ['match_id', 'season', 'tourney_id', 'tourney_date', 'round',
                  'player_a_id', 'player_b_id', 'player_a_name', 'player_b_name']
EDGE_COLUMNS = ['match_id', 'source_match_id', 'source_season', 'match_utc']
ROUND_ORDER = {r: i for i, r in enumerate(('R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F'))}


def snapshot_counts(rows, history):
    """Count logical edge uses before replicating six graph/surface settings."""
    cutoffs = rows.nominal_date-pd.Timedelta(days=21)
    result = dict(target_rows=len(rows), snapshots=int(cutoffs.nunique()),
                  nonempty_snapshots=0, snapshot_edge_uses=0, target_edge_uses=0,
                  violating_snapshot_edge_uses=0, violating_target_edge_uses=0,
                  affected_snapshots=0, affected_targets=0)
    violations, nominal_ages, timestamp_ages = [], [], []
    for cutoff, target_count in cutoffs.value_counts().sort_index().items():
        admitted = history.utc.le(cutoff)
        bad = admitted & history.nominal_date.gt(cutoff)
        count, bad_count = int(admitted.sum()), int(bad.sum())
        result['nonempty_snapshots'] += int(count > 0)
        result['snapshot_edge_uses'] += count
        result['target_edge_uses'] += count*int(target_count)
        result['violating_snapshot_edge_uses'] += bad_count
        result['violating_target_edge_uses'] += bad_count*int(target_count)
        result['affected_snapshots'] += int(bad_count > 0)
        result['affected_targets'] += int(target_count)*int(bad_count > 0)
        if count:
            target_date = cutoff+pd.Timedelta(days=21)
            nominal_ages.append((target_date-history.loc[admitted, 'nominal_date']).dt.total_seconds().min()/86400)
            timestamp_ages.append((target_date-history.loc[admitted, 'utc']).dt.total_seconds().min()/86400)
        for match_id in history.loc[bad, 'match_id']:
            violations.append(dict(cutoff=cutoff.isoformat(), edge_match_id=match_id,
                                   target_count=int(target_count)))
    result.update(unique_edges_ever_admitted=int(history.utc.le(cutoffs.max()).sum()),
                  minimum_nominal_date_age_days=float(min(nominal_ages)) if nominal_ages else None,
                  minimum_provider_timestamp_age_days=float(min(timestamp_ages)) if timestamp_ages else None,
                  violations=violations)
    return result


def chronology_checks(history):
    appearances = []
    for side in ('a', 'b'):
        columns = ['match_id', 'source_match_id', 'tourney_id', 'round', 'utc']
        part = history[columns].copy()
        part['player_id'] = history[f'player_{side}_id']
        part['player_name'] = history[f'player_{side}_name']
        appearances.append(part)
    appearances = pd.concat(appearances, ignore_index=True)
    appearances['stage'] = appearances['round'].map(ROUND_ORDER)
    assert appearances.stage.notna().all()
    same_groups = []
    for (event, player, utc), group in appearances.groupby(['tourney_id', 'player_id', 'utc']):
        if group['round'].nunique() > 1:
            same_groups.append(dict(tourney_id=event, player_id=player, utc=utc.isoformat(),
                                    matches=group[['match_id', 'round']].to_dict('records')))
    nonincreasing = []
    for _, group in appearances.groupby(['tourney_id', 'player_id']):
        group = group.sort_values(['stage', 'match_id'])
        previous = group.shift()
        bad = group.utc.le(previous.utc) & group.stage.gt(previous.stage)
        for index, row in group[bad].iterrows():
            prior = previous.loc[index]
            nonincreasing.append(dict(player_id=row.player_id, player_name=row.player_name,
                earlier_round_match_id=prior.match_id, earlier_round=prior['round'],
                earlier_round_utc=prior.utc.isoformat(), later_round_match_id=row.match_id,
                later_round=row['round'], later_round_utc=row.utc.isoformat()))
    return dict(same_player_different_round_same_utc_groups=len(same_groups),
                same_utc_appearance_rows=sum(len(g['matches']) for g in same_groups),
                nonincreasing_own_round_transitions=len(nonincreasing),
                same_utc_examples=same_groups, nonincreasing_examples=nonincreasing)


def main():
    out = ROOT/'results/round8/market_graph'
    target_path = ROOT/'results/round3/development_targets.csv'
    edge_path = out/'opening_history.csv'
    link_path = ROOT/'results/round5/market/linked_matches.csv'
    rows = pd.read_csv(target_path, usecols=TARGET_COLUMNS)
    history = pd.read_csv(edge_path, usecols=EDGE_COLUMNS)
    links = pd.read_csv(link_path, usecols=['match_id', 'season', 'status', 'market_match_utc'])
    registry = json.loads((out/'search_registry.json').read_text())
    assert sha256(target_path) == registry['target_sha256']
    assert sha256(link_path) == registry['linkage_sha256']
    assert rows.season.between(2009, 2023).all() and len(rows) == 36421
    assert history.source_season.isin([2021, 2022, 2023]).all() and len(history) == 7341
    assert not rows.match_id.duplicated().any() and not history.match_id.duplicated().any()
    rows['nominal_date'] = pd.to_datetime(rows.tourney_date, utc=True, errors='raise')
    date = rows.nominal_date.dt
    assert (date.year.eq(rows.season) | (date.year.eq(rows.season-1)
            & date.month.eq(12) & date.day.ge(25))).all()
    history = history.merge(rows, on='match_id', validate='one_to_one')
    assert len(history) == 7341 and history.source_season.eq(history.season).all()
    history['utc'] = pd.to_datetime(history.match_utc, utc=True, errors='raise')
    assert history.utc.dt.year.le(2023).all()
    history['offset_days'] = (history.utc-history.nominal_date).dt.total_seconds()/86400
    period_masks = dict(all_development=rows.season.le(2023),
        early_2021H2=rows.season.eq(2021) & rows.nominal_date.ge('2021-07-01'),
        gate_2022=rows.season.eq(2022), exploratory_2023=rows.season.eq(2023))
    counts = {key: snapshot_counts(rows[mask], history) for key, mask in period_masks.items()}
    locked = snapshot_counts(rows, history[history.utc.lt('2023-01-01')])
    assert all(x['violating_snapshot_edge_uses'] == 0 for x in [*counts.values(), locked])
    assert counts['exploratory_2023']['target_rows'] == 2588
    coverage = {}
    for year in (2021, 2022, 2023):
        ids = rows.loc[rows.season.eq(year), 'match_id']
        subset = links[links.match_id.isin(ids)]
        assert len(subset) == len(ids) and not subset.match_id.duplicated().any()
        matched = subset.status.eq('matched')
        assert pd.to_datetime(subset.loc[matched, 'market_match_utc'], utc=True, errors='raise').notna().all()
        coverage[str(year)] = dict(targets=len(ids), matched_time=int(matched.sum()),
                                  unmatched_time=int((~matched).sum()))
    def offsets(frame):
        return frame[['match_id', 'source_match_id', 'tourney_date', 'match_utc', 'offset_days']].to_dict('records')
    result = dict(status='passed_with_timestamp_provenance_limitations',
        purpose='Audit existing edge inclusion; no new model or availability rule implemented.',
        source_rows=len(history), periods=counts, locked_pre2023=locked,
        offset_days=dict(min=float(history.offset_days.min()), max=float(history.offset_days.max()),
            negative_rows=int(history.offset_days.lt(0).sum()),
            negative_by_season={str(y): int(n) for y, n in history[history.offset_days.lt(0)].groupby('source_season').size().items()},
            floor_day_histogram={str(int(day)): int(n) for day, n in np.floor(history.offset_days).value_counts().sort_index().items()},
            earliest_examples=offsets(history.nsmallest(12, 'offset_days')),
            latest_examples=offsets(history.nlargest(8, 'offset_days'))),
        chronology=chronology_checks(history), linked_time_coverage=coverage,
        conclusions=dict(dual_cutoff_would_remove_edges=False,
            existing_numerical_predictions_changed=False,
            independent_actual_start_or_completion_proven=False,
            original_nominal_tournament_date_guard_satisfied=True,
            safe_to_shorten_lag_on_provider_utc_alone=False),
        limitations=[
            'tourney_date is the linked, corrected repository tournament date; it is not a per-match timestamp or an independently certified end date.',
            'Negative offsets can be legitimate Sunday starts or timezone effects; they are not individually labelled source errors.',
            'A provider match-time field with contradictory own-round chronology cannot certify actual starts, finishes, quote capture times or publication time.',
            'Zero nominal-date violations establishes the existing guard, not universal proof that every historical quote/result was available at the earlier cutoff.',
            'Edge-use counts share one logical graph per cutoff before six setting/surface repetitions; target-edge uses weight each cutoff by its target count.',
            'No raw source, odds, target outcomes, score, duration or 2024/2025 final files are read by this timing audit.'],
        input_columns=dict(targets=TARGET_COLUMNS, edges=EDGE_COLUMNS,
                           links=['match_id', 'season', 'status', 'market_match_utc']),
        artifact_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
            (target_path, edge_path, link_path, out/'search_registry.json')},
        audit_code_sha256=sha256(ROOT/'scripts/audit_market_timing_round8.py'))
    (out/'timing_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], source_rows=len(history),
                         periods=counts, chronology_counts={k: v for k, v in result['chronology'].items() if not isinstance(v, list)}), indent=2))


if __name__ == '__main__':
    main()

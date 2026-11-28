"""Freeze bounded Tier B availability from reviewed metadata; no outcomes read."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/round11/completion'
TARGET_SHA = '9bcb3ad4b0fbe745b23b776a750256f92ab9acc9fd1cb57aed0a16cef86435d2'
PROTOCOL = ROOT / 'docs/research/round11_freshness_protocol.md'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def midnight(day):
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def normalize_probe(probe):
    sources = {}
    for case in probe['cases']:
        evidence = case.get('completion_evidence', [case])
        full = [e for e in evidence if e.get('inspection') == 'full_text']
        day, pub = case.get('completed_by_day'), case.get('displayed_publication_day')
        if day and pub and full:
            sources[case['source_tourney_id']] = dict(
                completed_by_day=day, displayed_publication_day=pub,
                upper_day=max(day, pub), evidence=full,
                tier='B_global_day_assumed_public_stable_results',
                publication_timezone=None, historical_snapshot=False)
    targets = {}
    def add_target(identifier, day, evidence):
        if identifier in targets:
            targets[identifier]['lower_day'] = min(targets[identifier]['lower_day'], day)
            targets[identifier]['evidence'].extend(evidence)
        else:
            targets[identifier] = dict(lower_day=day, evidence=list(evidence))

    for case in probe['cases']:
        for evidence in case.get('target_start_evidence', []):
            day = evidence.get('announced_main_start_day') or evidence.get('reported_main_start_day')
            if (evidence.get('inspection') == 'full_text' and day
                    and evidence.get('start_semantics', '').startswith('explicit_')):
                add_target(case['source_tourney_id'], day, [evidence])
    for item in probe.get('target_start_evidence', []):
        if item.get('inspection') == 'full_text' and item.get('target_lower_day'):
            add_target(item['target_tourney_id'], item['target_lower_day'], [item])
    for item in probe.get('authorized_target_start_followup', {}).get('targets', []):
        if not item.get('start_supported'):
            continue
        evidence = [e for e in item['target_start_evidence'] if e.get('inspection') == 'full_text'
                    and e.get('announced_main_start_day')]
        if evidence:
            # The earliest supported lower bound is conservative if several differ.
            add_target(item['target_tourney_id'],
                       min(e['announced_main_start_day'] for e in evidence), evidence)
    return sources, targets


def main():
    paths = [ROOT/'results/round10/completion/source_frame.csv',
             ROOT/'results/round10/completion/source_frame_2023_unqualified.csv']
    pinned_frames = {
        'source_frame.csv': '5ab34f830540fa79feec12eeee8e6f53b0a9d2acd09ad68bc0c4cc89c5ee507f',
        'source_frame_2023_unqualified.csv': 'ea7922107d77f113251017d01ecd96144d4acc49462668dd39066da6344b09a6'}
    assert all(digest(p) == pinned_frames[p.name] for p in paths)
    target_path = ROOT/'results/round3/development_targets.csv'
    assert digest(target_path) == TARGET_SHA
    rows = pd.read_csv(target_path, usecols=['match_id', 'season', 'tourney_id', 'tourney_date',
                                          'player_a_id', 'player_b_id'])
    assert len(rows) == 36421 and rows.match_id.is_unique
    frames = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(frames) == 763 and not frames.duplicated(['source_tourney_id','target_tourney_id']).any()
    sources, targets, probes, hashes = {}, {}, {}, {}
    for year, count in [(2021,57), (2022,65), (2023,62)]:
        path = OUT/f'probe_{year}_first10.json'
        probe = json.loads(path.read_text())
        expected_frame = paths[0] if year < 2023 else paths[1]
        probe_hashes = probe['input_sha256']
        assert probe_hashes.get('source_frame', probe_hashes.get(str(expected_frame.relative_to(ROOT)))) == digest(expected_frame)
        raw_path = ROOT/f'data/raw/{year}.csv'
        assert probe_hashes.get(f'raw_{year}', probe_hashes.get(str(raw_path.relative_to(ROOT)))) == digest(raw_path)
        scope = frames.loc[frames.season.eq(year), ['source_tourney_id','source_nominal_date']]
        scope = scope.drop_duplicates().sort_values(['source_nominal_date','source_tourney_id'])
        expected = scope.source_tourney_id.tolist()
        assert len(expected) == count
        assert [v['source_tourney_id'] for v in probe['worklist']] == expected
        assert [v['source_tourney_id'] for v in probe['cases']] == expected[:10]
        assert not any(v['admitted_for_prediction'] for v in probe['worklist'])
        ss, tt = normalize_probe(probe)
        assert set(ss).issubset(expected[:10])
        sources.update(ss); targets.update(tt); probes[year] = probe
        hashes[str(path.relative_to(ROOT))] = digest(path)
    cases=[]
    for row in frames.to_dict('records'):
        sid, tid = row['source_tourney_id'], row['target_tourney_id']
        source_ids, target_ids = json.loads(row['source_match_ids']), json.loads(row['target_match_ids'])
        source = rows.loc[rows.tourney_id.eq(sid)]
        target = rows.loc[rows.tourney_id.eq(tid)]
        assert source.match_id.tolist() == source_ids and target.match_id.tolist() == target_ids
        assert source.tourney_date.nunique() == target.tourney_date.nunique() == 1
        assert source.tourney_date.iloc[0] == row['source_nominal_date']
        assert target.tourney_date.iloc[0] == row['target_nominal_date']
        assert int(row['season']) == int(tid[:4]) == int(target.season.iloc[0])
        assert int(row['source_season']) == int(sid[:4]) == int(source.season.iloc[0])
        assert midnight(row['original_T_minus_21_date']) == midnight(row['target_nominal_date']) - timedelta(days=21)
        assert midnight(row['original_T_minus_21_date']) < midnight(row['source_nominal_date']) <= midnight(row['target_nominal_date'])
        assert not set(source_ids).intersection(target_ids)
        item = dict(season=int(row['season']),source_tourney_id=sid,target_tourney_id=tid,
                    source_nominal_date=row['source_nominal_date'],target_nominal_date=row['target_nominal_date'],
                    original_T_minus_21_date=row['original_T_minus_21_date'],
                    source_match_ids=source_ids,target_match_ids=target_ids,
                    admitted_for_prediction=False)
        if sid not in sources:
            item['reason']='source_not_qualified_in_fixed_first10_probe'
        elif tid not in targets:
            item['reason']='target_lower_day_unresolved'
        else:
            upper=midnight(sources[sid]['upper_day'])+timedelta(hours=36)
            lower=midnight(targets[tid]['lower_day'])-timedelta(hours=14)
            item.update(source_upper_utc=upper.isoformat(),target_lower_utc=lower.isoformat(),
                        source_upper_inclusive=True,target_lower_inclusive=True,
                        clearance_hours=(lower-upper).total_seconds()/3600)
            item['admitted_for_prediction']=upper<lower
            item['reason']='qualified_under_frozen_Tier_B_assumptions' if upper<lower else 'bounds_overlap'
            assert midnight(row['source_nominal_date'])>midnight(row['original_T_minus_21_date'])
        people=set(source.player_a_id)|set(source.player_b_id)
        direct=target.player_a_id.isin(people)|target.player_b_id.isin(people)
        item['direct_participant_target_ids']=target.loc[direct,'match_id'].tolist()
        cases.append(item)
    summary=[]
    for year in (2021,2022,2023):
        cc=[c for c in cases if c['season']==year and c['admitted_for_prediction']]
        target_ids=set(i for c in cc for i in c['target_match_ids'])
        direct=set(i for c in cc for i in c['direct_participant_target_ids'])
        summary.append(dict(season=year,total_target_matches=int(rows.season.eq(year).sum()),
            qualified_pairs=len(cc),qualified_source_editions=len({c['source_tourney_id'] for c in cc}),
            qualified_target_editions=len({c['target_tourney_id'] for c in cc}),
            qualified_source_match_ids=len({i for c in cc for i in c['source_match_ids']}),
            target_edition_rows=len(target_ids),direct_participant_target_rows=len(direct),
            direct_participant_target_ids=sorted(direct),model_probability_changes_not_yet_measured=True))
    for path in [*paths,target_path,PROTOCOL,Path(__file__)]:
        hashes[str(path.relative_to(ROOT))]=digest(path)
    payload=dict(schema_version='round11_bounded_global_Tier_B_v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        root_limited_experiment_authorized=True,strict_asof_reconstruction=False,
        assumptions=['Completed-draw public winner outcomes were available and stable within the dated report global day bound.',
                     'Retrospective target-day statements or announced periods bound earliest main play; no earlier rescheduling.',
                     'Unchanged stale comparator retains its original T-minus-21 assumptions.'],
        protocol_sha256=digest(PROTOCOL),input_sha256=hashes,sources=sources,targets=targets,
        cases=cases,qualified_pairs=[c for c in cases if c['admitted_for_prediction']],summary=summary,
        models_fitted_at_freeze=0,final_test_outcomes_opened=False)
    output=OUT/'qualified_pairs.json'
    if output.exists():
        raise FileExistsError('Do not overwrite the pre-fit availability freeze')
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in s.items() if not k.endswith('_ids')} for s in summary],indent=2))
    print('Frozen availability SHA256:',digest(output))


if __name__ == '__main__':
    main()

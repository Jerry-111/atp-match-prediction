"""Independent raw-count sample audit plus full archive chronology checks.

Does not import the sequence builder, its token aggregation, or technical_pair.
Uses the established metadata correction policy, then independently parses the
pinned annual CSVs and computes the stated 22 channels for a diverse fixed sample.
No model is fitted, no prediction is scored, and no final-year raw file is opened.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import date,datetime,timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.model_event_metadata import corrected_tour_row

ROOT=Path(__file__).resolve().parents[1]
SURFACES=('Hard','Clay','Grass','Carpet')
PRIORS=(.06,.08,.60,.70,.50,.30,.50)
COLUMNS=('win_rate','log_matches','log_own_rank','log_opponent_rank','serve_rate','return_rate',
         'ace','double_fault','first_in','first_won','second_won','return_first_won',
         'return_second_won','log_serve_points','log_return_points','missing_stats_fraction',
         'surface_hard','surface_clay','surface_grass','surface_carpet','log_gap_days','log_age_days')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_date(value):
    s=str(value)
    return datetime.strptime(s,'%Y%m%d').date() if len(s)==8 else date.fromisoformat(s[:10])


def manual_technical(raw):
    """Direct physical counts, independently of production numerator vectors."""
    parsed={}
    for side in ('w','l'):
        try:
            a,d,n,f,fw,sw=[float(raw[side+'_'+key]) for key in ('ace','df','svpt','1stIn','1stWon','2ndWon')]
        except (ValueError,TypeError,KeyError):return None
        numbers=(a,d,n,f,fw,sw)
        if any(not math.isfinite(v) or v<0 or not v.is_integer() for v in numbers):return None
        if n<=0 or f>n or fw>f or sw>n-f or d+sw>n-f or a>fw+sw:return None
        parsed[side]=dict(ace=a,df=d,total=n,first_in=f,first_won=fw,second_won=sw)
    return parsed


def manual_token(records,gap,age):
    n=len(records);wins=sum(r['won'] for r in records)
    serve_won=serve_total=return_won=return_total=0.
    nums=np.zeros(7);dens=np.zeros(7);valid=0
    for r in records:
        pair=r['technical']
        if pair is None:continue
        valid+=1
        own=pair[r['raw_side']];other=pair['l' if r['raw_side']=='w' else 'w']
        os=own['total']-own['first_in'];rs=other['total']-other['first_in']
        serve_won+=own['first_won']+own['second_won'];serve_total+=own['total']
        return_won+=other['total']-other['first_won']-other['second_won'];return_total+=other['total']
        nums+=np.array([own['ace'],own['df'],own['first_in'],own['first_won'],own['second_won'],
                       other['first_in']-other['first_won'],rs-other['second_won']])
        dens+=np.array([own['total'],os,own['total'],own['first_in'],os,other['first_in'],rs])
    smooth=(nums+100*np.array(PRIORS))/(dens+100)
    return np.array([(wins+1)/(n+2),math.log1p(n),
        math.fsum(math.log(r['rank']) for r in records)/n,
        math.fsum(math.log(r['opponent_rank']) for r in records)/n,
        (serve_won+65)/(serve_total+100),(return_won+35)/(return_total+100),*smooth,
        math.log1p(serve_total),math.log1p(return_total),(n-valid)/n,
        *[sum(r['surface']==s for r in records)/n for s in SURFACES],
        math.log1p(gap),math.log1p(age)],dtype=np.float32)


def main():
    out=ROOT/'results/round9/sequence';registry=json.loads((out/'search_registry.json').read_text())
    metadata=registry['source_metadata']
    paths=dict(feature_code=ROOT/'scripts/sequence_features.py',learner=ROOT/'scripts/sequence_learning.py',
               target=ROOT/'results/round3/development_targets.csv',sequence=out/'sequence_features.npz')
    for name,path in paths.items():
        expected=registry['feature_code_sha256' if name=='feature_code' else name+'_sha256']
        if sha(path)!=expected:raise ValueError(f'Frozen {name} changed')
    inventory_path=ROOT/'data/processed/mens_singles_2006_2025.csv'
    if sha(inventory_path)!=metadata['inventory_sha256']:raise ValueError('Inventory hash changed')
    inventory=pd.read_csv(inventory_path,dtype={'a_won':'string'})
    inventory=inventory[inventory.source.eq('tour')&inventory.season.between(2006,2023)].copy()
    targets=pd.read_csv(paths['target'])
    with np.load(paths['sequence'],allow_pickle=False) as archive:
        values=archive['values'];mask=archive['mask'];days=archive['dates'];ids=archive['match_ids'];indices=archive['indices']
        if tuple(archive['columns'])!=COLUMNS:raise ValueError('Token column order changed')
    assert values.shape==(36421,2,16,22) and np.array_equal(ids,targets.match_id)
    assert np.array_equal(indices,targets.index) and mask.dtype==bool
    cutoffs=np.array([(parse_date(d)-timedelta(days=21)).toordinal() for d in targets.tourney_date])
    assert np.all(values[~mask]==0) and np.all(days[~mask]==-1)
    assert np.isfinite(values).all() and not np.any(np.diff(mask.astype(int),axis=2)<0)
    assert not np.any(mask&(days>cutoffs[:,None,None]))
    assert not np.any(mask&(days<cutoffs[:,None,None]-1460))
    assert not np.any((np.diff(days,axis=2)<=0)&mask[:,:,1:]&mask[:,:,:-1])
    ages=np.maximum(cutoffs[:,None,None]-days,0)
    age_error=float(np.max(np.abs(values[:,:,:,21][mask]-np.log1p(ages[mask]).astype(np.float32))))
    expected_gaps=np.zeros_like(days)
    expected_gaps[:,:,1:]=np.where(mask[:,:,1:]&mask[:,:,:-1],np.diff(days,axis=2),0)
    gap_error=float(np.max(np.abs(values[:,:,:,20][mask]-np.log1p(expected_gaps[mask]).astype(np.float32))))
    assert age_error==0 and gap_error==0
    earliest=date.fromordinal(int(cutoffs.min())-1460);latest=date.fromordinal(int(cutoffs.max()))
    expected={str(r['match_id']):r for r in inventory.to_dict('records') if earliest<=parse_date(r['tourney_date'])<=latest}
    by_player=defaultdict(lambda:defaultdict(list));seen=set();invalid=0;source_hashes={}
    for entry in metadata['source_files']:
        year=int(entry['season'])
        if year not in range(2006,2024):raise ValueError('Forbidden final-year source')
        path=ROOT/'data/raw'/f'{year}.csv'
        if sha(path)!=entry['sha256']:raise ValueError('Raw source checksum changed')
        source_hashes[str(year)]=entry['sha256']
        with path.open(encoding='utf-8-sig',newline='') as handle:
            for original in csv.DictReader(handle):
                key=f"tour:{original['tourney_id']}:{int(original['match_num'])}"
                if key not in expected:continue
                assert key not in seen;seen.add(key)
                raw=corrected_tour_row(original);row=expected[key]
                a,b=str(row['player_a_id']),str(row['player_b_id']);w,l=raw['winner_id'],raw['loser_id']
                day=parse_date(raw['tourney_date'])
                assert {a,b}=={w,l} and day==parse_date(row['tourney_date'])
                assert int(row['season'])==year and raw['surface']==row['surface']
                assert int(w==a)==int(row['a_won'])
                pair=manual_technical(raw);invalid+=pair is None
                for player,opponent,rank,orank in [(a,b,row['rank_a'],row['rank_b']),(b,a,row['rank_b'],row['rank_a'])]:
                    assert math.isfinite(float(rank)) and float(rank)>0
                    by_player[player][day].append(dict(key=key,won=int(player==w),rank=float(rank),
                        opponent_rank=float(orank),surface=raw['surface'],technical=pair,raw_side='w' if player==w else 'l'))
    assert seen==set(expected) and len(seen)==metadata['historical_matches']
    assert invalid==metadata['missing_statistic_matches']
    assert sum(len(groups) for groups in by_player.values())==metadata['player_date_groups']
    sample=set();selection_reasons=defaultdict(list)
    def add(index,reason):
        sample.add(int(index));selection_reasons[int(index)].append(reason)
    # Fixed, label-free diversity sample: time/surface and metadata edge cases.
    for year in (2009,2016,2020,2021,2022,2023):
        subset=targets[targets.season.eq(year)]
        for surface in sorted(subset.surface.unique()):
            candidates=subset.index[subset.surface.eq(surface)].to_numpy()
            for index in (candidates[0],candidates[-1]):add(index,f'{year}_{surface}_endpoint')
    candidates=np.flatnonzero(targets.season.eq(2023))
    for index in np.random.default_rng(9009).choice(candidates,12,replace=False):add(index,'fixed_random_2023')
    conditions={
        'zero_history':(~mask.any(2)).any(1),
        'missing_statistics':(values[:,:,:,15]>0).any((1,2)),
        'tied_date_multiple_matches':(values[:,:,:,1]>math.log(2)+1e-6).any((1,2)),
        'short_nonzero_history':((mask.sum(2)>0)&(mask.sum(2)<16)).any(1),
        'exact_cutoff':(mask&(days==cutoffs[:,None,None])).any((1,2)),
        'exact_window_boundary':(mask&(days==cutoffs[:,None,None]-1460)).any((1,2)),
        'carpet_history':(values[:,:,:,19]>0).any((1,2)),
        'corrected_2019_washington_date':(mask&(days==date(2019,7,29).toordinal())).any((1,2))}
    for reason,condition in conditions.items():
        candidates=np.flatnonzero(condition)
        if len(candidates):
            for index in (candidates[0],candidates[-1]):add(index,reason)
    checks=[];max_error=0.;sampled_tokens=0;sampled_rows=set()
    for index in sorted(sample):
        row=targets.iloc[index];cutoff=date.fromordinal(int(cutoffs[index]));first=cutoff-timedelta(days=1460)
        rebuilt=np.zeros((2,16,22),np.float32);rebuilt_mask=np.zeros((2,16),bool);rebuilt_dates=np.full((2,16),-1,np.int32)
        source_matches=0
        for side,player in enumerate((str(row.player_a_id),str(row.player_b_id))):
            selected=sorted(d for d in by_player[player] if first<=d<=cutoff)[-16:]
            for position,day in enumerate(selected):
                records=sorted(by_player[player][day],key=lambda r:r['key'])
                gap=0 if position==0 else (day-selected[position-1]).days
                rebuilt[side,16-len(selected)+position]=manual_token(records,gap,(cutoff-day).days)
                rebuilt_mask[side,16-len(selected)+position]=True
                rebuilt_dates[side,16-len(selected)+position]=day.toordinal()
                source_matches+=len(records);sampled_tokens+=1;sampled_rows.update(r['key'] for r in records)
        assert np.array_equal(rebuilt_mask,mask[index]) and np.array_equal(rebuilt_dates,days[index])
        error=float(np.max(np.abs(rebuilt-values[index])));max_error=max(max_error,error)
        assert error<=1e-6,(row.match_id,error)
        checks.append(dict(match_id=str(row.match_id),season=int(row.season),surface=str(row.surface),
            cutoff=str(cutoff),reasons=selection_reasons[index],valid_tokens=int(rebuilt_mask.sum()),
            historical_match_appearances=source_matches,max_abs_channel_error=error))
    report=dict(status='passed',scope='Full archive identity/mask/time/age/gap checks; independently recomputed all 22 channels for a diverse metadata-selected sample, not the full token archive',
        targets=len(targets),sampled_targets=len(sample),sampled_tokens=sampled_tokens,
        distinct_sampled_source_matches=len(sampled_rows),max_abs_channel_error=max_error,
        full_archive_age_error=age_error,full_archive_gap_error=gap_error,
        eligible_history_matches=len(seen),valid_statistic_matches=len(seen)-invalid,
        missing_statistic_matches=invalid,player_date_groups=sum(len(g) for g in by_player.values()),
        zero_history_appearances=int((~mask.any(2)).sum()),
        edge_case_population_counts={n:int(v.sum()) for n,v in conditions.items()},samples=checks,
        source_sha256={name:sha(path) for name,path in paths.items()},raw_sha256=source_hashes,
        audit_script_sha256=sha(__file__),outcome_scores_computed=False,model_fits=0,final_test_raw_opened=False,
        independence_notes=['No production sequence builder/token/technical_pair used.',
            'Known corrected_tour_row metadata policy reused; its historical provenance was audited in prior rounds, not re-sourced here.',
            'Raw identities/outcomes/date/surface independently checked against fixed eligible inventory for all source rows.',
            'Neural learner read-only review: shared normalization, masks, antisymmetric head, fixed shuffle and chronological stopping conform to protocol; model-state replay is a separate audit.'])
    (out/'source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'samples','raw_sha256','source_sha256'}},indent=2))


if __name__=='__main__':main()

"""Provider-recorded past RET state; not an injury/health-status model.

T-21 inclusive, unordered same-date batches, no current-match statistics.
The incomplete official semantic audit is retained as a limitation. All three
candidate groups retain the ordinary-inactivity block; R/RC add four columns.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import timedelta
from math import exp, log1p

import numpy as np
import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.event_state_features import EXCLUDED_PRIOR_IDS
from scripts.model_event_metadata import EVENT_METADATA
from scripts.multilevel_features import _player, _season_date
from scripts.serve_features import _annual_rows, development_entries

LAG = 21
WINDOW = 364
RETURN_GAP = 28
DECAY_DAYS = 90.
DECAY_APPEARANCES = 5.
ORDINARY = ('ordinary_gap', 'ordinary_return_recency', 'ordinary_return_state')
RETIREMENT = ('ret_recency', 'ret_presence', 'ret_return_state')
FEATURE_NAMES = [f'{name}_{suffix}' for name in ORDINARY + RETIREMENT for suffix in ('diff','sum')]
GROUPS = {
    'I': [f'{name}_{suffix}' for name in ORDINARY for suffix in ('diff','sum')],
    'R': [f'{name}_{suffix}' for name in ORDINARY + RETIREMENT[:2] for suffix in ('diff','sum')],
    'RC': [f'{name}_{suffix}' for name in ORDINARY + (RETIREMENT[0],RETIREMENT[2]) for suffix in ('diff','sum')],
}


def status_from_score(score):
    score=str(score).strip().upper()
    status=[]
    if 'RET' in score:status.append('retirement')
    if 'W/O' in score or 'WALKOVER' in score:status.append('walkover')
    if 'DEF' in score:status.append('default')
    return status[0] if len(status)==1 else 'ambiguous' if status else 'completed_or_unknown'


def _target_records(targets):
    if targets.empty or targets.index.has_duplicates or targets.match_id.duplicated().any():
        raise ValueError('Unique nonempty development targets required')
    records=[]
    for row in targets.to_dict('records'):
        season=float(row['season'])
        if not season.is_integer() or not 2009<=season<=2023:
            raise ValueError('Final holdout locked; target seasons must be 2009..2023')
        day=_date(row['tourney_date']);_season_date(day,int(season))
        a,b=_player(row['player_a_id']),_player(row['player_b_id'])
        if a==b:raise ValueError('Target has identical players')
        records.append((day,int(season),a,b))
    return records


def player_state(completed, returns, retirements, first, cutoff):
    """Query sorted unique date groups; no same-date ordering is required."""
    high=bisect_right(completed,cutoff)
    gap=min(365,(cutoff-completed[high-1]).days) if high else 365
    ordinary_anchor=bisect_right(returns,cutoff)-1
    ret_anchor=bisect_right(retirements,cutoff)-1
    def state(dates,position):
        if position<0 or dates[position]<first:return 0.,0.,0.
        anchor=dates[position]
        r=exp(-(cutoff-anchor).days/DECAY_DAYS)
        k=high-bisect_right(completed,anchor)
        return r,1.,r*exp(-k/DECAY_APPEARANCES)
    ir,_,iq=state(returns,ordinary_anchor)
    rr,rp,rq=state(retirements,ret_anchor)
    return (log1p(gap)/log1p(365),ir,iq,rr,rp,rq)


def build_retirement_features(targets,inventory,*,cache_dir=DEFAULT_CACHE,raw_rows=None):
    targets_info=_target_records(targets)
    latest=max(row[0] for row in targets_info)-timedelta(days=LAG)
    complete=defaultdict(set);seen=set()
    seasons=pd.to_numeric(inventory['season'],errors='coerce')
    eligible=inventory.loc[inventory.source.eq('tour') & seasons.isin(range(2006,2024))]
    for row in eligible.to_dict('records'):
        day=_date(row['tourney_date']);_season_date(day,int(row['season']))
        if day>latest:continue
        key=str(row['match_id'])
        if key in seen:raise ValueError('Duplicate completed inventory identity')
        seen.add(key)
        a,b=_player(row['player_a_id']),_player(row['player_b_id'])
        if a==b:raise ValueError('Invalid completed-history identity')
        complete[a].add(day);complete[b].add(day)
    complete={p:sorted(days) for p,days in complete.items()}
    returns={p:[b for a,b in zip(days,days[1:]) if (b-a).days>=RETURN_GAP]
             for p,days in complete.items()}
    entries=development_entries() if raw_rows is None else []
    source=_annual_rows(cache_dir,entries) if raw_rows is None else raw_rows
    rets=defaultdict(set);seen_ret=set();counts=Counter();ret_year=Counter();ret_round=Counter()
    for season,raw in source:
        if not 2006<=int(season)<=2023:continue
        if raw.get('tourney_level') not in ('250','500','M','G'):continue
        row=raw
        # Use the already audited event-date correction without reading ranks
        # or winner/loser metadata that this state does not need yet.
        date_value=(EVENT_METADATA[row['tourney_id']][0]
            if row['tourney_id'] in EVENT_METADATA and row.get('round') not in ('Q1','Q2','Q3','Q4')
            else row['tourney_date'])
        day=_date(date_value);_season_date(day,int(season))
        if day>latest:continue
        status=status_from_score(row.get('score',''))
        if status=='completed_or_unknown':continue
        counts[status]+=1
        if status!='retirement':continue
        key=f"{row['tourney_id']}:{int(row['match_num'])}"
        if key in EXCLUDED_PRIOR_IDS:
            counts['known_excluded_ret']+=1;continue
        if key in seen_ret:raise ValueError('Duplicate RET source identity')
        seen_ret.add(key)
        winner,loser=_player(row['winner_id']),_player(row['loser_id'])
        if winner==loser:raise ValueError('Invalid RET identity')
        rets[loser].add(day);ret_year[int(season)]+=1;ret_round[str(row['round'])]+=1
    rets={p:sorted(days) for p,days in rets.items()}
    result=[];coverage=defaultdict(Counter)
    for day,season,a,b in targets_info:
        first,cutoff=day-timedelta(days=WINDOW),day-timedelta(days=LAG)
        states=[player_state(complete.get(p,[]),returns.get(p,[]),rets.get(p,[]),first,cutoff)
                for p in (a,b)]
        result.append([value for x,y in zip(*states) for value in (x-y,x+y)])
        coverage[season]['targets']+=1
        coverage[season]['either_ret_364']+=int(any(v[4]>0 for v in states))
        coverage[season]['both_ret_364']+=int(all(v[4]>0 for v in states))
        coverage[season]['either_ordinary_return']+=int(any(v[1]>0 for v in states))
    frame=pd.DataFrame(result,index=targets.index,columns=FEATURE_NAMES,dtype=float)
    if not np.isfinite(frame.to_numpy()).all():raise ValueError('Nonfinite state')
    frame.attrs.update(feature_version='provider-ret-state-v1',groups=GROUPS,
        source_files=entries,source_rows_by_status=dict(counts),ret_rows_by_season=dict(ret_year),
        ret_rows_by_round=dict(ret_round),completed_inventory_rows=len(seen),
        latest_history_cutoff=latest.isoformat(),known_excluded_ret_rows=int(counts.get('known_excluded_ret',0)),
        coverage_by_season={str(k):dict(v) for k,v in coverage.items()},
        source_semantics_fully_verified=False,official_verified_sample_rows=10,
        official_unresolved_sample_rows=20,clinical_interpretation=False,
        information_regime='Provider-recorded historical RET and completed-appearance dates, nominal T-21 inclusive',
        root_limited_experiment_authorized=True)
    return frame

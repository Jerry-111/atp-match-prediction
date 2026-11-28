"""Last sixteen eligible player-date groups, with no fabricated intra-event order."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from math import log1p, fsum

import numpy as np
import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _player, _season_date
from scripts.serve_features import _annual_rows, development_entries
from scripts.style_features import technical_pair, RATE_PRIORS

VERSION = 'player-date-sequence-1.0'
TOKENS, WINDOW_DAYS, LAG_DAYS = 16, 1460, 21
SURFACES = ('Hard','Clay','Grass','Carpet')
COLUMNS = ('win_rate','log_matches','log_own_rank','log_opponent_rank',
    'serve_rate','return_rate',*RATE_PRIORS.keys(),'log_serve_points','log_return_points',
    'missing_stats_fraction','surface_hard','surface_clay','surface_grass','surface_carpet',
    'log_gap_days','log_age_days')


@dataclass
class SequenceFeatures:
    values: np.ndarray
    mask: np.ndarray
    dates: np.ndarray
    match_ids: np.ndarray
    indices: np.ndarray
    columns: tuple
    metadata: dict

    def take(self, selected):
        return SequenceFeatures(self.values[selected],self.mask[selected],self.dates[selected],
            self.match_ids[selected],self.indices[selected],self.columns,self.metadata)

    def swapped(self):
        return SequenceFeatures(self.values[:,::-1].copy(),self.mask[:,::-1].copy(),
            self.dates[:,::-1].copy(),self.match_ids.copy(),self.indices.copy(),self.columns,self.metadata)


def validate_targets(targets, inventory):
    if (targets.empty or targets.match_id.duplicated().any() or targets.index.duplicated().any()
            or not pd.to_numeric(targets.season,errors='coerce').isin(range(2009,2024)).all()):
        raise ValueError('Unique development targets 2009..2023 required; final holdout sealed')
    permitted = inventory.loc[inventory.source.eq('tour') &
        pd.to_numeric(inventory.season,errors='coerce').isin(range(2006,2024))].copy()
    if permitted.match_id.duplicated().any():
        raise ValueError('Duplicate historical source ID')
    lookup = permitted.set_index('match_id').to_dict('index')
    queries=[]
    for row in targets.to_dict('records'):
        day,year=_date(row['tourney_date']),int(row['season']);_season_date(day,year)
        a,b=_player(row['player_a_id']),_player(row['player_b_id'])
        key=str(row['match_id']);key=key if key.startswith('tour:') else 'tour:'+key
        if key not in lookup or a==b or row['surface'] not in SURFACES:
            raise ValueError('Target missing from fixed inventory or invalid identity')
        meta=lookup[key]
        if (int(meta['season'])!=year or _date(meta['tourney_date'])!=day
                or str(meta['tourney_id'])!=str(row['tourney_id']) or meta['surface']!=row['surface']
                or {a,b}!={_player(meta['player_a_id']),_player(meta['player_b_id'])}):
            raise ValueError('Target metadata differs from published inventory')
        queries.append((day,a,b))
    return permitted,queries


def _rank(value):
    rank=float(value)
    if not np.isfinite(rank) or rank<=0:
        raise ValueError('Positive finite historical rank required')
    return rank


def _token(records):
    """Aggregate every row in one player/date group in canonical source order."""
    records=sorted(records,key=lambda r:r[0])
    n=len(records)
    wins=fsum(r[1] for r in records)
    own=fsum(np.log(r[2]) for r in records)/n
    opponent=fsum(np.log(r[3]) for r in records)/n
    counts=np.sum(np.array([r[5] for r in records],dtype=np.float64),axis=0)
    valid=sum(r[6] for r in records)
    serve_won,serve_n=counts[6]+counts[8],counts[1]
    return_won,return_n=counts[10]+counts[12],counts[11]+counts[13]
    rates=[(counts[2*i]+100*p)/(counts[2*i+1]+100) for i,p in enumerate(RATE_PRIORS.values())]
    surfaces=[sum(r[4]==surface for r in records)/n for surface in SURFACES]
    return np.array([(wins+1)/(n+2),log1p(n),own,opponent,
        (serve_won+65)/(serve_n+100),(return_won+35)/(return_n+100),
        *rates,log1p(serve_n),log1p(return_n),(n-valid)/n,*surfaces],dtype=np.float64)


def build_sequence_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Return same-order (N,2,16,22) histories and explicit masks/dates.

    Tied matches are aggregated by player/date, never by source row order.
    Only metadata of the target itself is inspected; its outcome is irrelevant.
    """
    allowed,queries=validate_targets(targets,inventory)
    latest=max(day for day,_,_ in queries)-timedelta(days=LAG_DAYS)
    earliest=min(day for day,_,_ in queries)-timedelta(days=LAG_DAYS+WINDOW_DAYS)
    expected={}
    for row in allowed.to_dict('records'):
        day,year=_date(row['tourney_date']),int(row['season']);_season_date(day,year)
        if earliest<=day<=latest:
            expected[str(row['match_id'])]=row
    entries=[e for e in development_entries() if earliest.year<=int(e['season'])<=latest.year+1] if raw_rows is None else []
    entries=[e for e in entries if int(e['season'])<=2023]
    source=_annual_rows(cache_dir,entries) if raw_rows is None else raw_rows
    seen=set();grouped=defaultdict(list);valid_stats=0;missing_stats=0
    for season,raw in source:
        if not 2006<=int(season)<=2023:
            continue
        key=f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in seen:
            raise ValueError('Duplicate eligible raw history')
        seen.add(key);raw=corrected_tour_row(raw);row=expected[key]
        day=_date(row['tourney_date']);_season_date(day,int(season))
        a,b=_player(row['player_a_id']),_player(row['player_b_id'])
        winner,loser=_player(raw['winner_id']),_player(raw['loser_id'])
        outcome=float(row['a_won'])
        if ({a,b}!={winner,loser} or a==b or outcome not in (0.,1.)
                or float(winner==a)!=outcome or _date(raw['tourney_date'])!=day
                or int(row['season'])!=int(season) or raw['surface']!=row['surface']):
            raise ValueError('Raw historical identity/date/outcome mismatch')
        ranks={a:_rank(row['rank_a']),b:_rank(row['rank_b'])}
        paired=technical_pair(raw)
        counts={winner:paired[0],loser:paired[1]} if paired is not None else {a:(0.,)*14,b:(0.,)*14}
        valid=int(paired is not None);valid_stats+=valid;missing_stats+=1-valid
        for player,opponent in ((a,b),(b,a)):
            grouped[player,day].append((key,float(player==winner),ranks[player],ranks[opponent],
                str(row['surface']),counts[player],valid))
    if set(expected)-seen:
        raise ValueError(f'Pinned raw rows missing {len(set(expected)-seen)} eligible histories')
    player_days=defaultdict(list);player_values=defaultdict(list)
    for (player,day),records in sorted(grouped.items(),key=lambda item:(item[0][0],item[0][1])):
        player_days[player].append(day);player_values[player].append(_token(records))
    values=np.zeros((len(targets),2,TOKENS,len(COLUMNS)),dtype=np.float32)
    mask=np.zeros((len(targets),2,TOKENS),dtype=bool)
    dates=np.full((len(targets),2,TOKENS),-1,dtype=np.int32)
    for i,(day,a,b) in enumerate(queries):
        cutoff=day-timedelta(days=LAG_DAYS);first=cutoff-timedelta(days=WINDOW_DAYS)
        for side,player in enumerate((a,b)):
            days=player_days[player]
            lo,hi=bisect_left(days,first),bisect_right(days,cutoff)
            lo=max(lo,hi-TOKENS);chosen=days[lo:hi];n=len(chosen)
            if not n:continue
            core=np.asarray(player_values[player][lo:hi])
            ordinals=np.array([d.toordinal() for d in chosen],dtype=np.int32)
            gaps=np.r_[0,np.diff(ordinals)];ages=cutoff.toordinal()-ordinals
            values[i,side,-n:]=np.column_stack([core,np.log1p(gaps),np.log1p(ages)])
            mask[i,side,-n:]=True;dates[i,side,-n:]=ordinals
    if values.shape[-1]!=22 or not np.isfinite(values).all():
        raise ValueError('Expected 22 finite sequence channels')
    metadata=dict(version=VERSION,source_files=entries,injected_raw_rows=raw_rows is not None,
        source_years=sorted({int(e['season']) for e in entries}),historical_matches=len(expected),
        valid_statistic_matches=valid_stats,missing_statistic_matches=missing_stats,
        player_date_groups=len(grouped),max_tokens=TOKENS,window_days=WINDOW_DAYS,lag_days=LAG_DAYS,
        grouped_tied_dates=True,zero_history_appearances=int((~mask.any(axis=2)).sum()),
        median_history_length=float(np.median(mask.sum(axis=2))),
        earliest_window_bound=str(earliest),latest_window_bound=str(latest),
        earliest_loaded_history=str(min((_date(r['tourney_date']) for r in expected.values()),default=earliest)),
        latest_loaded_history=str(max((_date(r['tourney_date']) for r in expected.values()),default=latest)),
        final_test_raw_opened=False,target_outcome_used=False)
    return SequenceFeatures(values,mask,dates,targets.match_id.to_numpy(dtype=str),
        targets.index.to_numpy(),COLUMNS,metadata)


def save_sequences(path, sequences):
    np.savez_compressed(path,values=sequences.values,mask=sequences.mask,dates=sequences.dates,
        match_ids=sequences.match_ids.astype(str),indices=sequences.indices,columns=np.array(sequences.columns))


def load_sequences(path, *, metadata=None):
    with np.load(path,allow_pickle=False) as data:
        return SequenceFeatures(data['values'],data['mask'],data['dates'],data['match_ids'],
            data['indices'],tuple(data['columns']),metadata or {})

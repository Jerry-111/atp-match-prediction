"""Conservative, outcome-blind linkage of a separately labelled market branch.

Only 2021--2023 public Valuebetennis files are permitted. No result, score, or
duration column is read. Opening and closing prices remain separate; no odds
availability filter removes a target. This is NOT the original T-21 information
regime: the provider describes these prices as opening and last pre-match,
but publishes no per-quote timestamps.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from math import isfinite, log
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/market-manifest.json'
DEFAULT_CACHE=ROOT/'data/raw/market'
ALLOWED_YEARS=(2021,2022,2023)
MARKET_COLUMNS=('match_id','date','tournoi','tournoi_id','categorie','genre','surface','tour',
                'joueur1','joueur1_id','joueur2','joueur2_id',
                'cote1_ouverture','cote2_ouverture','cote1_cloture','cote2_cloture')
SURFACES={'dur':'Hard','dur intérieur':'Hard','terre battue':'Clay','gazon':'Grass'}
CATEGORIES={'Main tour','Masters','Grand Chelem'}
MIN_EVENT_ANCHORS=3
EARLY_ROUNDS={'R128','R64','R32','R16'}
LATE_ROUNDS={'QF':9,'SF':10,'F':12}

# Manually reviewed raw-name crosswalks: unique full-name expansions, omitted
# middle-name token, or exact token-order reversal. No fuzzy fallback. Each
# crosswalk also pins the source player ID and both raw names for review.
# Other shortened/spelling variants remain unresolved rather than guessed.
REVIEWED_NAMES={
    'SM37':('Diego Schwartzman','Diego Sebastian Schwartzman',1292),
    'KB95':('Jason Kubler','Jason Murray Kubler',2412),
    'MU94':('Jaume Munar','Jaume Antoni Munar Clar',10698),
    'MO44':('Pedro Martinez','Pedro Martinez Portero',12525),
    'FB98':('Taylor Fritz','Taylor Harry Fritz',9023),
    'KD35':('Thai-Son Kwiatkowski','Thai Kwiatkowski',2792),
    'MJ08':('Thiago Monteiro','Thiago Moura Monteiro',1174),
    'Y09V':('Yunchaokete Bu','Bu Yunchaokete',25003),
}


def normalize_name(value):
    text=unicodedata.normalize('NFKD',str(value)).encode('ascii','ignore').decode().lower()
    return re.sub('[^a-z0-9]','',text)


def de_vig(odds_a,odds_b):
    """Return (odd logit, even overround, availability), without clipping odds.

    Normalized implied p(A)=(1/oa)/(1/oa+1/ob); logit=log(ob)-log(oa).
    Both prices must be finite and strictly greater than1. Missing is neutral
    logit0 with availability0, so downstream full-cohort fallback is explicit.
    """
    try:a,b=float(odds_a),float(odds_b)
    except (TypeError,ValueError):return 0.,0.,0.
    if not isfinite(a) or not isfinite(b) or a<=1 or b<=1:return 0.,0.,0.
    return log(b)-log(a),1/a+1/b-1,1.


def load_market_rows(cache_dir=DEFAULT_CACHE,seasons=ALLOWED_YEARS):
    seasons=tuple(sorted(set(map(int,seasons))))
    if not set(seasons).issubset(ALLOWED_YEARS):
        raise ValueError('Market final holdout is locked; only2021--2023 downloads allowed')
    manifest=json.loads(MANIFEST.read_text())
    entries={int(e['season']):e for e in manifest['files']}
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    frames=[]
    for year in seasons:
        entry=entries[year];path=cache/f'{year}.csv'
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=cache,delete=False) as handle:temp=Path(handle.name)
            try:
                # urllib's default UA is rejected by this public endpoint;
                # ordinary curl with a browser UA succeeds without credentials.
                subprocess.run(['curl','-sS','-L','--fail','--max-time','60','-A','Mozilla/5.0',
                                '-o',str(temp),entry['url']],check=True)
                if hashlib.sha256(temp.read_bytes()).hexdigest()!=entry['sha256']:
                    raise ValueError(f'Market source hash changed: {year}')
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Market cache hash mismatch: {year}')
        # Explicit usecols is the outcome barrier: labels/scores never loaded.
        data=pd.read_csv(path,sep=';',usecols=list(MARKET_COLUMNS))
        if len(data)!=entry['rows']:raise ValueError('Market source row count changed')
        frames.append(data.assign(source_season=year))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=[*MARKET_COLUMNS,'source_season'])


def _prepare_source(source):
    # Filter the permitted annual source before accessing names or odds.
    if 'source_season' not in source:raise ValueError('Market source_season is required')
    source=source.loc[pd.to_numeric(source.source_season).isin(ALLOWED_YEARS)].copy()
    source=source.loc[source.genre.eq('atp')&source.categorie.isin(CATEGORIES)].copy()
    source['tour']=pd.to_numeric(source.tour,errors='coerce')
    source=source.loc[source.tour.ge(4)].copy()
    if source.match_id.duplicated().any():raise ValueError('Duplicate market match IDs')
    source['utc']=pd.to_datetime(source.date,utc=True,errors='coerce')
    calendar_year=source.utc.dt.year
    declared=pd.to_numeric(source.source_season)
    valid_year=(calendar_year.eq(declared)|
                (calendar_year.eq(declared-1)&source.utc.dt.month.eq(12)&source.utc.dt.day.ge(25)))
    source=source.loc[valid_year&calendar_year.le(2023)].copy()
    source['mapped_surface']=source.surface.map(SURFACES)
    source=source.loc[source.utc.notna()&source.mapped_surface.notna()]
    names=defaultdict(set)
    for side in (1,2):
        for name,identifier in zip(source[f'joueur{side}'],source[f'joueur{side}_id']):
            names[normalize_name(name)].add(str(int(identifier)))
    return source,names


def build_market_features(targets,*,cache_dir=DEFAULT_CACHE,market_rows=None):
    """Return (8 numeric columns, matched metadata), both at exact target index.

    Accepts2009--2023 targets; pre2021 rows remain present with missing flags.
    Linkage needs a cohort with at least3 unique pair/date/surface anchors per
    event; tiny isolated target subsets may intentionally remain unmatched.
    Names/players/date/surface/event IDs/round only determine linkage. Never
    access a_won or market result columns. No benchmark score is computed.
    """
    if targets.empty or not pd.to_numeric(targets.season).between(2009,2023).all():
        raise ValueError('Development targets must be2009--2023; final holdout locked')
    dates=pd.to_datetime(targets.tourney_date,errors='coerce')
    seasons=pd.to_numeric(targets.season)
    valid_year=(dates.dt.year.eq(seasons)|
                (dates.dt.year.eq(seasons-1)&dates.dt.month.eq(12)&dates.dt.day.ge(25)))
    if dates.isna().any() or dates.dt.year.gt(2023).any() or not valid_year.all():
        raise ValueError('Target date/season mismatch or final holdout date')
    if targets.match_id.duplicated().any():raise ValueError('Duplicate target match IDs')
    needed=sorted(set(int(y) for y in targets.season if int(y) in ALLOWED_YEARS))
    source=load_market_rows(cache_dir,needed) if market_rows is None else market_rows
    source,names=_prepare_source(source)
    target_records=targets.to_dict('records')
    target_name_ids=defaultdict(set)
    for row in target_records:
        for side in ('a','b'):
            target_name_ids[normalize_name(row[f'player_{side}_name'])].add(str(row[f'player_{side}_id']))
    def raw_identity(identifier,name):
        identifier=str(identifier);norm=normalize_name(name)
        if len(target_name_ids[norm])!=1:return None,'ambiguous_target_name'
        if identifier in REVIEWED_NAMES:
            original,other,source_id=REVIEWED_NAMES[identifier]
            if norm!=normalize_name(original):return None,'crosswalk_target_name_changed'
            candidates=names.get(normalize_name(other),set())
            return (str(source_id),'reviewed_name_crosswalk') if candidates=={str(source_id)} else (None,'crosswalk_not_unique')
        candidates=names.get(norm,set())
        return (next(iter(candidates)),'normalized_full_name') if len(candidates)==1 else (None,'name_unmatched_or_ambiguous')
    resolved={}
    source_to_target=defaultdict(set);target_to_source=defaultdict(set)
    for row in target_records:
        if int(row['season']) not in ALLOWED_YEARS:continue
        for side in ('a','b'):
            identifier=str(row[f'player_{side}_id']);name=row[f'player_{side}_name']
            result=raw_identity(identifier,name)
            resolved[identifier,normalize_name(name)]=result
            if result[0] is not None:
                source_to_target[result[0]].add(identifier)
                target_to_source[identifier].add(result[0])
    def identity(identifier,name):
        identifier=str(identifier)
        source_id,rule=resolved[identifier,normalize_name(name)]
        if source_id is not None and (len(source_to_target[source_id])!=1 or len(target_to_source[identifier])!=1):
            return None,'player_mapping_not_bijective'
        return source_id,rule
    pairs=defaultdict(list)
    for row in source.to_dict('records'):
        row['id1'],row['id2']=str(int(row['joueur1_id'])),str(int(row['joueur2_id']))
        if row['id1']==row['id2']:continue
        row['source_id']=str(int(row['match_id']))
        row['event_id']=str(int(row['tournoi_id']))
        row['source_day']=row['utc'].tz_localize(None).normalize()
        key=(int(row['source_season']),tuple(sorted((row['id1'],row['id2']))))
        pairs[key].append(row)
    candidate_lists=[];identities=[];reasons=[]
    anchor_ids=defaultdict(lambda:defaultdict(set))
    for row in target_records:
        if int(row['season']) not in ALLOWED_YEARS:
            candidate_lists.append([]);identities.append((None,None,'outside_source_years'));reasons.append('outside_source_years');continue
        a,arule=identity(row['player_a_id'],row['player_a_name'])
        b,brule=identity(row['player_b_id'],row['player_b_name'])
        identities.append((a,b,arule+'|'+brule))
        if a is None or b is None or a==b:
            candidate_lists.append([]);reasons.append('name_unmatched_or_ambiguous');continue
        start=pd.Timestamp(row['tourney_date']).normalize()
        candidates=[]
        for m in pairs[(int(row['season']),tuple(sorted((a,b))))]:
            days=(m['source_day']-start).days
            round_ok=(row['round'] in EARLY_ROUNDS and 4<=m['tour']<=7) or LATE_ROUNDS.get(row['round'])==m['tour']
            if -1<=days<=15 and m['mapped_surface']==row['surface'] and round_ok:
                candidates.append(m)
        candidate_lists.append(candidates);reasons.append('no_pair_date_surface_round_candidate')
        if len(candidates)==1:
            anchor_ids[str(row['tourney_id'])][candidates[0]['event_id']].add(candidates[0]['source_id'])
    anchors=defaultdict(Counter,{event:Counter({source_event:len(ids) for source_event,ids in groups.items()})
                                for event,groups in anchor_ids.items()})
    event_map={}
    for event,counts in anchors.items():
        # Unanimous independent anchors; no majority-based guessing.
        if len(counts)==1 and sum(counts.values())>=MIN_EVENT_ANCHORS:
            event_map[event]=next(iter(counts))
    metadata=[]
    for row,candidates,identity_info,reason in zip(target_records,candidate_lists,identities,reasons):
        event=str(row['tourney_id']);a,b,rule=identity_info
        item=dict(match_id=row['match_id'],season=int(row['season']),status=reason,
            target_event_id=event,market_match_id=None,market_event_id=None,market_match_utc=None,
            market_player_a_id=a,market_player_b_id=b,name_rule=rule,candidate_count=len(candidates),
            event_anchor_count=int(sum(anchors[event].values())),event_anchor_unique_events=len(anchors[event]),
            days_after_target_start=None,source_name_a=None,source_name_b=None,
            opening_odds_a=None,opening_odds_b=None,closing_odds_a=None,closing_odds_b=None)
        if candidates:
            if event not in event_map:item['status']='event_anchor_ambiguous_or_insufficient'
            else:
                accepted=[m for m in candidates if m['event_id']==event_map[event]]
                item['status']='within_event_ambiguous' if len(accepted)>1 else 'no_candidate_in_anchored_event'
                if len(accepted)==1:
                    m=accepted[0];side=1 if a==m['id1'] else 2;other=3-side
                    item.update(status='matched',market_match_id=m['source_id'],market_event_id=m['event_id'],
                        market_match_utc=m['utc'].isoformat(),source_name_a=m[f'joueur{side}'],source_name_b=m[f'joueur{other}'],
                        days_after_target_start=int((m['source_day']-pd.Timestamp(row['tourney_date']).normalize()).days),
                        opening_odds_a=m[f'cote{side}_ouverture'],opening_odds_b=m[f'cote{other}_ouverture'],
                        closing_odds_a=m[f'cote{side}_cloture'],closing_odds_b=m[f'cote{other}_cloture'])
        metadata.append(item)
    usage=Counter(m['market_match_id'] for m in metadata if m['status']=='matched')
    output=[]
    for item in metadata:
        if item['status']=='matched' and usage[item['market_match_id']]>1:item['status']='source_match_reused_ambiguous'
        usable=item['status']=='matched'
        opened=de_vig(item['opening_odds_a'],item['opening_odds_b']) if usable else (0.,0.,0.)
        closed=de_vig(item['closing_odds_a'],item['closing_odds_b']) if usable else (0.,0.,0.)
        item['opening_available'],item['closing_available']=bool(opened[2]),bool(closed[2])
        output.append(dict(market_open_logit_diff=opened[0],market_open_overround=opened[1],market_open_available=opened[2],
            market_close_logit_diff=closed[0],market_close_overround=closed[1],market_close_available=closed[2],
            market_movement_logit_diff=(closed[0]-opened[0]) if opened[2]*closed[2] else 0.,
            market_both_available=opened[2]*closed[2]))
    features=pd.DataFrame(output,index=targets.index);matched=pd.DataFrame(metadata,index=targets.index)
    if not np.isfinite(features.to_numpy()).all():raise ValueError('Nonfinite market features')
    coverage={str(int(year)):dict(targets=int((matched.season==year).sum()),
        matched=int(((matched.season==year)&matched.status.eq('matched')).sum()),
        opening_available=int(matched.loc[matched.season==year,'opening_available'].sum()),
        closing_available=int(matched.loc[matched.season==year,'closing_available'].sum())) for year in sorted(set(matched.season))}
    features.attrs.update(source='Valuebetennis',source_url='https://www.valuebetennis.com/en/donnees.htm',
        license='CC BY 4.0',quote_timestamps_present=False,
        information_regime='Provider-described opening and last prematch prices, not tournament-start T-21 history',
        coverage_by_season=coverage,match_status_counts=dict(Counter(matched.status)),
        linkage='Bijective unique reviewed/full names; consistent calendar/source season; same players, surface, round family, UTC day -1..15; >=3 distinct unanimous source-match event anchors; one-to-one source match IDs',
        outcomes_read_for_linkage=False,final_test_evaluated=False,
        missing_policy='All target rows retained; zero logit/overround plus availability0. Downstream probability must fall back explicitly to frozen historical model.')
    return features,matched

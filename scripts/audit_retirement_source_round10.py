"""Frozen pre-2023 RET semantics sample; no model fitting or target outcomes.

Official draw PDFs are public evidence cached outside the repository. Extraction
is not adjudication: review records must explicitly identify the bracket path,
status and source hash, or the sample remains unverified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from scripts.advanced_features import _date
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _season_date
from scripts.serve_features import development_entries
from scripts.train_models import ROOT, sha256

OUT = ROOT/'results/round10/retirement'
TOKEN = 'round10-retirement-semantics-v1'
STAT_COLUMNS = ['w_svpt','l_svpt','w_1stWon','w_2ndWon','l_1stWon','l_2ndWon']
RAW_COLUMNS = ['tourney_id','match_num','tourney_date','tourney_name','tourney_level',
    'surface','round','winner_id','winner_name','loser_id','loser_name','score']+STAT_COLUMNS

def tie(text):
    return hashlib.sha256((TOKEN+'|'+text).encode()).hexdigest()


def load_candidates():
    entries=[e for e in development_entries() if 2006<=int(e['season'])<=2022]
    assert [int(e['season']) for e in entries]==list(range(2006,2023))
    records=[];levels=Counter()
    for e in entries:
        path=ROOT/'data/raw'/f"{e['season']}.csv"
        if not path.exists() or sha256(path)!=e['sha256']:
            raise ValueError('Existing pinned source required: '+str(path))
        frame=pd.read_csv(path,usecols=RAW_COLUMNS,dtype=str).fillna('')
        for raw in frame[frame.tourney_level.isin(['250','500','M','G'])].to_dict('records'):
            score=raw['score'].strip().upper()
            flags=[s for s,marker in [('retirement','RET'),('walkover','W/O'),('default','DEF')] if marker in score]
            if 'WALKOVER' in score and 'walkover' not in flags:flags.append('walkover')
            if not flags:continue
            row=corrected_tour_row(raw);day=_date(row['tourney_date']);_season_date(day,int(e['season']))
            year=int(e['season']);status=flags[0] if len(flags)==1 else 'ambiguous_markers'
            missing=any(not row[c].strip() for c in STAT_COLUMNS)
            era='2006-2011' if year<=2011 else '2012-2017' if year<=2017 else '2018-2022'
            stage='early' if row['round'] in ['R128','R64','R32'] else 'middle' if row['round'] in ['R16','QF'] else 'late' if row['round'] in ['SF','F'] else 'other'
            records.append(dict(match_id=f"{row['tourney_id']}:{int(row['match_num'])}",season=year,
                tourney_id=row['tourney_id'],tourney_name=row['tourney_name'],tourney_level=row['tourney_level'],
                date=day.isoformat(),surface=row['surface'],round=row['round'],winner_id=row['winner_id'],
                winner_name=row['winner_name'],loser_id=row['loser_id'],loser_name=row['loser_name'],
                score=score,status=status,missing_technical_fields=missing,era=era,round_group=stage))
            levels[status]+=1
    frame=pd.DataFrame(records)
    assert not frame.match_id.duplicated().any()
    if not frame.winner_id.ne('').all() or not frame.loser_id.ne('').all() or not frame.winner_id.ne(frame.loser_id).all():
        raise ValueError('Invalid source identity')
    return frame,entries


def choose_sample(frame):
    ret=frame[frame.status.eq('retirement')].copy()
    ret['tie']=ret.match_id.map(tie)
    chosen=ret[ret.missing_technical_fields].sort_values('tie').match_id.tolist()
    assert len(chosen)==3, 'Frozen missing-statistics census changed'
    carpet=ret[ret.surface.eq('Carpet') & ~ret.match_id.isin(chosen)].sort_values('tie')
    assert len(carpet)>0
    chosen.append(carpet.iloc[0].match_id)
    strata=[(era,surface,stage) for era in ['2006-2011','2012-2017','2018-2022']
        for surface in ['Hard','Clay','Grass'] for stage in ['early','middle','late']]
    while len(chosen)<30:
        seen=ret[ret.match_id.isin(chosen)].groupby(['era','surface','round_group']).size().to_dict()
        ordered=sorted(strata,key=lambda s:(seen.get(s,0),tie('|'.join(s))))
        for era,surface,stage in ordered:
            available=ret[ret.era.eq(era)&ret.surface.eq(surface)&ret.round_group.eq(stage)&~ret.match_id.isin(chosen)]
            if len(available):
                chosen.append(available.sort_values('tie').iloc[0].match_id);break
        else:raise ValueError('Insufficient deterministic strata')
    sample=ret.set_index('match_id').loc[chosen].reset_index().drop(columns='tie')
    assert len(sample)==30 and not sample.match_id.duplicated().any()
    return sample


def registry_for(sample,entries):
    return dict(sample_id=TOKEN,source_years=list(range(2006,2023)),source_files=entries,
        sampling_rule='All 3 missing-technical RET rows, one hash-selected Carpet row, then least-filled era/surface/round-group cells with fixed SHA256 tie breaks to 30. Not a prevalence sample.',
        sampling_fields=['match_id','era','surface','round_group','missing_technical_fields'],
        target_outcomes_or_model_predictions_used=False,sample=sample.to_dict('records'))


def coverage(frame):
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv',usecols=['match_id','season','tourney_date','player_a_id','player_b_id'])
    rows=rows[rows.season.le(2022)].copy();rows['date']=pd.to_datetime(rows.tourney_date)
    by={p:pd.to_datetime(g.date).to_numpy(dtype='datetime64[D]') for p,g in frame[frame.status.eq('retirement')].groupby('loser_id')}
    result={}
    for year,targets in rows.groupby('season'):
        totals={90:0,364:0}
        for r in targets.itertuples():
            cutoff=np.datetime64(r.date-pd.Timedelta(days=21),'D')
            for window in totals:
                lower=np.datetime64(r.date-pd.Timedelta(days=window),'D')
                totals[window]+=int(any(np.any((by.get(p,np.array([],dtype='datetime64[D]'))>=lower)&(by.get(p,np.array([],dtype='datetime64[D]'))<=cutoff)) for p in (r.player_a_id,r.player_b_id)))
        result[str(year)]=dict(targets=len(targets),either_ret_loser_90=totals[90],either_ret_loser_364=totals[364])
    return result


def fetch_draws(sample,cache,previous=None):
    cache.mkdir(parents=True,exist_ok=True)
    records=dict(previous or {});paused_hosts=set();requests={}
    for event in sorted(sample.tourney_id.unique()):
        year,code=event.split('-',1)
        assert 2006<=int(year)<=2022 and code.isdigit()
        requests[event]=f'https://www.protennislive.com/posting/{year}/{code}/mds.pdf'
    for key,source in records.items():
        if key not in requests and source.get('sha256'):requests[key]=source['url']
    for event,url in requests.items():
        path=cache/f'{event}-mds.pdf';text_path=path.with_suffix('.txt')
        if event in records and records[event].get('sha256'):
            old_path=Path(records[event]['cache_path'])
            if old_path.exists():path=old_path;text_path=path.with_suffix('.txt')
        info=dict(url=url,cache_path=str(path))
        # An unavailable URL is evidence, not a reason to repeatedly retry it.
        if event in records and not records[event].get('sha256') and not path.exists():
            continue
        host=urlsplit(url).hostname
        if host in paused_hosts and not path.exists():
            records[event]=dict(info,status='deferred_after_rate_limit')
            continue
        try:
            if not path.exists():
                temporary=path.with_suffix('.download')
                run=subprocess.run(['curl','-L','--fail','--silent','--show-error','--max-time','30',url,'-o',str(temporary)],capture_output=True,text=True)
                if run.returncode:
                    temporary.unlink(missing_ok=True);raise RuntimeError(run.stderr.strip())
                if not temporary.read_bytes().startswith(b'%PDF'):
                    temporary.unlink(missing_ok=True);raise ValueError('Response is not PDF')
                temporary.replace(path)
            info['sha256']=sha256(path)
            subprocess.run(['pdftotext','-layout',str(path),str(text_path)],check=True,capture_output=True)
            info.update(status='fetched_not_adjudicated',text_path=str(text_path))
        except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:
            info.update(status='unavailable',error=str(e))
            if '429' in str(e):paused_hosts.add(host)
        records[event]=info
    return records


def review_checks(sample,evidence,reviews):
    """Validate human adjudication against the frozen sample and cached source."""
    if not set(reviews).issubset(set(sample.match_id)):
        raise ValueError('Review contains a match outside the frozen sample')
    checks=[]
    for row in sample.to_dict('records'):
        review=reviews.get(row['match_id'],dict(status='unverified'))
        source=evidence.get(review.get('source_key',row['tourney_id']),{})
        if review['status'] not in ['verified','contradicted','unverified']:
            raise ValueError('Unknown evidence status')
        if review['status']=='verified':
            if review['source_sha256']!=source.get('sha256'):
                raise ValueError('Review/source hash mismatch')
            if (review['winner_id'],review['loser_id'])!=(row['winner_id'],row['loser_id']):
                raise ValueError('Review identity direction mismatch')
            if review['official_status']!='retirement' or not review.get('bracket_or_report_evidence'):
                raise ValueError('Explicit official status evidence required')
            if not review.get('visual_pages'):
                raise ValueError('PDF layout must be reviewed, not only text names')
            source_path=Path(source['cache_path'])
            if not source_path.exists() or sha256(source_path)!=source['sha256']:
                raise ValueError('Reviewed official source cache missing or changed')
        checks.append(dict(**row,official_source=source,review=review))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-official',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path(tempfile.gettempdir())/'atp-round10-retirement')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    frame,entries=load_candidates();sample=choose_sample(frame);registry=registry_for(sample,entries)
    path=OUT/'sample_registry.json'
    if path.exists():assert json.loads(path.read_text())==registry,'Frozen sample changed'
    else:path.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n')
    evidence_path=OUT/'official_sources.json'
    if args.fetch_official:
        old=json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        evidence=fetch_draws(sample,args.cache_dir,old)
        if evidence_path.exists():
            old=json.loads(evidence_path.read_text())
            for key in set(old)&set(evidence):
                if old[key].get('sha256') and evidence[key].get('sha256'):
                    assert old[key]['sha256']==evidence[key]['sha256'],'Official source content changed'
        evidence_path.write_text(json.dumps(evidence,indent=2)+'\n')
    else:evidence=json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    review_path=OUT/'official_reviews.json';reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
    checks=review_checks(sample,evidence,reviews)
    review_counts=dict(Counter(c['review']['status'] for c in checks))
    result=dict(status='semantic_audit_complete' if sum(review_counts.values())==30 and review_counts.get('unverified',0)==0 else 'semantic_audit_incomplete',
        source_years_read=list(range(2006,2023)),source_rows_by_status=frame.status.value_counts().to_dict(),
        unique_ret_losers=int(frame[frame.status.eq('retirement')].loser_id.nunique()),sample_rows=30,
        ret_census_by_surface=frame[frame.status.eq('retirement')].surface.value_counts().to_dict(),
        ret_census_by_round=frame[frame.status.eq('retirement')]['round'].value_counts().to_dict(),
        non_ret_statuses_not_pooled=True,
        observed_official_contradictions=review_counts.get('contradicted',0),
        source_semantics_fully_verified=review_counts.get('verified',0)==30,
        model_fitting_ready=False,
        root_limited_experiment_authorized=True,
        authorization_scope='Three nested inactivity/RET experiments accepted with 10/30 official checks and 20 unresolved; this is not full source verification.',
        decision='Partial source evidence retained; coordinator authorized a bounded provider-recorded-status experiment, not a validated injury model.',
        sample_distribution={field:sample[field].value_counts().to_dict() for field in ['era','surface','round','missing_technical_fields']},
        review_counts=review_counts,sample_registry_sha256=sha256(path),
        official_reviews_sha256=sha256(review_path) if review_path.exists() else None,
        official_sources_sha256=sha256(evidence_path) if evidence_path.exists() else None,
        official_verified_by_era=dict(Counter(c['era'] for c in checks if c['review']['status']=='verified')),
        official_verified_by_surface=dict(Counter(c['surface'] for c in checks if c['review']['status']=='verified')),
        coverage_pre2023=coverage(frame),coverage_2023='Not measured: this source audit does not read 2023 annual raw data.',
        sampled_checks=checks,limitations=['RET is a source noncompletion status, not a clinical diagnosis.',
        'W/O and DEF are separate; no status is silently relabelled and no original target is changed.',
        'This purposive coverage sample cannot estimate a population error rate.',
        'An unavailable official source stays unverified; source-to-source outcome agreement is insufficient.',
        'Official name/bracket adjudication supports the existing IDs through the pinned raw row; official PDFs do not supply these archive IDs.',
        'Official confirmation of RET need not confirm a numerical score; 2007 Roland Garros only states Ret.',
        'RR remains a legitimate source status context; no knockout path or within-event chronology is inferred.',
        'Coverage is a status/identity census under the existing nominal T-21 policy, not evidence of predictive value.'],
        fitting_performed=False,target_outcomes_read=False,final_year_raw_read=False,
        audit_code_sha256=sha256(Path(__file__)))
    (OUT/'source_feasibility.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sampled_checks','coverage_pre2023']},indent=2,ensure_ascii=False))
    print(sample[['match_id','era','surface','round','winner_name','loser_name','score','missing_technical_fields']].to_string(index=False),flush=True)

if __name__=='__main__':main()

"""Source-evidence checks; these tests do not fit or score a model."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.audit_retirement_source_round10 import choose_sample, fetch_draws, review_checks


def test_sample_is_row_order_invariant_and_keeps_missing_and_rr():
    rows=[]
    for year in ('2006-2011','2012-2017','2018-2022'):
        for surface in ('Hard','Clay','Grass'):
            for stage in ('early','middle','late'):
                for k in range(2):
                    rows.append(dict(match_id=f'{year}-{surface}-{stage}-{k}',
                        era=year,surface=surface,round_group=stage,
                        missing_technical_fields=False,status='retirement'))
    for k in range(3):
        rows[k]['missing_technical_fields']=True
    rows[0]['round_group']='other'  # RR must not disappear from missingness stratum.
    rows.append(dict(match_id='carpet',era='2006-2011',surface='Carpet',round_group='middle',
                     missing_technical_fields=False,status='retirement'))
    frame=pd.DataFrame(rows)
    a=choose_sample(frame);b=choose_sample(frame.sample(frac=1,random_state=7))
    assert a.match_id.tolist()==b.match_id.tolist()
    assert len(a)==30
    assert set(frame.loc[frame.missing_technical_fields,'match_id'])<=set(a.match_id)
    assert 'carpet' in set(a.match_id)


def evidence_fixture(tmp_path):
    path=tmp_path/'draw.pdf';path.write_bytes(b'%PDF synthetic hash fixture')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sample=pd.DataFrame([dict(match_id='2007-433:27',tourney_id='2007-433',
                              winner_id='winner',loser_id='retiring')])
    source={'2007-433':dict(cache_path=str(path),sha256=digest)}
    review={'2007-433:27':dict(status='verified',source_sha256=digest,
        winner_id='winner',loser_id='retiring',official_status='retirement',
        bracket_or_report_evidence='Explicit bracket path and retirement marker',visual_pages=[1])}
    return sample,source,review,path


def test_verified_review_requires_matching_direction_and_visual_evidence(tmp_path):
    sample,source,review,_=evidence_fixture(tmp_path)
    assert review_checks(sample,source,review)[0]['review']['status']=='verified'
    review['2007-433:27']['loser_id']='winner'
    with pytest.raises(ValueError,match='direction'):
        review_checks(sample,source,review)
    review['2007-433:27']['loser_id']='retiring'
    review['2007-433:27']['visual_pages']=[]
    with pytest.raises(ValueError,match='layout'):
        review_checks(sample,source,review)


def test_changed_or_missing_official_bytes_cannot_keep_verified(tmp_path):
    sample,source,review,path=evidence_fixture(tmp_path)
    path.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='changed'):
        review_checks(sample,source,review)
    path.unlink()
    with pytest.raises(ValueError,match='missing'):
        review_checks(sample,source,review)


def test_unavailable_source_stays_unverified(tmp_path):
    sample,_,_,_=evidence_fixture(tmp_path)
    result=review_checks(sample,{}, {})
    assert result[0]['review']['status']=='unverified'
    with pytest.raises(ValueError,match='outside'):
        review_checks(sample,{}, {'extra-target':{'status':'verified'}})


def test_fetch_never_retries_recorded_rejection(tmp_path):
    sample=pd.DataFrame([{'tourney_id':'2012-360'}])
    old={'2012-360':dict(status='unavailable',error='HTTP 429',url='https://www.protennislive.com/posting/2012/360/mds.pdf')}
    with patch('scripts.audit_retirement_source_round10.subprocess.run') as run:
        result=fetch_draws(sample,tmp_path,old)
    run.assert_not_called()
    assert result==old


def test_fetch_rejects_final_year_before_network(tmp_path):
    with patch('scripts.audit_retirement_source_round10.subprocess.run') as run:
        with pytest.raises(AssertionError):
            fetch_draws(pd.DataFrame([{'tourney_id':'2024-580'}]),tmp_path)
    run.assert_not_called()

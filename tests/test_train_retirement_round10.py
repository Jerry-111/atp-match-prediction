import numpy as np
import pandas as pd
import pytest
import json

from scripts.retirement_features import GROUPS, FEATURE_NAMES
from scripts.train_retirement_round10 import candidate_specs, select_single, choose_procedure,verify_source_authorization
from scripts.train_round5 import REFERENCE,fit_model
from scripts.train_models import mirror_features, temporal_masks,sha256,ROOT


def test_nested_input_schema_and_synthetic_swap():
    base=['log_rank_diff','elo_k32_diff','log_rank_mean']
    specs=candidate_specs(dict(base=base,**GROUPS))
    assert [len(s['candidate_input_columns']) for s in specs]==[9,13,13]
    assert set(specs[0]['candidate_input_columns'])<=set(specs[1]['candidate_input_columns'])
    assert set(specs[0]['candidate_input_columns'])<=set(specs[2]['candidate_input_columns'])
    assert len(specs[1]['columns'])==len(specs[2]['columns'])
    assert len(specs[1]['spline_columns'])==len(specs[2]['spline_columns'])
    assert len(specs[1]['interactions'])==len(specs[2]['interactions'])
    rng=np.random.default_rng(3)
    x=pd.DataFrame(rng.normal(size=(80,len(base)+len(FEATURE_NAMES))),columns=base+FEATURE_NAMES)
    rows=pd.DataFrame(dict(a_won=rng.integers(0,2,80),season=np.repeat(2021,80)))
    for spec in specs:
        model=fit_model(spec,x,rows)
        np.testing.assert_allclose(model.predict_proba(x)+model.predict_proba(mirror_features(x)),1,atol=1e-12)


def test_selection_respects_loss_guard_and_early_mask():
    y=np.ones(400,dtype=int);seasons=np.repeat([2020,2021,2022],[100,100,200])
    ref=np.full(400,.7);ref[::10]=.4
    high_accuracy_bad_loss=np.full(400,.51)
    p={REFERENCE:ref,'bad_loss':high_accuracy_bad_loss,'candidate':ref.copy()}
    p['candidate'][:200]=np.full(200,.72)
    early=seasons<2022
    choice=select_single(y,p,early)
    assert choice['choice']=='candidate' and 'bad_loss' not in choice['eligible_models']
    gate=choose_procedure(y,p,seasons)
    assert gate['procedure']=='retain_round4'  # early improvement did not survive separate gate.
    changed={k:v.copy() for k,v in p.items()};changed['candidate'][~early]=.001
    assert select_single(y,changed,early)['choice']=='candidate'
    with pytest.raises(ValueError,match='2020'):
        choose_procedure(y,p,np.where(seasons==2022,2023,seasons))


def test_year_boundary_purge():
    rows=pd.DataFrame(dict(season=[2021,2021,2022],tourney_date=['2021-12-01','2021-12-20','2022-01-03']))
    train,valid=temporal_masks(rows,2022)
    assert train.tolist()==[True,False,False] and valid.tolist()==[False,False,True]


def test_source_gate_stops_contradictions_or_stale_reviews(tmp_path):
    record=dict(source_semantics_fully_verified=False,model_fitting_ready=False,
        root_limited_experiment_authorized=True,observed_official_contradictions=0,
        review_counts={'verified':10,'unverified':20},sample_rows=30,
        audit_code_sha256=sha256(ROOT/'scripts/audit_retirement_source_round10.py'))
    for field,filename in [('sample_registry_sha256','sample_registry.json'),
                           ('official_reviews_sha256','official_reviews.json'),
                           ('official_sources_sha256','official_sources.json')]:
        (tmp_path/filename).write_text('{}')
        record[field]=sha256(tmp_path/filename)
    path=tmp_path/'source_feasibility.json';path.write_text(json.dumps(record))
    assert verify_source_authorization(tmp_path)==record
    record['observed_official_contradictions']=1;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='scope'):verify_source_authorization(tmp_path)
    record['observed_official_contradictions']=0;path.write_text(json.dumps(record))
    (tmp_path/'official_reviews.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Stale'):verify_source_authorization(tmp_path)

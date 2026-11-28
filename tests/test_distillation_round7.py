"""Soft-loss, temporal isolation and teacher-free inference invariants."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.round2_learning import AntisymmetricSplineClassifier
from scripts.train_distillation_round7 import (DistilledSpline, HARD, REFERENCE,
    R5_COMPARATOR, R5_LOSS, SOURCE_COLUMNS, LINK_COLUMNS, build_teacher,
    candidate_specs, choose_procedure, fold_masks, load_teacher,
    predict_distillation, select_candidate, sha256, soft_targets)


def targets(n=40):
    return pd.DataFrame(dict(match_id=[f'm{i}' for i in range(n)], season=2021,
        tourney_date='2021-01-01', tourney_id='2021-1', a_won=np.arange(n)%2))


def teachers(n=40):
    return pd.DataFrame(dict(teacher_probability=np.linspace(.2,.8,n), teacher_available=1,
        teacher_match_utc=pd.to_datetime(['2021-01-02']*n,utc=True), source_match_id=np.arange(n)+1))


class DistillationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(158)
        self.x=pd.DataFrame({'signal_diff':rng.normal(size=40), 'log_rank_diff':rng.normal(size=40),
                             'log_rank_mean':rng.normal(size=40)})
        self.rows, self.teacher=targets(), teachers()
        self.specs=candidate_specs(self.x.columns)

    def test_identical_capacity_and_training_availability_controls(self):
        self.assertEqual(len(self.specs),7)
        baseline={k:v for k,v in self.specs[0].items() if k not in ('name','teacher','lambda_')}
        for spec in self.specs:
            self.assertEqual({k:v for k,v in spec.items() if k not in ('name','teacher','lambda_')},baseline)
        y=self.rows.a_won.to_numpy()
        teacher=self.teacher.copy()
        teacher.loc[0,['teacher_available','teacher_probability']]=[0,np.nan]
        teacher.loc[1,'teacher_match_utc']=pd.Timestamp('2021-03-01',tz='UTC')
        market, m=soft_targets(y,teacher,'market',.5,'2021-02-01')
        uniform, u=soft_targets(y,teacher,'uniform',.5,'2021-02-01')
        np.testing.assert_array_equal(m,u)
        np.testing.assert_array_equal(market[:2],y[:2])
        np.testing.assert_array_equal(uniform[:2],y[:2])
        np.testing.assert_allclose(market[2:],.5*y[2:]+.5*teacher.teacher_probability.to_numpy()[2:])
        np.testing.assert_allclose(uniform[2:],.5*y[2:]+.25)

    def test_fractional_cross_entropy_objective_independent_gradient(self):
        x=pd.DataFrame({'signal_diff':np.ones(40)})
        spec=next(s for s in candidate_specs(x.columns) if s['name']=='market_lambda1')
        spec['params']['C']=1.
        teacher=self.teacher.assign(teacher_probability=.8)
        model=DistilledSpline(spec).fit(x,self.rows,teacher,cutoff='2021-02-01')
        coefficient=model.model_.estimator_.coef_[0,0]
        # Direct derivative of 40 Bernoulli soft losses plus beta^2/(2C).
        self.assertLess(abs(40*(expit(coefficient)-.8)+coefficient),.002)
        self.assertEqual(model.teacher_rows_,40)

    def test_hard_control_exact_and_swap_and_joblib(self):
        hard=DistilledSpline(self.specs[0]).fit(self.x,self.rows,self.teacher,cutoff='2021-02-01')
        original=AntisymmetricSplineClassifier(self.specs[0]).fit(self.x,self.rows.a_won,
                                                                seasons=self.rows.season)
        np.testing.assert_array_equal(hard.predict_proba(self.x),original.predict_proba(self.x))
        swap=self.x.copy()
        swap[[c for c in swap if c.endswith('_diff')]]*=-1
        for spec in (self.specs[2],self.specs[-1]):
            model=DistilledSpline(spec).fit(self.x,self.rows,self.teacher,cutoff='2021-02-01')
            np.testing.assert_allclose(model.predict_proba(swap),1-model.predict_proba(self.x),atol=2e-15)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.joblib'
                joblib.dump(model,path)
                np.testing.assert_array_equal(joblib.load(path).predict_proba(self.x),model.predict_proba(self.x))

    def test_training_seal_indices_and_outside_fit_values(self):
        for rows in (self.rows.assign(season=2023,tourney_date='2023-01-01'),
                     self.rows.assign(tourney_date='2024-01-01')):
            with self.assertRaises(ValueError):
                DistilledSpline(self.specs[1]).fit(self.x,rows,self.teacher,cutoff='2021-02-01')
        with self.assertRaises(ValueError):
            DistilledSpline(self.specs[1]).fit(self.x,self.rows,self.teacher,cutoff='2020-12-31')
        with self.assertRaises(ValueError):
            DistilledSpline(self.specs[1]).fit(self.x,self.rows,self.teacher.set_axis(range(1,41)),cutoff='2021-02-01')
        teacher=self.teacher.copy()
        teacher.loc[0,'teacher_probability']=np.nan
        with self.assertRaises(ValueError):
            DistilledSpline(self.specs[1]).fit(self.x,self.rows,teacher,cutoff='2021-02-01')

    def test_early_split_retains_old_history_and_purges(self):
        rows=pd.DataFrame(dict(match_id=list('abcde'),season=[2009,2021,2021,2021,2022],
            tourney_date=['2009-01-01','2021-06-10','2021-06-11','2021-07-01','2022-01-01']))
        train,valid,cutoff=fold_masks(rows,'2021H2')
        self.assertEqual(cutoff,pd.Timestamp('2021-06-10'))
        self.assertEqual(rows[train].match_id.tolist(),['a','b'])
        self.assertEqual(rows[valid].match_id.tolist(),['d'])
        train,valid,_=fold_masks(rows,'2022')
        self.assertEqual(rows[train].match_id.tolist(),list('abcd'))
        self.assertEqual(rows[valid].match_id.tolist(),['e'])

    def test_selection_uses_early_fold_gate_and_excludes_prior_r5(self):
        specs=[{'name':'early'},{'name':'late'}]
        y=np.ones(4)
        folds=np.array(['2021H2','2021H2','2022','2022'])
        p={REFERENCE:np.full(4,.499),'early':np.full(4,.501),
           'late':np.array([.498,.498,.99,.99]),R5_COMPARATOR:np.full(4,.999)}
        gate=choose_procedure(y,p,specs,folds)
        self.assertEqual(gate['early_selection']['choice'],'early')
        self.assertEqual(gate['procedure'],'adopt_distillation')
        p['early'][2:]=.498
        gate=choose_procedure(y,p,specs,folds)
        self.assertEqual(gate['early_selection']['choice'],'early')
        self.assertEqual(gate['procedure'],'retain_round4')
        with self.assertRaises(ValueError):
            choose_procedure(y,p,specs,['2021H2','2021H2','2023','2023'])
        guarded={REFERENCE:np.array([.55,.55,.49,.1]),'early':np.array([.51,.51,.51,.49])}
        selected=select_candidate(np.array([1,1,1,0]),guarded,[{'name':'early'}],np.ones(4,bool))
        self.assertEqual(selected['choice'],REFERENCE)

    def fixed_fixture(self):
        rows=targets(2)
        links=pd.DataFrame([dict(match_id='m0',season=2021,status='matched',target_event_id='2021-1',
            market_match_id=10,market_event_id=100,market_match_utc='2021-01-02T12:00:00Z',
            market_player_a_id=1,market_player_b_id=2),dict(match_id='m1',season=2021,status='unmatched')])
        links.loc[1,'target_event_id']='2021-1'
        source=pd.DataFrame([dict(match_id=10,source_season=2021,date='2021-01-02T12:00:00Z',
            tournoi_id=100,joueur1_id=2,joueur2_id=1,cote1_ouverture=3.,cote2_ouverture=1.5)])
        return rows,links,source

    def test_fixed_link_quote_orientation_and_future_poisoning(self):
        rows,links,source=self.fixed_fixture()
        teacher=build_teacher(rows,links,source)
        self.assertAlmostEqual(teacher.teacher_probability.iloc[0],2/3)
        self.assertEqual(teacher.teacher_available.tolist(),[1,0])
        future_link=pd.DataFrame([{**{c:'POISON' for c in LINK_COLUMNS},'season':2023}])
        future_source=pd.DataFrame([{**{c:'POISON' for c in SOURCE_COLUMNS},'source_season':2023}])
        poisoned=build_teacher(rows.assign(a_won='POISON'),pd.concat([links,future_link]),
            pd.concat([source.assign(vainqueur_id='POISON',score='POISON'),future_source]))
        pd.testing.assert_frame_equal(teacher,poisoned)
        bad=source.assign(joueur1_id=99)
        with self.assertRaises(ValueError):build_teacher(rows,links,bad)
        with self.assertRaises(ValueError):build_teacher(rows.assign(season=2023,tourney_date='2023-01-01'),links,source)

    def test_loader_opens_only_two_raw_years_and_no_outcomes_or_closing(self):
        rows,links,source=self.fixed_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            lp=path/'links.csv';links.to_csv(lp,index=False)
            entries=[]
            for year in (2021,2022):
                raw=source.drop(columns='source_season').copy()
                if year==2022:raw=raw.iloc[:0]
                raw.to_csv(path/f'{year}.csv',sep=';',index=False)
                entries.append(dict(season=year,sha256=sha256(path/f'{year}.csv'),rows=len(raw)))
            mp=path/'manifest.json';mp.write_text(json.dumps(dict(files=entries)))
            ap=path/'audit.json';ap.write_text(json.dumps(dict(artifact_sha256={'linked_matches.csv':sha256(lp)},
                                                            source_manifest_sha256=sha256(mp))))
            original=pd.read_csv;calls=[]
            def checked(file,*args,**kwargs):
                calls.append((Path(file).name,kwargs.get('usecols')))
                self.assertNotIn('2023',Path(file).name)
                return original(file,*args,**kwargs)
            with patch('scripts.train_distillation_round7.pd.read_csv',side_effect=checked):
                result=load_teacher(rows,cache=path,manifest_path=mp,linkage_path=lp,audit_path=ap)
            self.assertEqual([name for name,_ in calls],['links.csv','2021.csv','2022.csv'])
            self.assertEqual(result.teacher_available.sum(),1)
            for _,columns in calls:
                self.assertFalse(set(columns)&{'a_won','score','vainqueur_id','cote1_cloture','cote2_cloture'})

    def test_prediction_never_uses_teacher_or_labels(self):
        model=DistilledSpline(self.specs[1]).fit(self.x,self.rows,self.teacher,cutoff='2021-02-01')
        bundle=dict(models={model.spec['name']:model},feature_names=list(self.x),previous_bundle={},
                    previous_aliases={REFERENCE:'old',R5_COMPARATOR:'old',R5_LOSS:'old'})
        def prior(_,x):return pd.DataFrame({'old':expit(x.signal_diff)},index=x.index)
        with patch('scripts.train_distillation_round7.predict_round5',side_effect=prior):
            p=predict_distillation(bundle,self.x)
            poisoned=self.x[self.x.columns[::-1]].assign(a_won='POISON',teacher_probability='POISON',
                                                        market_open_logit_diff='POISON')
            pd.testing.assert_frame_equal(p,predict_distillation(bundle,poisoned))
            with self.assertRaises(ValueError):predict_distillation(bundle,self.x.drop(columns='signal_diff'))
            with self.assertRaises(ValueError):predict_distillation(bundle,self.x.assign(signal_diff=np.inf))


if __name__=='__main__':
    unittest.main()

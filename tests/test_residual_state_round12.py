"""Synthetic-only likelihood, chronology and serialization checks for Round12."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from scipy.special import expit

from scripts.residual_state_round12 import (
    C, INITIAL_SD, INNOVATION_SD, ResidualStateModel, _metadata, _solve,
    build_state_design, fit_shared_control, map_objective, sample_weights,
)
from scripts.round2_learning import AntisymmetricSplineClassifier, candidate_specs


def synthetic_fullsize_problem():
    """Optional reproducible numerical smoke; never reads repository datasets."""
    rng=np.random.default_rng(8158);n,d,players=33833,256,925
    loading=rng.normal(size=(8,d));latent=rng.normal(size=(n,8))
    x=latent@loading+.05*rng.normal(size=(n,d))
    x/=np.sqrt(np.mean(x*x,axis=0))
    a=rng.integers(players,size=n)
    b=(a+rng.integers(1,players,size=n))%players
    years=rng.integers(2009,2023,size=n)
    meta=pd.DataFrame({'match_id':[f'synthetic:{i}' for i in range(n)],'season':years,
        'tourney_date':pd.to_datetime([f'{y}-01-05' for y in years]),
        'player_a_id':a.astype(str),'player_b_id':b.astype(str)})
    states=build_state_design(meta,'dynamic');weights,_=sample_weights(meta)
    beta=rng.normal(0,.03,d);u=rng.normal(0,.15,len(states.keys))
    y=rng.binomial(1,expit(x@beta+states.matrix@u))
    return x,states,y,weights


class ResidualStateTests(unittest.TestCase):
    def fixture(self, n=120):
        rng = np.random.default_rng(158)
        x = pd.DataFrame({'log_rank_diff': rng.normal(size=n),
                          'elo_k32_diff': rng.normal(size=n)*100,
                          'log_rank_mean': rng.uniform(1, 5, n)})
        a = np.array(['a', 'a', 'b', 'c']*(n//4))
        b = np.array(['b', 'c', 'c', 'd']*(n//4))
        eta = .6*x.log_rank_diff.to_numpy() + .5*(a == 'a')
        rows = pd.DataFrame({'match_id': [f'train:{i}' for i in range(n)],
            'season': np.repeat([2018, 2019], n//2),
            'tourney_date': np.repeat(['2018-06-04', '2019-06-03'], n//2),
            'player_a_id': a, 'player_b_id': b,
            'a_won': rng.binomial(1, expit(eta))})
        spec = next(s for s in candidate_specs(x.columns)
                    if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
        return spec, x, rows

    def query(self):
        meta = pd.DataFrame({'match_id': ['query:1', 'query:2', 'query:3'],
            'season': [2020]*3, 'tourney_date': ['2020-02-03', '2020-03-02', '2020-03-02'],
            'player_a_id': ['a', 'new', 'cold1'], 'player_b_id': ['b', 'b', 'cold2']}, index=[200,201,202])
        x = pd.DataFrame({'log_rank_diff': [.2, .3, 0.], 'elo_k32_diff': [10., 20., 0.],
                          'log_rank_mean': [3., 4., 3.]}, index=meta.index)
        return x, meta

    def model(self, structure='static'):
        spec, x, rows = self.fixture()
        control = fit_shared_control(spec, x, rows)
        return ResidualStateModel(spec, structure).fit(x, rows, shared_model=control), x, rows

    def history(self, x, rows):
        extra = pd.DataFrame({'match_id': ['extra:1','extra:2','future:1'], 'season': [2020]*3,
            'tourney_date': ['2020-01-13','2020-02-10','2020-03-16'],
            'player_a_id': ['new','new','new'], 'player_b_id': ['b','b','b'], 'a_won': [1,1,0]},
            index=[300,301,302])
        ex = pd.DataFrame({'log_rank_diff': [0.,0.,1.], 'elo_k32_diff': [0.,0.,100.],
                           'log_rank_mean': [3.,3.,4.]}, index=extra.index)
        return pd.concat([x,ex]), pd.concat([rows,extra])

    def test_shared_control_is_exact_original_fit_and_bound(self):
        spec,x,rows=self.fixture()
        old=AntisymmetricSplineClassifier(spec).fit(x,rows.a_won.to_numpy(),rows.season.to_numpy())
        control=fit_shared_control(spec,x,rows)
        np.testing.assert_array_equal(old.predict_proba(x),control.predict_proba(x))
        np.testing.assert_array_equal(old._design(x),control._design(x))
        self.assertEqual(control.residual_training_match_ids_,tuple(rows.match_id))
        with self.assertRaises(ValueError):
            ResidualStateModel(spec,'static').fit(x,rows.iloc[::-1],shared_model=control)
        changed=rows.copy();changed.loc[0,'a_won']=1-changed.loc[0,'a_won']
        with self.assertRaises(ValueError):
            ResidualStateModel(spec,'static').fit(x,changed,shared_model=control)

    def test_prior_and_difference_matrix_are_hand_calculable(self):
        rows=pd.DataFrame({'match_id':['a','b','c'],'season':[2020,2020,2021],
            'tourney_date':['2019-12-30','2020-04-06','2021-01-04'],
            'player_a_id':['p','p','p'],'player_b_id':['q','q','q']})
        meta=_metadata(rows)
        static=build_state_design(meta,'static')
        np.testing.assert_array_equal(static.matrix.toarray(),np.tile([1.,-1.],(3,1)))
        np.testing.assert_array_equal(static.penalty_matrix.toarray(),np.eye(2)/INITIAL_SD)
        self.assertEqual(static.prior_links,())
        dynamic=build_state_design(meta,'dynamic')
        self.assertEqual(dynamic.keys[0],('p','2019Q4'))  # Calendar date, not season2020.
        self.assertEqual(dynamic.node_dates[0],'2019-10-01')
        self.assertEqual(dynamic.keys[-1],('q','2021Q1'))
        self.assertEqual(dynamic.latest_indices,{'p':2,'q':5})
        self.assertAlmostEqual(dynamic.prior_links[0][2],183/365.25)
        u=np.array([.1,.3,.2,-.2,-.1,.0])
        expected=(u[0]**2+u[3]**2)/INITIAL_SD**2
        for old,new,dt in dynamic.prior_links:
            expected+=(u[new]-u[old])**2/(INNOVATION_SD**2*dt)
        self.assertAlmostEqual(float(np.sum((dynamic.penalty_matrix@u)**2)),expected)

    def test_joint_and_conditional_analytic_gradients(self):
        _,x,rows=self.fixture(40);meta=_metadata(rows);rng=np.random.default_rng(9)
        design=x.to_numpy()/np.array([1.,100.,4.]);y=rows.a_won.to_numpy();w,_=sample_weights(meta)
        for structure in ('static','dynamic'):
            states=build_state_design(meta,structure)
            beta=rng.normal(0,.03,3);u=rng.normal(0,.02,len(states.keys))
            for fixed in (None,beta):
                values=np.r_[beta,u] if fixed is None else u
                inputs=design if fixed is None else design@beta
                loss,gradient=map_objective(values,inputs,states,y,w,fixed_beta=fixed)
                eta=design@beta+states.matrix@u
                manual=np.sum(w*(np.logaddexp(0,eta)-y*eta))+.5*np.sum((states.penalty_matrix@u)**2)
                if fixed is None:manual+=.5*np.dot(beta,beta)/C
                self.assertAlmostEqual(loss,manual)
                for j in range(len(values)):
                    step=np.zeros(len(values));step[j]=1e-5
                    hi=map_objective(values+step,inputs,states,y,w,fixed_beta=fixed)[0]
                    lo=map_objective(values-step,inputs,states,y,w,fixed_beta=fixed)[0]
                    self.assertAlmostEqual((hi-lo)/2e-5,gradient[j],places=7)

    def test_zero_state_objective_matches_native_shared_ridge(self):
        spec,x,rows=self.fixture();control=fit_shared_control(spec,x,rows)
        meta=_metadata(rows);design=control._design(x);states=build_state_design(meta,'static')
        beta=control.estimator_.coef_.ravel();w,_=sample_weights(meta);y=rows.a_won.to_numpy()
        loss,gradient=map_objective(np.r_[beta,np.zeros(len(states.keys))],design,states,y,w)
        expected=np.sum(w*(np.logaddexp(0,design@beta)-y*(design@beta)))+.5*np.dot(beta,beta)/C
        self.assertAlmostEqual(loss,expected)
        self.assertLess(np.max(np.abs(gradient[:len(beta)]))/len(y),2e-4)

    def test_persistent_player_signal_and_balanced_null(self):
        spec,x,rows=self.fixture(200);x.loc[:,:]=0
        rows.loc[:,'player_a_id']='a';rows.loc[:,'player_b_id']='b'
        for balanced in (False,True):
            rows.loc[:,'a_won']=np.tile([1,0] if balanced else [1,1,1,0],100 if balanced else 50)
            control=fit_shared_control(spec,x,rows)
            for structure in ('static','dynamic'):
                model=ResidualStateModel(spec,structure).fit(x,rows,shared_model=control)
                qx,q=self.query();qx.loc[:,:]=0
                p=model.predict_asof(qx,q,x,rows)
                if balanced:self.assertAlmostEqual(p[0],.5,places=10)
                else:self.assertGreater(p[0],.6)
                self.assertEqual(p[2],.5)

    def test_asof_cutoff_is_inclusive_and_new_players_can_update(self):
        model,x,rows=self.model();hx,hr=self.history(x,rows);qx,q=self.query()
        p=model.predict_asof(qx,q,hx,hr)
        first=model.state_snapshots_['2020-01-13'];second=model.state_snapshots_['2020-02-10']
        self.assertIn('extra:1',first.eligible_match_ids)
        self.assertNotIn('extra:2',first.eligible_match_ids)
        self.assertIn('extra:2',second.eligible_match_ids)
        self.assertNotIn('future:1',second.eligible_match_ids)
        self.assertNotIn('new',model.training_state_.latest_indices)
        self.assertIn('new',second.latest_indices)
        self.assertGreater(second.coefficients[second.latest_indices['new']],0)
        self.assertEqual(second.differences(q.iloc[[2]])[0],0)
        self.assertEqual(p[2],expit(model._design(qx.iloc[[2]])@model.beta_)[0])
        self.assertEqual(second.diagnostics['weight_anchor_season'],2020)
        self.assertEqual(second.diagnostics['max_eligible_date'],'2020-02-10')

    def test_future_poison_is_never_consumed(self):
        for structure in ('static','dynamic'):
            model,x,rows=self.model(structure);other=deepcopy(model)
            hx,hr=self.history(x,rows);qx,q=self.query()
            expected=model.predict_asof(qx,q,hx,hr)
            hx.loc[302,:]=np.nan;hr['a_won']=hr.a_won.astype(object);hr.loc[302,'a_won']='poison'
            actual=other.predict_asof(qx,q,hx,hr)
            np.testing.assert_array_equal(expected,actual)
            self.assertEqual([s.history_sha256 for s in model.state_snapshots_.values()],
                             [s.history_sha256 for s in other.state_snapshots_.values()])

    def test_query_batch_history_permutation_and_complement(self):
        for structure in ('static','dynamic'):
            model,x,rows=self.model(structure);other=deepcopy(model)
            hx,hr=self.history(x,rows);qx,q=self.query()
            p=model.predict_asof(qx,q,hx,hr)
            reverse=other.predict_asof(qx.iloc[::-1],q.iloc[::-1],hx.iloc[::-1],hr.iloc[::-1])
            np.testing.assert_array_equal(p,reverse[::-1])
            for index in q.index:
                np.testing.assert_array_equal(model.predict_from_snapshots(qx.loc[[index]],q.loc[[index]]),
                                               p[q.index.get_loc(index):q.index.get_loc(index)+1])
            swapped_x=qx.copy();swapped_x[['log_rank_diff','elo_k32_diff']]*=-1
            swapped=q.copy();swapped[['player_a_id','player_b_id']]=q[['player_b_id','player_a_id']].to_numpy()
            np.testing.assert_allclose(model.predict_from_snapshots(swapped_x,swapped),1-p,atol=2e-15,rtol=0)

    def test_prior_and_training_bindings_fail_closed(self):
        model,x,rows=self.model();hx,hr=self.history(x,rows);qx,q=self.query()
        changed=hr.copy();changed.loc[0,'a_won']=1-changed.loc[0,'a_won']
        with self.assertRaises(ValueError):model.predict_asof(qx,q,hx,changed)
        with self.assertRaises(ValueError):model.predict_asof(qx,q,hx.drop(0),hr.drop(0))
        labelled=q.assign(a_won=0)
        with self.assertRaises(ValueError):model.predict_asof(qx,labelled,hx,hr)
        with self.assertRaises(ValueError):model.predict_asof(qx.iloc[::-1],q,hx,hr)
        model.beta_[0]+=1
        with self.assertRaises(ValueError):model.predict_asof(qx,q,hx,hr)

    def test_changed_frozen_transform_and_season_are_rejected(self):
        model,x,rows=self.model();qx,q=self.query()
        early=q.copy();early['tourney_date']='2020-01-06';model.training_max_date_=pd.Timestamp('2019-12-30')
        with self.assertRaises(ValueError):model.predict_asof(qx,early,x,rows)
        model,x,rows=self.model();model.shared_model_.odd_scale_[0]*=2
        with self.assertRaises(ValueError):model.predict_asof(qx,q,x,rows)
        model,x,rows=self.model();future=q.copy();future['season']=2024;future['tourney_date']='2024-02-01'
        with self.assertRaises(ValueError):model.predict_asof(qx,future,x,rows)

    def test_snapshot_is_explicit_and_reload_requires_no_solver(self):
        model,x,rows=self.model('dynamic');hx,hr=self.history(x,rows);qx,q=self.query()
        expected=model.predict_asof(qx,q,hx,hr)
        for snapshot in model.state_snapshots_.values():
            self.assertEqual(len(snapshot.keys),len(snapshot.node_dates))
            self.assertEqual(len(snapshot.keys),len(snapshot.coefficients))
            self.assertTrue(snapshot.diagnostics['success'])
            self.assertLessEqual(snapshot.diagnostics['max_gradient_scaled'],1e-8)
            self.assertAlmostEqual(snapshot.diagnostics['objective_unscaled'],
                snapshot.diagnostics['objective_scaled']*snapshot.diagnostics['rows'])
            self.assertEqual(set(model.training_match_ids_) - set(snapshot.eligible_match_ids),set())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib';joblib.dump(model,path);loaded=joblib.load(path)
            with patch('scripts.residual_state_round12.minimize',side_effect=AssertionError('must not refit')):
                np.testing.assert_array_equal(expected,loaded.predict_from_snapshots(qx,q))
                np.testing.assert_array_equal(expected,loaded.predict_asof(qx,q,hx,hr))
        missing=q.iloc[[0]].copy();missing['tourney_date']='2020-04-06'
        with self.assertRaises(ValueError):model.predict_from_snapshots(qx.iloc[[0]],missing)

    def test_no_silent_cache_reuse_for_changed_eligible_data(self):
        model,x,rows=self.model();hx,hr=self.history(x,rows);qx,q=self.query()
        model.predict_asof(qx,q,hx,hr);changed=hr.copy();changed.loc[300,'a_won']=0
        with self.assertRaises(ValueError):model.predict_asof(qx,q,hx,changed)

    def test_nonbinary_training_and_final_season_fit_are_rejected(self):
        spec,x,rows=self.fixture();bad=rows.copy();bad['a_won']=bad.a_won.astype(float);bad.loc[0,'a_won']=.4
        with self.assertRaises(ValueError):fit_shared_control(spec,x,bad)
        bad=rows.copy();bad['season']=2023;bad['tourney_date']='2023-01-02'
        with self.assertRaises(ValueError):fit_shared_control(spec,x,bad)

    def test_optimizer_failure_cannot_produce_predictions(self):
        _,x,rows=self.fixture(40);meta=_metadata(rows);states=build_state_design(meta,'static')
        fake=OptimizeResult(x=np.zeros(len(states.keys)),success=False,status=1,message='synthetic failure',nit=1,nfev=1)
        with patch('scripts.residual_state_round12.minimize',return_value=fake):
            with self.assertRaises(RuntimeError):
                _solve(np.zeros(len(rows)),states,rows.a_won.to_numpy(),np.ones(len(rows)),fixed_beta=np.zeros(1))


if __name__=='__main__':
    unittest.main()

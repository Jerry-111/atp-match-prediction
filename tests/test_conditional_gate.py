import io
import unittest

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from scripts.conditional_gate import (EXPERTS,CONTEXTS,FEATURE_FIELDS,PROBABILITY_EPS,
    WEIGHT_FLOOR,ContextScaler,context_frame,contrast_matrix,global_objective,
    conditional_objective,fit_global,fit_conditional,equal_mixture,meta_training_mask)


def synthetic(n=400,seed=601):
    rng=np.random.default_rng(seed)
    p=rng.uniform(.15,.85,(n,4));y=rng.integers(0,2,n)
    x=pd.DataFrame(rng.normal(size=(n,5)),columns=CONTEXTS)
    return p,y,x


class ConditionalGateTests(unittest.TestCase):
    def test_objective_gradients_and_orthonormal_zero_sum_coordinates(self):
        p,y,x=synthetic(83);w=np.array([.1,.2,.3,.4]);z=ContextScaler.fit(x).transform(x)
        h=contrast_matrix();assert_allclose(h.T@h,np.eye(3),atol=3e-16)
        assert_allclose(h.sum(axis=0),0,atol=2e-16)
        _,grad=global_objective(w,p,y)
        numerical=approx_derivative(lambda a:global_objective(a,p,y)[0],w,method='3-point').ravel()
        assert_allclose(grad,numerical,atol=2e-9,rtol=2e-7)
        theta=np.linspace(-.2,.3,15)
        _,grad=conditional_objective(theta,p,y,z,w)
        numerical=approx_derivative(lambda a:conditional_objective(a,p,y,z,w)[0],theta,method='3-point').ravel()
        assert_allclose(grad,numerical,atol=2e-9,rtol=2e-7)

    def test_constant_context_exactly_uses_same_floored_global_control(self):
        p,y,_=synthetic();x=pd.DataFrame(np.tile([1,2,3,4,0],(len(y),1)),columns=CONTEXTS)
        global_model=fit_global(p,y);conditional=fit_conditional(p,y,x,global_model)
        assert_allclose(conditional.theta,0,atol=0,rtol=0)
        expected=np.broadcast_to(global_model.weights,(len(y),4))
        assert_allclose(conditional.weights_for(x),expected,atol=0,rtol=0)
        # Even a future surprise in a formerly constant field cannot activate it.
        surprise=x+1e9
        assert_allclose(conditional.weights_for(surprise),expected,atol=0,rtol=0)
        assert_allclose(conditional.predict_proba(p,x),global_model.predict_proba(p),atol=2e-16)

    def test_schema_even_context_exposure_identity_and_training_only_scaler(self):
        raw=pd.DataFrame({FEATURE_FIELDS[0]:[1.,2.,3.],FEATURE_FIELDS[1]:[10,20,30],
            FEATURE_FIELDS[2]:[3,5,7],FEATURE_FIELDS[3]:[100,400,900],FEATURE_FIELDS[4]:[0,1,0],
            'arbitrary_diff':[8.,-2.,3.],'a_won':[1,0,1]})
        contexts=context_frame(raw);swapped=raw.copy();swapped.arbitrary_diff*=-1;swapped.a_won=1-swapped.a_won
        assert_allclose(contexts,context_frame(swapped),atol=0,rtol=0)
        a=np.array([100,500,900]);b=np.array([200,400,1000])
        assert_allclose(contexts.point_coverage,(np.log1p(a)+np.log1p(b)-np.abs(np.log1p(a)-np.log1p(b)))/2)
        scaler=ContextScaler.fit(contexts.iloc[:2]);mean=scaler.mean.copy()
        query=contexts.iloc[2:].copy();query.iloc[0,0]=1e9
        assert scaler.transform(query)[0,0]==3
        assert_allclose(scaler.mean,mean,atol=0,rtol=0)

    def test_swap_average_bounds_query_independence_serialization_and_no_input_mutation(self):
        p,y,x=synthetic();original=p.copy();g=fit_global(p,y);model=fit_conditional(p,y,x,g)
        prediction=model.predict_proba(p,x)
        assert_allclose(model.predict_proba(1-p,x)+prediction,1,atol=3e-16)
        assert_allclose(prediction,model.predict_proba(p[::-1],x.iloc[::-1])[::-1],atol=2e-16)
        chunks=np.concatenate([model.predict_proba(p[:17],x.iloc[:17]),model.predict_proba(p[17:],x.iloc[17:])])
        assert_allclose(prediction,chunks,atol=2e-16)
        assert_allclose(model.weights_for(x).sum(1),1,atol=3e-16)
        assert (prediction>=p.min(1)).all() and (prediction<=p.max(1)).all()
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
        assert_allclose(prediction,joblib.load(stream).predict_proba(p,x),atol=0,rtol=0)
        assert_allclose(p,original,atol=0,rtol=0)

    def test_global_boundary_optimum_uses_one_shared_floor_and_rejects_bad_data(self):
        y=np.tile([0,1],30);good=np.where(y,.9,.1)
        p=np.column_stack([good,np.full(len(y),.5),1-good,np.full(len(y),.6)])
        g=fit_global(p,y)
        assert_allclose(g.weights,[1-3*WEIGHT_FLOOR,*([WEIGHT_FLOOR]*3)],atol=2e-9)
        with self.assertRaisesRegex(ValueError,'binary'):fit_global(p,np.full(len(y),.4))
        with self.assertRaisesRegex(ValueError,'ordered'):
            equal_mixture(pd.DataFrame(p,columns=EXPERTS[::-1]))
        p[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):fit_global(p,y)

    def test_meta_chronology_purge_warmup_and_date_poisoning(self):
        rows=pd.DataFrame(dict(match_id=list('abcdef'),season=[2020,2020,2021,2021,2022,2022],
            tourney_date=['2020-01-01','2020-12-25','2021-01-04','2021-12-25','2022-01-03','2022-10-31']))
        train,meta=meta_training_mask(rows,rows[rows.season.eq(2021)])
        self.assertEqual(rows.loc[train,'match_id'].tolist(),['a'])
        self.assertEqual(meta['purged_rows'],1)
        train,_=meta_training_mask(rows,rows[rows.season.eq(2022)])
        self.assertEqual(rows.loc[train,'match_id'].tolist(),['a','b','c'])
        with self.assertRaisesRegex(ValueError,'2021'):
            meta_training_mask(rows,rows[rows.season.eq(2020)])
        bad=rows.copy();bad.loc[0,'tourney_date']='2024-01-01'
        with self.assertRaisesRegex(ValueError,'date'):meta_training_mask(bad,rows[rows.season.eq(2022)])
        bad=rows.copy();bad.loc[0,'season']=2023
        with self.assertRaisesRegex(ValueError,'2020'):
            meta_training_mask(bad,rows[rows.season.eq(2022)])

    def test_known_synthetic_reliability_switch_beats_global_on_separate_queries(self):
        rng=np.random.default_rng(785)
        x=pd.DataFrame(rng.normal(size=(2000,5)),columns=CONTEXTS)
        p=np.tile([.85,.15,.55,.45],(2000,1))
        truth=np.where(x.rank_gap.to_numpy()>0,.8,.2);y=rng.binomial(1,truth)
        g=fit_global(p[:1200],y[:1200]);model=fit_conditional(p[:1200],y[:1200],x.iloc[:1200],g)
        q=model.predict_proba(p[1200:],x.iloc[1200:]);baseline=g.predict_proba(p[1200:])
        from scripts.conditional_gate import bernoulli_loss
        self.assertLess(bernoulli_loss(y[1200:],q),bernoulli_loss(y[1200:],baseline)-.02)
        self.assertLess(model.diagnostics['max_abs_gradient'],1e-6)

    def test_context_can_reactivate_an_expert_at_global_floor(self):
        rng=np.random.default_rng(817)
        regime=rng.random(6000)<.2;orientation=rng.integers(0,2,len(regime))
        x=pd.DataFrame(np.zeros((len(regime),5)),columns=CONTEXTS);x.rank_gap=regime.astype(float)
        p=np.tile([.65,.01,.5,.55],(len(regime),1));p[orientation==0]=1-p[orientation==0]
        truth=np.where(regime,.15,.85);truth[orientation==0]=1-truth[orientation==0]
        y=rng.binomial(1,truth);g=fit_global(p,y);model=fit_conditional(p,y,x,g)
        self.assertAlmostEqual(g.weights[1],WEIGHT_FLOOR,places=8)
        self.assertGreater(model.weights_for(x)[regime,1].mean(),.1)
        self.assertGreater(model.weights_for(x)[regime,1].mean(),model.weights_for(x)[~regime,1].mean())

    def test_identical_experts_shrink_gate_to_global(self):
        rng=np.random.default_rng(820);single=rng.uniform(.1,.9,180)
        p=np.tile(single[:,None],(1,4));y=rng.binomial(1,single)
        x=pd.DataFrame(rng.normal(size=(180,5)),columns=CONTEXTS)
        g=fit_global(p,y);model=fit_conditional(p,y,x,g)
        assert_allclose(model.theta,0,atol=1e-12)
        assert_allclose(model.predict_proba(p,x),single,atol=2e-16)

    def test_named_probability_context_and_label_indices_must_align(self):
        p,y,x=synthetic();frame=pd.DataFrame(p,columns=EXPERTS)
        g=fit_global(frame,pd.Series(y));model=fit_conditional(frame,pd.Series(y),x,g)
        with self.assertRaisesRegex(ValueError,'row indices'):
            fit_conditional(frame,pd.Series(y),x.iloc[::-1],g)
        with self.assertRaisesRegex(ValueError,'row indices'):
            model.predict_proba(frame,x.iloc[::-1])
        with self.assertRaisesRegex(ValueError,'row indices'):
            fit_global(frame,pd.Series(y).iloc[::-1])

    def test_nearly_identical_experts_do_not_generate_large_routing(self):
        rng=np.random.default_rng(821);single=rng.uniform(.2,.8,240)
        p=single[:,None]+rng.normal(scale=1e-6,size=(240,4));y=rng.binomial(1,single)
        x=pd.DataFrame(rng.normal(size=(240,5)),columns=CONTEXTS)
        g=fit_global(p,y);model=fit_conditional(p,y,x,g)
        self.assertLess(np.linalg.norm(model.theta),1e-3)
        self.assertLess(np.max(np.abs(model.weights_for(x)-g.weights[None,:])),1e-3)


if __name__=='__main__':unittest.main()

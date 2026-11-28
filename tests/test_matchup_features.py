import copy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.matchup_features import (STYLE_NAMES,SHARED_NAMES,SURFACE_NAMES,
    MatchupTransformer,MatchupClassifier,candidate_specs,reconstruct_styles,skew_products)
from scripts.train_models import mirror_features,SymmetricClassifier


def frame(a,b):
    out={}
    for i,name in enumerate(STYLE_NAMES):
        out[f'style_{name}_diff']=a[:,i]-b[:,i]
        out[f'style_{name}_sum']=a[:,i]+b[:,i]
    n=len(a)
    out['log_rank_diff']=np.linspace(-1,1,n)
    for i,s in enumerate(('hard','clay','grass')):out['surface_is_'+s]=(np.arange(n)%3==i).astype(float)
    return pd.DataFrame(out)


def fixture(n=120):
    rng=np.random.default_rng(158)
    x=frame(rng.uniform(.1,.8,(n,7)),rng.uniform(.1,.8,(n,7)))
    rows=pd.DataFrame(dict(match_id=[f'm{i}' for i in range(n)],season=np.where(np.arange(n)%2,2021,2022),
        tourney_date=np.where(np.arange(n)%2,'2021-06-01','2022-06-01'),a_won=rng.integers(0,2,n)))
    return x,rows


class MatchupFeatureTests(unittest.TestCase):
    def test_hand_products_recover_styles_and_skew_matrix(self):
        a=np.array([[.1,.2,.3,.4,.5,.6,.7]])
        b=np.array([[.7,.5,.6,.2,.3,.4,.1]])
        x=frame(a,b)
        first,second=reconstruct_styles(x)
        np.testing.assert_allclose(first,a,atol=1e-16)
        np.testing.assert_allclose(second,b,atol=1e-16)
        z=skew_products(a,b)
        self.assertEqual(z.shape,(1,21))
        self.assertAlmostEqual(z[0,0],.1*.5-.2*.7)
        np.testing.assert_array_equal(skew_products(b,a),-z)
        np.testing.assert_array_equal(skew_products(a,a),np.zeros((1,21)))
        rng=np.random.default_rng(9);matrix=rng.normal(size=(7,7));matrix=matrix-matrix.T
        theta=matrix[np.triu_indices(7,k=1)]
        np.testing.assert_allclose(z@theta,np.einsum('ni,ij,nj->n',a,matrix,b),atol=1e-15)

    def test_train_only_normalization_and_partial_pooling_not_rescaled(self):
        x,rows=fixture();transform=MatchupTransformer().fit(x.iloc[:80])
        a,b=reconstruct_styles(x.iloc[:80]);players=np.r_[a,b]
        np.testing.assert_array_equal(transform.center_,players.mean(axis=0))
        np.testing.assert_array_equal(transform.scale_,players.std(axis=0))
        out=transform.transform(x.iloc[:80])
        np.testing.assert_allclose(np.sqrt(np.mean(out[SHARED_NAMES].to_numpy()**2,axis=0)),.2,atol=1e-15)
        for j,s in enumerate(('hard','clay','grass')):
            np.testing.assert_array_equal(out[SURFACE_NAMES[j*21:(j+1)*21]],
                .5*out[SHARED_NAMES].to_numpy()*x.iloc[:80][['surface_is_'+s]].to_numpy())
        mean=transform.center_.copy();scale=transform.scale_.copy()
        future=x.iloc[80:].copy();future.loc[:,[c for c in future if c.endswith('_sum')]]=1.5
        future.loc[:,[c for c in future if c.startswith('style_') and c.endswith('_diff')]]=.01
        transform.transform(future)
        np.testing.assert_array_equal(mean,transform.center_);np.testing.assert_array_equal(scale,transform.scale_)
        carpet=x.copy();carpet[['surface_is_hard','surface_is_clay','surface_is_grass']]=0
        self.assertEqual(np.count_nonzero(transform.transform(carpet)[SURFACE_NAMES]),0)

    def test_linear_control_matches_prior_learner_and_common_design(self):
        x,rows=fixture()
        specs=candidate_specs(x)
        models=[MatchupClassifier(s).fit(x,rows) for s in specs]
        old=SymmetricClassifier(dict(kind='logistic',columns=specs[0]['main_columns'],params={'C':.01},half_life=3.))
        old.fit(x,rows.a_won.to_numpy(),rows.season.to_numpy())
        np.testing.assert_allclose(models[0].predict_proba(x),old.predict_proba(x),atol=1e-12,rtol=0)
        width=len(specs[0]['main_columns'])
        for model in models[1:]:np.testing.assert_array_equal(model.design(x)[:,:width],models[0].design(x))
        self.assertEqual([m.design(x).shape[1] for m in models],[width,width+21,width+84])

    def test_cyclic_styles_need_more_than_additive_strengths(self):
        vertices=np.full((3,7),.5);vertices[:,:3]=np.eye(3)*.6+.1
        # Every vertex beats the next; add reversed orientations with complements.
        a=np.repeat(vertices,120,axis=0);b=np.repeat(vertices[[1,2,0]],120,axis=0)
        x=frame(np.r_[a,b],np.r_[b,a]);x['log_rank_diff']=0
        rows=pd.DataFrame(dict(match_id=[f'm{i}' for i in range(len(x))],season=2022,
            tourney_date='2022-06-01',a_won=np.r_[np.ones(len(a)),np.zeros(len(a))]))
        specs=candidate_specs(x)
        baseline=MatchupClassifier(specs[0]).fit(x,rows)
        shared=MatchupClassifier(specs[1]).fit(x,rows)
        p0,p1=baseline.predict_proba(x),shared.predict_proba(x)
        self.assertLessEqual(np.mean((p0>=.5)==rows.a_won),2/3)
        self.assertEqual(np.mean((p1>=.5)==rows.a_won),1.)
        self.assertTrue((p1[:len(a)]>.5).all())

    def test_swap_reload_column_order_and_unused_label_poison(self):
        x,rows=fixture()
        for spec in candidate_specs(x):
            model=MatchupClassifier(spec).fit(x,rows)
            p=model.predict_proba(x)
            np.testing.assert_allclose(p+model.predict_proba(mirror_features(x)),1,atol=2e-15,rtol=0)
            contributions=model.contributions(x)
            np.testing.assert_allclose(expit(contributions.sum(axis=1)),p,atol=2e-15,rtol=0)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.joblib';joblib.dump(model,path)
                loaded=joblib.load(path)
                np.testing.assert_array_equal(loaded.predict_proba(x[x.columns[::-1]].assign(a_won='POISON')),p)

    def test_invalid_year_identity_features_fail_closed(self):
        x,rows=fixture();spec=candidate_specs(x)[1]
        for badrows in (rows.assign(season=2023,tourney_date='2023-01-01'),rows.assign(tourney_date='2024-01-01'),
                        rows.assign(a_won=.5),rows.assign(match_id='duplicate')):
            with self.assertRaises(ValueError):MatchupClassifier(spec).fit(x,badrows)
        for bad in (x.iloc[:0],x.drop(columns='style_ace_sum'),x.assign(style_ace_sum=3),
                    x.assign(style_ace_diff=np.inf)):
            with self.assertRaises(ValueError):MatchupTransformer().fit(bad)
        transform=MatchupTransformer().fit(x)
        with self.assertRaises(ValueError):transform.transform(x.assign(surface_is_hard=1,surface_is_clay=1))


if __name__=='__main__':unittest.main()

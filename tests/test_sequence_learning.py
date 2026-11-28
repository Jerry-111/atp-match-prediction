"""Optional Torch tests: cleanly skipped in the base non-neural environment."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from scripts.sequence_features import COLUMNS,TOKENS,SequenceFeatures
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from scripts.sequence_learning import (ARCHITECTURES,SEEDS,SequenceNormalizer,SequenceNet,
        FittedSequence,fit_sequence,inner_masks,network_logits,prepare_arrays,predict_average,
        recency_weights,shuffle_table)
    from scripts.train_models import mirror_features


def synthetic(n=32):
    rng=np.random.default_rng(9)
    x=pd.DataFrame(dict(signal_diff=rng.normal(size=n),context=rng.normal(size=n)))
    mask=np.zeros((n,2,TOKENS),bool)
    for i in range(n):
        for side in range(2):
            length=(i+side)%17
            if length:mask[i,side,-length:]=True
    values=rng.normal(size=(n,2,TOKENS,len(COLUMNS))).astype(np.float32);values[~mask]=0
    dates=np.where(mask,730000+np.arange(TOKENS),-1).astype(np.int32)
    s=SequenceFeatures(values,mask,dates,np.array([f'm{i}' for i in range(n)]),x.index.to_numpy(),COLUMNS,{})
    return x,s


@unittest.skipUnless(HAS_TORCH,'Optional Torch environment required (.venv-foundation)')
class SequenceLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_train_only_scaler_exchange_and_padded_invariance(self):
        x,s=synthetic();norm=SequenceNormalizer.fit(x.iloc[:16],s.take(np.arange(16)))
        center=norm.token_center.copy();staticcenter=norm.static_center.copy()
        x2=x.copy();x2.iloc[16:]=1e6;s2=s.take(np.arange(32));s2.values=s2.values.copy();s2.values[16:]=1e5
        same=SequenceNormalizer.fit(x2.iloc[:16],s2.take(np.arange(16)))
        np.testing.assert_array_equal(norm.token_center,same.token_center)
        np.testing.assert_array_equal(norm.static_center,same.static_center)
        a,t,m=norm.transform(x,s);b,u,n=norm.transform(mirror_features(x),s.swapped())
        np.testing.assert_array_equal(b[:,0],-a[:,0]);np.testing.assert_array_equal(b[:,1],a[:,1])
        np.testing.assert_array_equal(u,t[:,::-1]);np.testing.assert_array_equal(n,m[:,::-1])
        poison=s.take(np.arange(32));poison.values=poison.values.copy();poison.values[~poison.mask]=1e20
        np.testing.assert_array_equal(norm.transform(x,poison)[1],t)
        np.testing.assert_array_equal(norm.token_center,center);np.testing.assert_array_equal(norm.static_center,staticcenter)

    def test_three_encoders_complement_and_zero_history(self):
        x,s=synthetic();norm=SequenceNormalizer.fit(x,s)
        for kind in ARCHITECTURES:
            with self.subTest(kind=kind):
                torch.manual_seed(158);net=SequenceNet(kind,x.columns)
                a=network_logits(net,prepare_arrays(norm,x,s,'cpu'))
                b=network_logits(net,prepare_arrays(norm,mirror_features(x),s.swapped(),'cpu'))
                np.testing.assert_allclose(a,-b,atol=1e-7)
                zero=net.encode(torch.zeros(3,TOKENS,len(COLUMNS)),torch.zeros(3,TOKENS,dtype=torch.bool))
                np.testing.assert_array_equal(zero.detach().numpy(),0)

    def test_fixed_shuffle_only_changes_performance_channels(self):
        x,s=synthetic();net=SequenceNet('sequence_shuffled_gru',x.columns)
        table=shuffle_table()
        for length in range(17):
            self.assertEqual(sorted(table[length].tolist()),list(range(16)))
            if length:self.assertTrue((table[length,-length:]>=16-length).all())
        self.assertFalse(np.array_equal(table[16],np.arange(16)))
        # Record exactly what enters the common projection.
        tokens=torch.arange(2*16*22,dtype=torch.float32).reshape(2,16,22)
        mask=torch.ones((2,16),dtype=torch.bool);captured=[]
        hook=net.projection.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy()))
        net.encode(tokens,mask);hook.remove();seen=captured[0]
        np.testing.assert_array_equal(seen[:,:,-2:],tokens.numpy()[:,:,-2:])
        np.testing.assert_array_equal(seen[:,:,:-2],tokens.numpy()[:,table[16],:-2])

    def test_inner_purge_fixed_seed_average_and_joblib_reload(self):
        x,s=synthetic(24)
        rows=pd.DataFrame(dict(match_id=s.match_ids,season=np.r_[np.full(12,2018),np.full(12,2019)],
            tourney_date=['2018-01-15']*11+['2018-12-20']+['2019-01-01']*12,a_won=np.arange(24)%2))
        tr,va,cutoff=inner_masks(rows)
        self.assertEqual(str(cutoff.date()),'2018-12-11');self.assertFalse(tr[11]);self.assertTrue(va[12:].all())
        self.assertAlmostEqual(recency_weights(rows[tr]).mean(),1,places=6)
        with patch('scripts.sequence_learning.MAX_EPOCHS',2):
            models=[fit_sequence('sequence_mean',seed,x,rows,s) for seed in SEEDS]
        expected=np.mean([m.predict_proba(x,s) for m in models],axis=0)
        np.testing.assert_array_equal(predict_average(models,x,s),expected)
        self.assertEqual(models[0].metadata['inner_training_rows'],11)
        self.assertEqual(models[0].metadata['refit_rows'],24)
        with self.assertRaises(ValueError):predict_average(models[::-1],x,s)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'models.joblib';joblib.dump(models,p)
            np.testing.assert_array_equal(predict_average(joblib.load(p),x,s),expected)

    def test_fixed_blends_and_selection_population(self):
        from scripts.train_sequence_round9 import candidate_specs,compose_predictions
        from scripts.train_round5 import REFERENCE
        self.assertEqual(len(candidate_specs()),12)
        averages={k:np.array([.2,.8]) for k in ARCHITECTURES};reference=np.array([.6,.4])
        out=compose_predictions(averages,reference)
        self.assertEqual(len(out.columns),13)
        np.testing.assert_array_equal(out[REFERENCE],reference)
        for name in ARCHITECTURES:
            np.testing.assert_array_equal(out[name],averages[name])
            for weight in (.25,.5,.75):
                np.testing.assert_array_equal(out[f'{name}_r4blend{weight:g}'],weight*averages[name]+(1-weight)*reference)

    def test_gru_recovers_synthetic_directional_change(self):
        # This is an expressiveness smoke test, not a tennis performance test.
        rng=np.random.default_rng(2718);n=192
        static=torch.zeros(n,2);mask=torch.zeros(n,2,16,dtype=torch.bool);mask[:,:,-2:]=True
        tokens=torch.zeros(n,2,16,22)
        observations=rng.normal(size=(n,2,2)).astype(np.float32)
        tokens[:,:,-2:,0]=torch.tensor(observations)
        tokens[:,:,-2:,-1]=torch.tensor([1.,0.])
        labels=torch.tensor((observations[:,0,1]-observations[:,0,0]>observations[:,1,1]-observations[:,1,0]).astype(np.float32))
        torch.manual_seed(158);net=SequenceNet('sequence_gru',['signal_diff','context'])
        opt=torch.optim.Adam(net.parameters(),lr=.01)
        for _ in range(120):
            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(net(static,tokens,mask),labels)
            loss.backward();opt.step()
        predicted=(net(static,tokens,mask).detach().numpy()>=0)
        self.assertGreater(np.mean(predicted==labels.numpy()),.9)


if __name__=='__main__':unittest.main()

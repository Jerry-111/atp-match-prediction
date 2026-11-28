import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Torch is optional; run in .venv-foundation')

import io
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scripts.tabular_neural import (NTPLinear,RobustScaleSmoothClip,make_network,
    schedule_value,train_network,mirror,inner_masks,fit_ensemble,predict_ensemble,
    configure_device,candidate_specs)






def sample_rows():
    return pd.DataFrame(dict(match_id=list('abcdef'),season=[2017,2018,2018,2019,2019,2019],
        tourney_date=['2017-01-01','2018-10-01','2018-12-20','2019-01-01','2019-02-01','2019-03-01'],
        a_won=[0,1,0,1,1,0]))

def raises(exception,match=None):
    case=unittest.TestCase()
    return case.assertRaises(exception) if match is None else case.assertRaisesRegex(exception,match)


class TabularNeuralTests(unittest.TestCase):
    def test_author_scaling_equation_zero_iqr_and_constant_future_values(self):
        x=np.array([[0,0,2],[0,0,2],[0,0,2],[4,8,2]],float)
        prep=RobustScaleSmoothClip().fit(x)
        query=np.array([[2,4,100]],float)
        z=(query-np.median(x,axis=0))*prep.factors_
        assert_allclose(prep.transform(query),z/np.sqrt(1+(z/3)**2),atol=1e-7)
        assert prep.transform(query)[0,2]==0
        # More than 75% identical values exercise the zero-IQR min–max fallback.
        prep=RobustScaleSmoothClip().fit(np.array([[0],[0],[0],[0],[4]],float))
        assert prep.factors_[0]==.5


    def test_author_ntp_formula_initialization_schedule_and_parameter_groups(self):
        torch.manual_seed(7)
        layer=NTPLinear(3,2)
        x=torch.tensor([[1.,2.,3.]],requires_grad=True)
        expected=x@layer.weight/np.sqrt(3)+layer.bias
        torch.testing.assert_close(layer(x),expected)
        layer(x).sum().backward()
        torch.testing.assert_close(x.grad,layer.weight.detach().sum(axis=1)[None,:]/np.sqrt(3))
        model=make_network('realmlp',326)
        assert sum(p.numel() for p in model.parameters())==216136
        assert torch.count_nonzero(model[-1].weight)==0
        assert torch.count_nonzero(model[-1].bias)==0
        from scripts.tabular_neural import _optimizer
        opt=_optimizer(model,'realmlp');params=list(model.parameters())
        assert len(opt.param_groups[0]['params'])==1
        assert [id(p) for p in opt.param_groups[1]['params']]==[id(p) for p in params[1::2]]
        assert [id(p) for p in opt.param_groups[2]['params']]==[id(p) for p in params[2::2]]
        assert opt.defaults['betas']==(.9,.95)
        assert_allclose(schedule_value(np.array([0,.2,1])),
                        .5-.5*np.cos(2*np.pi*np.log2(1+15*np.array([0,.2,1]))),atol=1e-15)


    def test_inner_tail_purge_dates_and_final_season_guard(self):
        rows=sample_rows();early,tail,metadata=inner_masks(rows)
        assert list(rows.loc[early,'match_id'])==['a','b']
        assert list(rows.loc[tail,'match_id'])==['d','e','f']
        assert metadata['purged_rows']==1
        poisoned=rows.copy();poisoned.loc[0,'tourney_date']='2024-01-01'
        with raises(ValueError,match='date'):inner_masks(poisoned)
        poisoned=rows.copy();poisoned.loc[5,'season']=2023
        with raises(ValueError,match='2009'):inner_masks(poisoned)


    def test_probe_sees_only_earlier_rows_refit_sees_all_and_keeps_schedule_horizon(self):
        rows=sample_rows();x=pd.DataFrame({'a_diff':np.arange(6),'b_sum':np.arange(6)+10})
        calls=[]
        def fake(xx,yy,columns,kind,seed,epochs,**kwargs):
            calls.append((xx.copy(),yy.copy(),seed,epochs,kwargs))
            return dict(selected_epoch=3,epochs=epochs,kind=kind,seed=seed,inner_losses=[.7,.6,.5],
                        state_dict={},prep=None)
        spec=candidate_specs()[0]
        with patch('scripts.tabular_neural.train_network',fake):
            result=fit_ensemble(x,rows,spec)
        assert len(calls)==6
        for i in range(0,6,2):
            assert len(calls[i][0])==2
            assert len(calls[i][4]['validation'][0])==3
            assert calls[i][3]==256
            assert len(calls[i+1][0])==6 and calls[i+1][3]==3
            assert 'schedule_epochs' not in calls[i+1][4]  # remains default 256, not selected 3
        assert result['training_ids']==list('abcdef')


    def test_real_fit_serialization_average_swap_chunk_order_and_query_poison(self):
        configure_device('cpu')
        rng=np.random.default_rng(42);x=pd.DataFrame(rng.normal(size=(24,3)),columns=['a_diff','b_sum','c_diff'])
        y=rng.integers(0,2,len(x));members=[]
        for seed in (17,18):
            members.append(train_network(x.to_numpy(),y,list(x),'plain',seed,2))
        bundle=dict(columns=list(x),members=members)
        p=predict_ensemble(bundle,x,batch_size=7)
        singles=[predict_ensemble(dict(bundle,members=[m]),x,batch_size=7) for m in members]
        assert_allclose(p,np.mean(singles,axis=0),atol=0,rtol=0)
        swapped=pd.DataFrame(mirror(x,list(x)),columns=x.columns)
        assert_allclose(p+predict_ensemble(bundle,swapped,batch_size=7),1,atol=2e-7)
        assert_allclose(p,predict_ensemble(bundle,x,batch_size=1),atol=2e-6)
        assert_allclose(p,predict_ensemble(bundle,x.iloc[::-1],batch_size=5)[::-1],atol=2e-6)
        buf=io.BytesIO();joblib.dump(bundle,buf);buf.seek(0)
        assert_allclose(p,predict_ensemble(joblib.load(buf),x,batch_size=7),atol=0,rtol=0)
        xx=x.copy();xx['a_won']=1-y
        with raises(ValueError,match='schema'):predict_ensemble(bundle,xx)
        with raises(ValueError,match='binary'):
            train_network(x.to_numpy(),np.full(len(x),.4),list(x),'plain',17,1)
        with raises(ValueError,match='inner validation'):
            train_network(x.to_numpy(),y,list(x),'plain',17,1,validation=(x.to_numpy(),np.full(len(x),.4)))
        xx=x.copy();xx.iloc[0,0]=np.inf
        with raises(ValueError,match='Finite'):predict_ensemble(bundle,xx)


    def test_train_randomness_is_seeded_and_input_is_not_mutated(self):
        configure_device('cpu')
        x=np.arange(24,dtype=float).reshape(8,3);original=x.copy();y=np.arange(8)%2
        a=train_network(x,y,['a_diff','b_sum','c_diff'],'realmlp',20,2)
        b=train_network(x,y,['a_diff','b_sum','c_diff'],'realmlp',20,2)
        for key in a['state_dict']:torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],atol=0,rtol=0)
        assert_allclose(x,original,atol=0,rtol=0)



if __name__ == '__main__':unittest.main()

import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Torch is optional; run in .venv-foundation')

from unittest.mock import patch
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scripts.tabular_neural import candidate_specs
from scripts.train_neural_round9 import (REFERENCE,R5_ACCURACY,R5_LOSS,add_blends,
    blend_registry,selectable,select_candidate,choose_procedure,predict_neural,verify_targets)

def raises(exception,match=None):
    case=unittest.TestCase()
    return case.assertRaises(exception) if match is None else case.assertRaisesRegex(exception,match)


class NeuralProtocolTests(unittest.TestCase):
    def test_fixed_menu_excludes_perfect_old_r5_comparators_and_seed_diagnostics(self):
        specs=candidate_specs();y=np.tile([0,1],10);p=np.full(len(y),.6)
        predictions=add_blends({REFERENCE:p,**{s['name']:p for s in specs}},specs)
        for name in (R5_ACCURACY,R5_LOSS,'seed1741'):
            predictions[name]=np.where(y,.99,.01)
        assert len(specs)==2 and all(s['seeds']==[1741,1742,1743] for s in specs)
        assert len(blend_registry(specs))==6 and len(selectable(predictions,specs))==9
        result=select_candidate(y,predictions,np.ones(len(y),bool),specs)
        assert result['choice'] not in (R5_ACCURACY,R5_LOSS,'seed1741')


    def test_gate_requires_half_percentage_point_and_excludes_2023(self):
        specs=candidate_specs();seasons=np.repeat([2020,2021,2022],1000)
        index=np.tile(np.arange(1000),3);y=index%2;correct=index<650
        reference=np.where(correct==y,.6,.4)
        for improvement,passed in [(4,False),(5,True)]:
            better=correct.copy()
            better[(seasons<2022)&(index>=650)&(index<660)]=True
            better[(seasons==2022)&(index>=650)&(index<650+improvement)]=True
            q=np.where(better==y,.6,.4)
            pred=add_blends({REFERENCE:reference,**{s['name']:q for s in specs}},specs)
            gate=choose_procedure(y,pred,seasons,specs)
            assert (gate['procedure']=='adopt_neural')==passed
        with raises(ValueError):choose_procedure(y,pred,np.where(seasons==2022,2023,seasons),specs)
        # A perfect accuracy model with catastrophic confident mistakes violates the loss guard.
        bad=reference.copy();bad[:]=np.where(y,.999999,.000001)
        bad[:300]=1-bad[:300]
        pred=add_blends({REFERENCE:reference,**{s['name']:bad for s in specs}},specs)
        from scripts.train_models import mean_loss
        self.assertGreater(mean_loss(y,bad),mean_loss(y,reference)+.005)
        result=select_candidate(y,pred,np.ones(len(y),bool),specs)
        assert result['choice'] not in [s['name'] for s in specs]


    def test_metadata_date_rank_season_and_label_poisoning_fail_closed(self):
        rows=pd.DataFrame(dict(match_id=['a','b'],season=[2019,2020],player_a_id=['A','C'],
            player_b_id=['B','D'],surface=['Hard','Clay'],rank_a=[1,3],rank_b=[2,4],a_won=[1,0],
            tourney_date=['2019-01-01','2020-01-01']))
        verify_targets(rows,rows.copy())
        for field,value in [('tourney_date','2024-01-01'),('season',2024),('rank_a',99),('a_won',0)]:
            altered=rows.copy();altered.loc[0,field]=value
            with raises(ValueError,match='changed'):verify_targets(altered,rows)


    def test_inference_id_alignment_ignores_outcomes_and_fixed_blends_complement(self):
        specs=candidate_specs();bundle=dict(specs=specs,columns=['a_diff','b_sum'],
            models={s['name']:{} for s in specs})
        features=pd.DataFrame(dict(match_id=['b','a'],a_diff=[1,-1],b_sum=[2,3]))
        refs=pd.DataFrame(dict(match_id=['a','b'],**{n:[.3,.7] for n in [REFERENCE,R5_ACCURACY,R5_LOSS]}))
        def fake(bundle,x,**kwargs):return .5+.1*x.a_diff.to_numpy()
        with patch('scripts.train_neural_round9.predict_ensemble',fake):
            p=predict_neural(bundle,features,refs)
            assert_allclose(p[REFERENCE],[.7,.3])
            assert_allclose(p['realmlp_td_s_r4blend0.25'],[.675,.325])
            features['a_won']=[0,1];features['score']=['bad','bad']
            assert_allclose(p,predict_neural(bundle,features,refs))
            swapped=features.copy();swapped.a_diff*=-1
            srefs=refs.copy()
            for name in [REFERENCE,R5_ACCURACY,R5_LOSS]:srefs[name]=1-srefs[name]
            assert_allclose(p+predict_neural(bundle,swapped,srefs),1)
            with raises(ValueError,match='Missing'):
                predict_neural(bundle,features,refs.iloc[:1])



if __name__ == '__main__':unittest.main()

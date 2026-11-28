import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scripts.conditional_gate import EXPERTS,CONTEXTS,fit_global,fit_conditional
from scripts.train_gating_round10 import (REPORT_NAMES,REFERENCE,select_2021,gate_2022,
    fit_stage,predict_inputs,predict_stage)


def probabilities(y,correct):return np.where(np.asarray(correct)==np.asarray(y),.6,.4)


class GatingRunnerTests(unittest.TestCase):
    def test_selection_uses_only_2021_and_future_gate_does_not_reselect(self):
        y=np.arange(1000)%2;rows=pd.DataFrame({'season':2021,'a_won':y})
        ref=probabilities(y,np.arange(1000)<650)
        first=probabilities(y,np.arange(1000)<660)
        pred=pd.DataFrame(dict(equal_mix=ref,global_mix=first,conditional_mix=ref,round4_reference=ref))
        selected=select_2021(rows,pred)
        self.assertEqual(selected['choice'],'global_mix')
        later=rows.copy();later.season=2022
        pred.global_mix=ref;pred.conditional_mix=np.where(y,.9,.1)
        gate=gate_2022(selected['choice'],later,pred)
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['selected_procedure'],'global_mix')
        with self.assertRaisesRegex(ValueError,'2021'):select_2021(later,pred)
        early=rows.copy();early.season=2020
        with self.assertRaisesRegex(ValueError,'2021'):select_2021(early,pred)
        with self.assertRaisesRegex(ValueError,'row indices'):select_2021(rows,pred.iloc[::-1])
        with self.assertRaisesRegex(ValueError,'row indices'):gate_2022('global_mix',later,pred.iloc[::-1])

    def test_half_percentage_accuracy_gate_and_loss_guard(self):
        y=np.arange(1000)%2;rows=pd.DataFrame({'season':2022,'a_won':y})
        ref=probabilities(y,np.arange(1000)<650)
        for gain,expected in [(4,False),(5,True)]:
            p=probabilities(y,np.arange(1000)<650+gain)
            pred=pd.DataFrame({n:p if n=='conditional_mix' else ref for n in REPORT_NAMES})
            gate=gate_2022('conditional_mix',rows,pred)
            self.assertEqual(gate['passed'],expected);self.assertEqual(gate['net_correct'],gain)
        # More correct but severely overconfident errors must not pass.
        p=np.where(y,.999999,.000001);p[:100]=1-p[:100]
        pred=pd.DataFrame({n:p if n=='conditional_mix' else ref for n in REPORT_NAMES})
        self.assertFalse(gate_2022('conditional_mix',rows,pred)['passed'])
        with self.assertRaisesRegex(ValueError,'2022'):
            gate_2022('conditional_mix',rows.assign(season=2023),pred)

    def test_one_global_and_one_conditional_fit_per_stage_same_purged_population(self):
        ids=list('abcdef');rows=pd.DataFrame(dict(match_id=ids,season=[2020,2020,2021,2021,2022,2022],
            tourney_date=['2020-10-01','2020-12-25','2021-01-04','2021-11-01','2022-01-03','2022-10-31'],
            a_won=[0,1,0,1,0,1]),index=ids)
        p=pd.DataFrame(np.full((6,4),.5),index=ids,columns=EXPERTS)
        x=pd.DataFrame(np.zeros((6,5)),index=ids,columns=CONTEXTS)
        global_model=type('Global',(),{'weights':np.full(4,.25),'diagnostics':{}})()
        conditional=type('Conditional',(),{'diagnostics':{},'scaler':type('Scaler',(),{
            'mean':np.zeros(5),'scale':np.ones(5),'active':np.zeros(5,bool)})()})()
        with (patch('scripts.train_gating_round10.fit_global',return_value=global_model) as fg,
              patch('scripts.train_gating_round10.fit_conditional',return_value=conditional) as fc):
            model,metadata=fit_stage(2021,rows,p,x,rows[rows.season.eq(2021)])
        self.assertEqual(fg.call_count,1);self.assertEqual(fc.call_count,1)
        self.assertEqual(fg.call_args.args[0].index.tolist(),['a'])
        self.assertEqual(fc.call_args.args[0].index.tolist(),['a'])
        self.assertIs(fc.call_args.args[3],global_model)
        self.assertEqual(metadata['training_ids'],['a']);self.assertEqual(metadata['purged_rows'],1)
        with self.assertRaisesRegex(ValueError,'query year'):
            fit_stage(2022,rows,p,x,rows[rows.season.eq(2021)])

    def test_saved_inference_complements_and_ignores_outcome_columns(self):
        rng=np.random.default_rng(441);p=pd.DataFrame(rng.uniform(.1,.9,(60,4)),columns=EXPERTS)
        x=pd.DataFrame(rng.normal(size=(60,5)),columns=CONTEXTS);y=rng.integers(0,2,60)
        global_model=fit_global(p,y);conditional=fit_conditional(p,y,x,global_model)
        models={'global':global_model,'conditional':conditional};bundle={'models':{'2023':models}}
        inputs=pd.concat([pd.DataFrame(dict(match_id=[str(i) for i in range(60)],season=2023,tourney_date='2023-01-02')),p,x],axis=1)
        pred=predict_inputs(bundle,inputs)
        self.assertEqual(tuple(pred.columns),REPORT_NAMES)
        poison=inputs.copy();poison['a_won']=1-y;poison['score']='unused'
        assert_allclose(pred,predict_inputs(bundle,poison),atol=0,rtol=0)
        swapped=inputs.copy();swapped[list(EXPERTS)]=1-swapped[list(EXPERTS)]
        assert_allclose(pred+predict_inputs(bundle,swapped),1,atol=4e-16)
        buf=io.BytesIO();joblib.dump(bundle,buf);buf.seek(0)
        assert_allclose(pred,predict_inputs(joblib.load(buf),inputs),atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'year'):
            predict_inputs(bundle,inputs.assign(season=2024))
        with self.assertRaisesRegex(ValueError,'date'):
            predict_inputs(bundle,inputs.assign(tourney_date='2024-01-02'))


if __name__=='__main__':unittest.main()

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scripts.joint_point_model import candidate_specs, generate_raw_predictions
from scripts.train_joint_points_round8 import (REFERENCE, R5_ACCURACY, R5_LOSS, R2_COMPARATOR,
    add_blends, blend_registry, calibrate, calibrated_predictions, choose_procedure,
    predict_joint_points, select_candidate, selectable)
from tests.test_joint_point_model import synthetic_history, queries


class JointPointProtocolTests(unittest.TestCase):
    def test_menu_has_21_entries_and_excludes_diagnostics_and_old_comparators(self):
        specs = candidate_specs()
        y=np.array([0,1]*10)
        old=np.full(len(y),.6)
        predictions={REFERENCE:old,**{s['name']:old for s in specs}}
        predictions=add_blends(predictions,specs)
        for name in (R5_ACCURACY,R5_LOSS,R2_COMPARATOR,'raw_joint_recent'):
            predictions[name]=np.where(y,.99,.01)
        self.assertEqual(len(blend_registry(specs)),15)
        self.assertEqual(len(selectable(predictions,specs)),21)
        selected=select_candidate(y,predictions,np.ones(len(y),bool),specs)
        self.assertNotIn(selected['choice'],(R5_ACCURACY,R5_LOSS,R2_COMPARATOR,'raw_joint_recent'))

    def test_exact_accuracy_gate(self):
        specs=candidate_specs();seasons=np.repeat([2020,2021,2022],1000);y=np.tile(np.arange(1000)%2,3)
        correct=np.tile(np.arange(1000)<650,3)
        reference=np.where(correct==y,.6,.4)
        for improvement,passed in ((4,False),(5,True)):
            better=correct.copy()
            better[(seasons<2022)&(np.tile(np.arange(1000),3)>=650)&(np.tile(np.arange(1000),3)<660)]=True
            better[(seasons==2022)&(np.tile(np.arange(1000),3)>=650)&(np.tile(np.arange(1000),3)<650+improvement)]=True
            p=np.where(better==y,.6,.4)
            predictions=add_blends({REFERENCE:reference,**{s['name']:p for s in specs}},specs)
            gate=choose_procedure(y,predictions,seasons,specs)
            self.assertEqual(gate['procedure']=='adopt_joint_points',passed)
            self.assertAlmostEqual(gate['accuracy_improvement'],improvement/1000)

    def test_calibration_excludes_late_year_and_2023_outcomes(self):
        rows=pd.DataFrame(dict(match_id=list('abcdef'),season=[2017,2018,2020,2022,2022,2023],
            tourney_date=['2017-01-01','2018-01-01','2020-01-01','2022-11-01','2022-12-20','2023-01-02'],
            a_won=[0,1,1,0,0,1]))
        raw=pd.DataFrame({'joint_recent':[.25,.7,.6,.2,.99,np.nan]})
        slopes,meta=calibrate(raw,rows,2023)
        self.assertEqual(meta['ids'],list('abcd'))
        rows.loc[4:,'a_won']=1-rows.loc[4:,'a_won']
        altered,_=calibrate(raw,rows,2023)
        self.assertEqual(slopes,altered)
        prediction=calibrated_predictions(raw.iloc[:4],slopes)['joint_recent']
        np.testing.assert_array_equal(prediction>=.5,raw.iloc[:4,0].to_numpy()>=.5)
        rows.loc[0,'tourney_date']='2024-01-01'
        with self.assertRaises(ValueError):calibrate(raw,rows,2023)

    def test_reload_style_prediction_permutation_and_poisoning(self):
        history,_,_=synthetic_history()
        target=queries()
        specs=candidate_specs()
        raw,_,states=generate_raw_predictions(target,history,specs,keep_states=True)
        bundle=dict(states=states,specs=specs,slopes={s['name']:.8 for s in specs},reference_names=[REFERENCE])
        refs=pd.DataFrame({'match_id':target.match_id,REFERENCE:[.7,.5]})
        p=predict_joint_points(bundle,target,refs)
        expected=add_blends({REFERENCE:np.array([.7,.5]),**calibrated_predictions(raw,bundle['slopes'])},specs)
        for name,v in expected.items():assert_allclose(p[name],v,atol=0,rtol=0)
        target['a_won']=[1,0];target['score']=['poison','poison']
        q=predict_joint_points(bundle,target.iloc[::-1],refs.iloc[::-1])
        assert_allclose(p,q.iloc[::-1],atol=0,rtol=0)
        target.loc[0,'tourney_date']='2019-12-23'
        with self.assertRaises(ValueError):predict_joint_points(bundle,target,refs)


if __name__=='__main__':unittest.main()

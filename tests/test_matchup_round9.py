import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.matchup_features import MatchupClassifier,candidate_specs
from scripts.train_models import mirror_features
from scripts.train_matchup_round9 import predict_matchup,select_single,choose_procedure,REFERENCE,STUDENT
from tests.test_matchup_features import fixture


class MatchupProtocolTests(unittest.TestCase):
    def test_three_structures_share_main_effects_and_no_blend_menu(self):
        x,_=fixture();specs=candidate_specs(x)
        self.assertEqual([s['structure'] for s in specs],['no_new_interaction','shared','surface'])
        for s in specs:
            self.assertEqual(s['main_columns'],specs[0]['main_columns'])
            self.assertEqual(s['C'],.01);self.assertEqual(s['half_life'],3.)
            self.assertEqual(s['shared_factor'],.2);self.assertEqual(s['surface_factor'],.1)

    def test_bundle_frozen_order_and_report_only_student(self):
        x,rows=fixture();specs=candidate_specs(x)
        models={s['name']:MatchupClassifier(s).fit(x,rows) for s in specs}
        bundle=dict(feature_names=list(x),models=models,previous_bundle={},student_bundle={},
            previous_aliases={REFERENCE:'old'},student_name='teacher')
        def prior(bundle,features):
            # Fixture comparator has no role in fitting or constructing features.
            return pd.DataFrame({'old':np.full(len(features),.5),'teacher':np.full(len(features),.5)},index=features.index)
        with patch('scripts.train_matchup_round9.predict_round5',side_effect=prior),patch('scripts.train_matchup_round9.predict_distillation',side_effect=prior):
            p=predict_matchup(bundle,x)
            self.assertEqual(list(p),[s['name'] for s in specs]+[REFERENCE,STUDENT])
            pd.testing.assert_frame_equal(p,predict_matchup(bundle,x[x.columns[::-1]].assign(a_won='POISON',winner='POISON')))
            np.testing.assert_allclose(p+predict_matchup(bundle,mirror_features(x)),1,atol=2e-15,rtol=0)
            for invalid in (x.iloc[:0],x.drop(columns='style_ace_sum'),x.assign(style_ace_sum=np.nan)):
                with self.assertRaises(ValueError):predict_matchup(bundle,invalid)

    def test_guarded_early_selection_isolated_from_2022_and_2023(self):
        y=np.ones(600);seasons=np.repeat([2020,2021,2022],200)
        ref=np.full(600,.6);ref[::4]=.4
        a=ref.copy();a[:400:8]=.6;a[400:600:8]=.6
        b=ref.copy();b[400:]=.7
        p={REFERENCE:ref,'shared':a,'surface':b}
        gate=choose_procedure(y,p,seasons)
        self.assertEqual(gate['early_selection']['choice'],'shared')
        changed={k:v.copy() for k,v in p.items()};changed['shared'][400:]=.1
        after=choose_procedure(y,changed,seasons)
        self.assertEqual(after['early_selection'],gate['early_selection'])
        self.assertEqual(after['procedure'],'retain_round4')
        self.assertEqual(gate['procedure'],'adopt_accuracy_model')
        with self.assertRaises(ValueError):choose_procedure(np.r_[y,1],{k:np.r_[v,.9] for k,v in p.items()},np.r_[seasons,2023])


if __name__=='__main__':unittest.main()

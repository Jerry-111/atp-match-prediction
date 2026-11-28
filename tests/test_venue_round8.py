import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.train_models import mirror_features
from scripts.train_round5 import REFERENCE
from scripts.train_venue_round8 import candidate_specs, select_single, choose_procedure, predict_venue
from tests import test_round6_protocol as prior_tests


class VenueProtocolTests(unittest.TestCase):
    def test_ten_fixed_configs_isolate_surface_venue_and_strength(self):
        groups = dict(base=['log_rank_diff','elo_k32_diff','log_rank_mean'],
            context=['venue_raw_ace','venue_missing_serve'],surface=['surface_style_diff'],
            raw=['raw_style_diff'],adjusted=['adjusted_style_diff'],strength=['adjusted_rank_diff'])
        specs = candidate_specs(groups)
        self.assertEqual(len(specs),10)
        self.assertEqual(json.loads(json.dumps(specs)),specs)
        for spec in specs:
            group = spec['family']
            expected = set(groups['base'])
            if group != 'base': expected.update(groups['surface'])
            if group in ('raw','adjusted','strength'): expected.update(groups[group])
            self.assertTrue(set(spec['columns']) <= expected)
            self.assertFalse(set(spec['columns']) & set(groups['context']))
            self.assertEqual(spec['params']['C'],.01)
            self.assertEqual(spec.get('half_life',spec['params'].get('half_life')),3.)

    def test_selection_and_gate_never_consult_later_rows(self):
        y = np.tile([0,1],300)
        seasons = np.repeat([2020,2021,2022],200)
        correct = np.where(y==1,.65,.35)
        ref = correct.copy();ref[::4] = 1-ref[::4]
        a = ref.copy();a[0:400:8] = correct[0:400:8];a[400:600:8] = correct[400:600:8]
        b = ref.copy();b[400:] = correct[400:]
        p = {REFERENCE:ref,'a':a,'b':b}
        gate = choose_procedure(y,p,seasons)
        self.assertEqual(gate['early_selection']['choice'],'a')
        self.assertEqual(gate['procedure'],'adopt_accuracy_model')
        altered = {k:v.copy() for k,v in p.items()}
        altered['a'][400:] = 1-correct[400:]
        failed = choose_procedure(y,altered,seasons)
        self.assertEqual(failed['early_selection'],gate['early_selection'])
        self.assertEqual(failed['procedure'],'retain_round4')
        with self.assertRaises(ValueError):
            choose_procedure(np.r_[y,0],{k:np.r_[v,.5] for k,v in p.items()},np.r_[seasons,2023])

    def test_loss_guard_excludes_overconfident_higher_accuracy(self):
        y = np.tile([0,1],100)
        reference = np.where(y==1,.6,.4)
        reference[:50] = 1-reference[:50]
        bad = np.where(y==1,.999999,.000001);bad[:20] = 1-bad[:20]
        result = select_single(y,{REFERENCE:reference,'bad':bad},np.ones(len(y),bool))
        self.assertEqual(result['choice'],REFERENCE)
        self.assertEqual(result['eligible_models'],[REFERENCE])
        self.assertFalse(result['mixture_search'])

    def test_nested_reload_input_order_label_poison_and_symmetry(self):
        bundle,x = prior_tests.RoundSixProtocolTests().bundle()
        bundle['frozen']['pooled_selection']={'choice':'new_sixth'}
        expected = predict_venue(bundle,x)
        with tempfile.TemporaryDirectory() as folder:
            filename=Path(folder)/'frozen.joblib';joblib.dump(bundle,filename)
            actual=predict_venue(joblib.load(filename),x[x.columns[::-1]].assign(a_won='SECRET'))
        pd.testing.assert_frame_equal(expected,actual)
        np.testing.assert_allclose(expected+predict_venue(bundle,mirror_features(x)),1,atol=2e-15,rtol=0)
        self.assertNotIn('accuracy_grid_blend',expected)
        np.testing.assert_array_equal(expected[REFERENCE],
            predict_venue(bundle,x.assign(official_new_diff=100))[REFERENCE])

    def test_frozen_invalid_inputs_or_probability_fail_closed(self):
        bundle,x=prior_tests.RoundSixProtocolTests().bundle()
        for bad in (x.iloc[:0],x.drop(columns='official_new_diff'),x.assign(official_new_diff=np.inf),
                    pd.concat([x,x[['official_new_diff']]],axis=1)):
            with self.assertRaises(ValueError): predict_venue(bundle,bad)
        altered=copy.deepcopy(bundle);altered['models']['new_sixth']=prior_tests.InvalidProbability()
        with self.assertRaises(ValueError): predict_venue(altered,x)


if __name__=='__main__':
    unittest.main()

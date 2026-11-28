"""Round-six ablation isolation and frozen nested-inference boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.rank_trajectory_features import build_rank_trajectory_features
from scripts.train_models import mirror_features
from scripts.train_round5 import REFERENCE, predict_round5
from scripts.train_round6 import candidate_specs, predict_round6
from tests import test_round5_protocol as previous_tests


class InvalidProbability:
    def predict_proba(self, x):
        return np.full(len(x), np.nan)


class RoundSixProtocolTests(unittest.TestCase):
    def test_matched_information_controls_do_not_include_other_new_blocks(self):
        groups = dict(base=['log_rank_diff', 'elo_k32_diff', 'log_rank_mean'],
                      extended=['official_strength_diff', 'official_exposure_sum'],
                      trajectory=['trajectory_peak_diff', 'trajectory_unseen_sum'],
                      breakpoint=['bp_hold_diff', 'bp_exposure_sum'],
                      all_previous=['log_rank_diff', 'elo_k32_diff', 'log_rank_mean', 'old_paper_diff'])
        specs = candidate_specs(groups)
        self.assertEqual(len(specs), 34)
        self.assertEqual(len({s['name'] for s in specs}), 34)
        self.assertEqual(json.loads(json.dumps(specs)), specs)
        for group in ('base', 'extended', 'trajectory', 'breakpoint'):
            allowed = set(groups['base']) | (set(groups[group]) if group != 'base' else set())
            found = [s for s in specs if s['family'] in (group+'_linear', group+'_spline')]
            self.assertEqual(len(found), 6)
            for spec in found:
                self.assertTrue(set(spec['columns']) <= allowed)
                self.assertEqual(len(spec['columns']), len(set(spec['columns'])))
                if group != 'base':
                    self.assertIn(groups[group][0], spec['columns'])
        # Report-only earlier exploration is never an eligible candidate.
        self.assertFalse(any('previous_round5' in s['name'] or s['kind'] == 'previous' for s in specs))

    def bundle(self):
        prior, x = previous_tests.RoundFiveProtocolTests().bundle()
        prior['models']['all_inputs_spline_C0.01'] = previous_tests.ColumnProbability('new_score_diff')
        prior['previous_aliases']['rank_reference'] = 'r4_selected'
        x['official_new_diff'] = [-.4, .8, .2]
        bundle = dict(feature_names=list(x),
            models={'new_sixth': previous_tests.ColumnProbability('official_new_diff')},
            previous_bundle=prior,
            previous_aliases={REFERENCE: REFERENCE,
                'previous_round5_exploration': 'accuracy_grid_blend',
                'previous_round5_best_loss': 'all_inputs_spline_C0.01',
                'rank_reference': 'rank_reference'},
            frozen={'recommendation': REFERENCE, 'procedure_gate': {'procedure': 'retain_round4'},
                'pooled_selection': {'components': ['new_sixth', REFERENCE], 'weights': [.7, .3]}})
        return bundle, x

    def test_retained_reference_is_not_overwritten_by_new_blend_or_old_alias(self):
        bundle, x = self.bundle()
        before = predict_round5(bundle['previous_bundle'], x)
        out = predict_round6(bundle, x)
        np.testing.assert_array_equal(out[REFERENCE], before[REFERENCE])
        np.testing.assert_array_equal(out.previous_round5_exploration, before.accuracy_grid_blend)
        np.testing.assert_array_equal(out.previous_round5_best_loss, before['all_inputs_spline_C0.01'])
        np.testing.assert_allclose(out.accuracy_grid_blend,
            .7*expit(x.official_new_diff)+.3*before[REFERENCE], rtol=0, atol=1e-15)
        self.assertFalse(np.array_equal(out.accuracy_grid_blend, out.previous_round5_exploration))
        np.testing.assert_array_equal(out[bundle['frozen']['recommendation']], before[REFERENCE])

    def test_nested_joblib_reload_keeps_order_symmetry_and_ignores_label_fields(self):
        bundle, x = self.bundle()
        expected = predict_round6(bundle, x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frozen.joblib'
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            reordered = x[x.columns[::-1]].assign(a_won='POISON', winner_id='POISON', score='POISON')
            pd.testing.assert_frame_equal(expected, predict_round6(loaded, reordered))
            swapped = predict_round6(loaded, mirror_features(x))
            np.testing.assert_allclose(expected+swapped, 1., rtol=0, atol=2e-15)
        self.assertEqual(list(expected.index), [9, 4, 7])

    def test_invalid_required_features_and_frozen_models_fail_closed(self):
        bundle, x = self.bundle()
        for bad in [x.drop(columns='official_new_diff'), x.iloc[:0],
                    pd.concat([x, x[['official_new_diff']]], axis=1),
                    x.assign(official_new_diff=np.inf)]:
            with self.subTest(columns=list(bad)):
                with self.assertRaises(ValueError):
                    predict_round6(bundle, bad)
        altered = copy.deepcopy(bundle)
        altered['models']['new_sixth'] = InvalidProbability()
        with self.assertRaises(ValueError):
            predict_round6(altered, x)
        altered = copy.deepcopy(bundle)
        altered['frozen']['pooled_selection']['weights'] = [.7, -.3]
        with self.assertRaises(ValueError):
            predict_round6(altered, x)

    def rank_fixture(self):
        target = pd.DataFrame([dict(match_id='m', season=2023, tourney_date='2023-07-01',
            player_a_id='a', player_b_id='b', rank_a=15, rank_b=30)])
        history = pd.DataFrame([dict(season=2022, tourney_date='2022-12-12',
            player_a_id='a', player_b_id='b', rank_a=50, rank_b=20)])
        return target, history

    def test_rank_snapshot_cannot_hide_final_year_in_development_season(self):
        target, history = self.rank_fixture()
        bad = target.copy()
        bad['tourney_date'] = '2024-01-01'
        with self.assertRaises(ValueError):
            build_rank_trajectory_features(bad, history)
        # Even a future-dated history row must fail before its rank is used.
        bad_history = history.copy()
        bad_history['tourney_date'] = '2024-01-01'
        with self.assertRaises(ValueError):
            build_rank_trajectory_features(target, bad_history)

    def test_rank_date_validation_preserves_legitimate_december_season_start(self):
        target, history = self.rank_fixture()
        target['tourney_date'] = '2022-12-26'
        history['tourney_date'] = '2022-11-01'
        out = build_rank_trajectory_features(target, history)
        self.assertEqual(out.trajectory_unseen_sum.iloc[0], 0.)
        # Previous-year November is not a legitimate next-season date.
        target['tourney_date'] = '2022-11-01'
        with self.assertRaises(ValueError):
            build_rank_trajectory_features(target, history)


if __name__ == '__main__':
    unittest.main()

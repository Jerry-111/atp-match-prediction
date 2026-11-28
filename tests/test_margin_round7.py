"""Fine-outcome supervision is training-only and properly calibrated in time."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.margin_targets import build_performance_targets, point_share
from scripts.train_margin_round7 import (
    CalibratedPerformanceClassifier, REFERENCE, R5_COMPARATOR, add_blends,
    candidate_specs, choose_procedure, select_candidate, soft_targets,
)


class MarginTargetTest(unittest.TestCase):
    def fixture(self):
        rows = pd.DataFrame([dict(match_id='2022-T:1', season=2022, tourney_date='2022-06-01',
             surface='Hard', player_a_id='A', player_b_id='B', a_won=1)], index=[91])
        raw = dict(tourney_id='2022-T', match_num='1', tourney_date='20220601', surface='Hard',
            winner_id='A', loser_id='B', score='6-4 6-4', best_of='3',
            w_1stWon='40', w_2ndWon='20', w_svpt='100', l_1stWon='30', l_2ndWon='10', l_svpt='80')
        return rows, raw

    def test_hand_point_and_game_labels(self):
        rows, raw = self.fixture()
        self.assertAlmostEqual(point_share(raw, 'A'), 100/180)
        self.assertAlmostEqual(point_share(raw, 'B'), 80/180)
        out = build_performance_targets(rows, raw_rows=[(2022, raw)])
        self.assertAlmostEqual(out.point_share.iloc[0], 100/180)
        self.assertAlmostEqual(out.game_share.iloc[0], .6)
        self.assertEqual(out.index.tolist(), [91])
        swapped = rows.assign(player_a_id='B', player_b_id='A', a_won=0)
        reverse = build_performance_targets(swapped, raw_rows=[(2022, raw)])
        np.testing.assert_allclose(out[['point_share', 'game_share']]+reverse[['point_share', 'game_share']], 1.)

    def test_no_2023_target_fields_access_and_source_guards(self):
        rows, raw = self.fixture()
        a = build_performance_targets(rows, raw_rows=[(2022, raw)])
        b = build_performance_targets(rows, raw_rows=[(2022, raw), (2023, {'DO NOT ACCESS': True}), (2025, {})])
        pd.testing.assert_frame_equal(a, b)
        with self.assertRaisesRegex(ValueError, 'training-only'):
            build_performance_targets(rows.assign(season=2023), raw_rows=[])
        with self.assertRaises(ValueError):
            build_performance_targets(rows.assign(tourney_date='2023-01-01'), raw_rows=[])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_performance_targets(rows, raw_rows=[(2022, raw), (2022, raw)])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            build_performance_targets(rows.assign(a_won=0), raw_rows=[(2022, raw)])
        with self.assertRaisesRegex(ValueError, 'omitted'):
            build_performance_targets(rows, raw_rows=[])

    def test_file_manifest_excludes_2023_before_open(self):
        rows, raw = self.fixture()
        entries = [{'season': str(s)} for s in (2006, 2009, 2022, 2023, 2024)]
        def annual(cache_dir, selected):
            self.assertEqual([int(e['season']) for e in selected], [2009, 2022])
            return iter([(2022, raw)])
        with patch('scripts.margin_targets.development_entries', return_value=entries), \
             patch('scripts.margin_targets._annual_rows', side_effect=annual):
            build_performance_targets(rows)

    def test_missing_fallback_and_matched_smoothing(self):
        rows, raw = self.fixture()
        out = build_performance_targets(rows, raw_rows=[(2022, {**raw, 'w_svpt': ''})])
        self.assertEqual(out.point_available.iloc[0], 0)
        self.assertEqual(out.game_available.iloc[0], 1)
        for kind in ('binary', 'points', 'uniform_points'):
            self.assertEqual(soft_targets(rows.a_won, out, kind)[0], 1.)
        self.assertEqual(soft_targets(rows.a_won, out, 'uniform_games')[0], .75)
        self.assertAlmostEqual(soft_targets(rows.a_won, out, 'games')[0], .8)
        valid = pd.DataFrame(dict(point_share=[.4, .6], game_share=[.3, .7], point_available=[1, 1], game_available=[1, 1]))
        for kind in ('binary', 'points', 'games', 'uniform_points', 'uniform_games'):
            q = soft_targets([1, 0], valid, kind)
            reverse = valid.copy()
            reverse[['point_share', 'game_share']] = 1-valid[['point_share', 'game_share']]
            np.testing.assert_allclose(q+soft_targets([0, 1], reverse, kind), 1.)


class MarginLearnerTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(702)
        x = pd.DataFrame(rng.normal(size=(120, 3)), columns=['one_diff', 'two_diff', 'three_diff'])
        x['ignored_context'] = rng.normal(size=len(x))
        y = (x.one_diff+.5*rng.normal(size=len(x)) > 0).astype(int)
        rows = pd.DataFrame(dict(season=np.repeat([2017, 2018, 2019], 40),
            tourney_date=np.repeat(['2017-06-01', '2018-06-01', '2019-06-01'], 40), a_won=y))
        perf = pd.DataFrame(dict(point_share=np.clip(.5+.07*x.one_diff, 0, 1),
            game_share=np.clip(.5+.15*x.one_diff, 0, 1), point_available=np.ones(len(x)), game_available=np.ones(len(x))))
        return x, rows, perf

    def test_predeclared_registry_and_half_blends(self):
        specs = candidate_specs(self.fixture()[0].columns)
        self.assertEqual(len(specs), 10)
        self.assertEqual(len({s['name'] for s in specs}), 10)
        p = {REFERENCE: np.array([.2, .8]), **{s['name']: np.array([.6, .3]) for s in specs}}
        result = add_blends(p, specs)
        self.assertEqual(len(result), 21)
        for s in specs:
            np.testing.assert_array_equal(result[s['name']+'_r4blend0.5'], [.4, .55])

    def test_fractional_logistic_equals_hard_label_control(self):
        x, rows, perf = self.fixture()
        spec = candidate_specs(x.columns)[0]
        model = CalibratedPerformanceClassifier(spec)
        scale, fitted = model._fit_head(x, rows, perf)
        values = np.clip(x[spec['columns']].to_numpy()/scale, -8, 8)
        w = 2.**(-(rows.season.max()-rows.season.to_numpy())/3)
        w *= len(w)/w.sum()
        ordinary = LogisticRegression(C=spec['C'], fit_intercept=False, tol=1e-8, max_iter=2000).fit(values, rows.a_won, sample_weight=w)
        np.testing.assert_allclose(fitted.coef_, ordinary.coef_, atol=1e-10)

    def test_every_objective_swap_and_joblib(self):
        x, rows, perf = self.fixture()
        swapped = x.copy()
        odd = [n for n in x if n.endswith('_diff')]
        swapped[odd] *= -1
        for spec in candidate_specs(x.columns)[:5]:
            model = CalibratedPerformanceClassifier(spec).fit(x, rows, perf)
            p = model.predict_proba(x)
            np.testing.assert_allclose(model.predict_proba(swapped), 1-p, atol=2e-15)
            self.assertGreater(model.calibration_slope_, 0)
            self.assertEqual(model.calibration_metadata_['calibration_year'], 2019)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.joblib'
                joblib.dump(model, path)
                np.testing.assert_array_equal(joblib.load(path).predict_proba(x), p)

    def test_inner_calibration_honesty_training_scaler_and_cutoff(self):
        x, rows, perf = self.fixture()
        rows.loc[79, ['season', 'tourney_date']] = [2018, '2018-12-20']
        rows.loc[80:, 'tourney_date'] = '2019-01-01'
        spec = next(s for s in candidate_specs(x.columns) if s['kind'] == 'points')
        seen = []
        original = CalibratedPerformanceClassifier._fit_head
        def observed(self, xx, rr, pp):
            result = original(self, xx, rr, pp)
            seen.append((xx.index.tolist(), result[0]))
            return result
        with patch.object(CalibratedPerformanceClassifier, '_fit_head', observed):
            model = CalibratedPerformanceClassifier(spec).fit(x, rows, perf)
        self.assertEqual(len(seen), 2)
        self.assertNotIn(79, seen[0][0])
        self.assertEqual(seen[0][0], list(range(79)))
        self.assertEqual(seen[1][0], list(range(120)))
        self.assertEqual(model.calibration_metadata_['inner_training_cutoff'], '2018-12-11T00:00:00')
        # Changing later performance labels cannot affect the inner estimator.
        _, first = original(CalibratedPerformanceClassifier(spec), x.iloc[:79], rows.iloc[:79], perf.iloc[:79])
        changed = perf.copy();changed.loc[80:, 'point_share'] = 0
        _, second = original(CalibratedPerformanceClassifier(spec), x.iloc[:79], rows.iloc[:79], changed.iloc[:79])
        np.testing.assert_array_equal(first.coef_, second.coef_)

    def test_fit_seal_and_extra_inference_labels_ignored(self):
        x, rows, perf = self.fixture()
        spec = candidate_specs(x.columns)[0]
        for changed in (rows.assign(season=2023), rows.assign(tourney_date='2024-01-01')):
            with self.assertRaises(ValueError):
                CalibratedPerformanceClassifier(spec).fit(x, changed, perf)
        model = CalibratedPerformanceClassifier(spec).fit(x, rows, perf)
        np.testing.assert_array_equal(model.predict_proba(x), model.predict_proba(x.assign(point_share='SECRET', a_won='SECRET')))

    def test_early_selection_gate_and_r5_exclusion(self):
        y = np.ones(6);seasons = np.array([2020, 2020, 2021, 2021, 2022, 2022])
        p = {REFERENCE: np.full(6, .499), 'early': np.full(6, .501),
             'late': np.array([.498]*4+[.999]*2), R5_COMPARATOR: np.full(6, .999)}
        gate = choose_procedure(y, p, seasons)
        self.assertEqual(gate['early_selection']['choice'], 'early')
        self.assertEqual(gate['procedure'], 'adopt_margin')
        p['early'][-2:] = .498
        self.assertEqual(choose_procedure(y, p, seasons)['procedure'], 'retain_round4')
        with self.assertRaises(ValueError):
            choose_procedure(y, p, np.array([2020, 2020, 2021, 2021, 2023, 2023]))


if __name__ == '__main__':
    unittest.main()

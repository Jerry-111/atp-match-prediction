"""Synthetic runner boundaries; no empirical fitting or raw-data access."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from scripts import train_residual_state_round12 as runner


class FakeShared:
    """Pickleable prediction double; deliberately has no optimizer."""
    design_width_ = 256
    estimator_ = SimpleNamespace(n_iter_=np.array([1]))

    def predict_proba(self, x):
        return np.full(len(x), .6)


class FakeSnapshot:
    """A known state contrast for independently checking diagnostic arithmetic."""

    def __init__(self, effect, rows):
        self.effect = effect
        self.coefficients = np.array([effect/2, -effect/2])
        self.diagnostics = dict(success=True, iterations=0, rows=rows,
                                max_gradient_scaled=0., mode='synthetic_no_optimizer')

    def differences(self, query):
        return np.full(len(query), self.effect)


class FakeResidual:
    """Exercises main's serialization/freeze order without fitting any model."""
    output = None
    mutate_freeze = False
    calls = []

    def __init__(self, spec, structure):
        self.structure = structure
        probability = .65 if structure == 'static' else .7
        self.beta_ = np.array([-.1])
        self.effect = np.log(probability/(1-probability)) + .1
        self.state_snapshots_ = {}

    def fit(self, x, rows, shared_model):
        self.training_year = int(rows.season.max())
        self.fit_diagnostics_ = dict(success=True, iterations=0, rows=len(rows),
                                    max_gradient_scaled=0., mode='synthetic_no_optimizer')
        self.training_state_ = FakeSnapshot(self.effect, len(rows))
        self.__class__.calls.append(('fit', self.structure, len(rows)))
        return self

    def _design(self, x):
        return np.ones((len(x), 1))

    def predict_asof(self, x, query, history_x, history):
        if 'a_won' in query:
            raise AssertionError('Runner sent target labels to state API')
        if not x.index.equals(query.index):
            raise AssertionError('Runner reordered query identities')
        year = int(query.season.iloc[0])
        if year == 2023:
            out = self.__class__.output
            if not (out/'pre2023_freeze.json').exists():
                raise AssertionError('2023 prediction precedes selection freeze')
            old = pd.read_csv(out/'oof_predictions.csv', usecols=['season'])
            if len(old) != 6130 or set(old.season) != {2020, 2021, 2022}:
                raise AssertionError('Frozen OOF cohort is incomplete or contains 2023')
            if self.__class__.mutate_freeze:
                p = out/'frozen_selection.json'
                changed = json.loads(p.read_text())
                changed['recommendation'] = 'tampered_after_freeze'
                p.write_text(json.dumps(changed))
        cutoffs = pd.to_datetime(query.tourney_date)-pd.Timedelta(days=21)
        for cutoff in cutoffs.unique():
            key = pd.Timestamp(cutoff).strftime('%Y-%m-%d')
            eligible = pd.to_datetime(history.tourney_date).le(cutoff)
            self.state_snapshots_[key] = FakeSnapshot(self.effect, int(eligible.sum()))
        return self.predict_from_snapshots(x, query)

    def predict_from_snapshots(self, x, query):
        return np.full(len(x), .65 if self.structure == 'static' else .7)


def gate_example(net_correct=13):
    seasons = np.r_[[2020]*1214, [2021]*2388, [2022]*2528]
    y = np.ones(len(seasons), dtype=int)
    probabilities = {name: np.full(len(y), .6) for name in runner.CANDIDATES}
    early = seasons < 2022
    probabilities['shared_static_residual'][early] = .9
    held = np.flatnonzero(seasons == 2022)
    for values in probabilities.values():
        values[held[:1000]] = .4
    probabilities['shared_static_residual'][held[:net_correct]] = .6
    return y, probabilities, seasons


class SelectionTests(unittest.TestCase):
    def test_thirteen_integer_improvements_are_required(self):
        for gain, expected in [(12, 'retain_round4'), (13, 'adopt_accuracy_model')]:
            y, p, seasons = gate_example(gain)
            result = runner.choose_procedure(y, p, seasons)
            self.assertEqual(result['procedure'], expected)
            self.assertEqual(result['early_selection']['choice'], 'shared_static_residual')
            self.assertEqual(result['net_correct'], gain)
            self.assertEqual(result['required_net_correct'], 13)
            self.assertEqual(result['gate_rows'], 2528)

    def test_held_year_cannot_change_early_choice(self):
        y, p, seasons = gate_example(13)
        p['shared_dynamic_residual'][seasons == 2022] = .99
        result = runner.choose_procedure(y, p, seasons)
        self.assertEqual(result['early_selection']['choice'], 'shared_static_residual')
        self.assertEqual(result['net_correct'], 13)

    def test_gate_loss_guard_overrides_correct_count(self):
        y, p, seasons = gate_example(13)
        held = np.flatnonzero(seasons == 2022)
        p['shared_static_residual'][held[13:1000]] = .01
        result = runner.choose_procedure(y, p, seasons)
        self.assertEqual(result['net_correct'], 13)
        self.assertGreater(result['candidate_log_loss'], result['reference_log_loss']+.005)
        self.assertEqual(result['procedure'], 'retain_round4')

    def test_integer_correct_ties_use_loss_then_name(self):
        y = np.ones(11, dtype=int)
        p = {name: np.r_[np.full(8, .6), [.4]*3] for name in runner.CANDIDATES}
        p[runner.SHARED] = np.r_[np.full(9, .6), [.4]*2]
        p['shared_static_residual'] = np.r_[np.full(9, .7), [.4]*2]
        choice = runner.select_single(y, p, np.ones(len(y), bool))
        self.assertEqual(choice['choice'], 'shared_static_residual')
        self.assertEqual(choice['correct'], 9)
        p['shared_dynamic_residual'] = p['shared_static_residual'].copy()
        self.assertEqual(runner.select_single(y, p, np.ones(len(y), bool))['choice'],
                         'shared_dynamic_residual')
        p = {name: np.full(11, .6) for name in runner.CANDIDATES}
        self.assertEqual(runner.select_single(y, p, np.ones(11, bool))['choice'], min(runner.CANDIDATES))

    def test_mask_ignores_other_year_predictions_and_uses_fixed_half(self):
        y = np.array([1, 0, 1, 1])
        p = {name: np.array([.5, .49, .01, .01]) for name in runner.CANDIDATES}
        mask = np.array([True, True, False, False])
        before = runner.select_single(y, p, mask)
        self.assertEqual(before['correct'], 2)
        for values in p.values():
            values[~mask] = .99
        self.assertEqual(before, runner.select_single(y, p, mask))

    def test_selector_rejects_extra_models_bad_probabilities_and_seasons(self):
        y, p, seasons = gate_example()
        with self.assertRaises(ValueError):
            runner.select_single(y, dict(p, posthoc=np.full(len(y), .9)), np.ones(len(y), bool))
        for bad in [np.nan, np.inf, -.1, 1.1]:
            changed = {n: v.copy() for n, v in p.items()}
            changed[runner.SHARED][0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runner.choose_procedure(y, changed, seasons)
        with self.assertRaises(ValueError):
            runner.choose_procedure(y[:-1], {n: v[:-1] for n, v in p.items()}, seasons[:-1])
        modified = seasons.copy(); modified[-1] = 2023
        with self.assertRaises(ValueError):
            runner.choose_procedure(y, p, modified)


class ControlTests(unittest.TestCase):
    def test_control_requires_probability_and_class_reproduction(self):
        y = np.array([0, 1]); old = np.array([.2, .8])
        self.assertEqual(runner.check_control(y, old, old)['max_absolute_error'], 0.)
        with self.assertRaises(ValueError):
            runner.check_control(y, old+1e-8, old)
        # A tiny change at .5 must fail even when below the numeric tolerance.
        with self.assertRaises(ValueError):
            runner.check_control([1], [.5], [.5-1e-12])

    def test_nonfinite_prior_control_must_fail_closed(self):
        for y, new, old in [([0], [.2], [np.nan]), ([0], [np.nan], [.2]),
                            ([0], [.2], [np.inf]), ([0], [-.1], [-.1]),
                            ([0], [1.1], [1.1]), ([2], [.2], [.2]),
                            ([0, 1], [.2], [.2]), ([], [], [])]:
            with self.subTest(y=y, new=new, old=old), self.assertRaises(ValueError):
                runner.check_control(y, new, old)


class InferenceTests(unittest.TestCase):
    def query(self):
        x = pd.DataFrame({'z_diff': [.2, -.2]}, index=[7, 3])
        q = pd.DataFrame({'match_id': ['2023-x:1', '2023-x:2'], 'season': [2023]*2,
            'tourney_date': ['2023-03-01']*2, 'player_a_id': ['A', 'B'],
            'player_b_id': ['B', 'A']}, index=x.index)
        return x, q

    def bundle(self):
        return {'shared_model': FakeShared(), 'models': {
            name: FakeResidual({}, structure) for name, structure in runner.STRUCTURES.items()}}

    def test_query_label_rejected_before_any_predictor(self):
        x, q = self.query(); q['a_won'] = [1, 0]
        bundle = self.bundle(); bundle['shared_model'] = Mock()
        with self.assertRaises(ValueError):
            runner.predict_saved(bundle, x, q)
        bundle['shared_model'].predict_proba.assert_not_called()
        with self.assertRaises(ValueError):
            runner.predict_saved(self.bundle(), x, q.drop(columns=['a_won', 'season']))

    def test_saved_inference_has_no_optimizer_and_preserves_index(self):
        x, q = self.query(); bundle = self.bundle()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'bundle.joblib'; joblib.dump(bundle, file)
            with patch.object(FakeResidual, 'fit', side_effect=AssertionError('No fitting on inference')):
                output = runner.predict_saved(joblib.load(file), x, q)
        self.assertTrue(output.index.equals(x.index))
        self.assertEqual(set(output), {runner.SHARED, *runner.STRUCTURES})
        np.testing.assert_array_equal(output[runner.SHARED], [.6, .6])
        bundle['shared_model'] = SimpleNamespace(predict_proba=lambda x: np.full(len(x), np.nan))
        with self.assertRaises(ValueError):
            runner.predict_saved(bundle, x, q)


class RunnerLifecycleTests(unittest.TestCase):
    def synthetic_inputs(self):
        parts = []
        for year, n in [(2009, 2), (2020, 1214), (2021, 2388), (2022, 2528), (2023, 2588)]:
            parts.append(pd.DataFrame({'match_id': [f'{year}-synthetic:{i}' for i in range(n)],
                'season': year, 'tourney_date': f'{year}-06-01', 'player_a_id': 'A',
                'player_b_id': 'B', 'a_won': 1}))
        rows = pd.concat(parts, ignore_index=True)
        x = pd.DataFrame({'z_diff': np.ones(len(rows))}, index=rows.index)
        old = rows[['match_id', 'season', 'a_won']].copy()
        old[runner.REFERENCE] = .55; old[runner.BASE] = .6
        old['rank_reference'] = .52; old['accuracy_grid_blend'] = .58
        return rows, x, {}, old[old.season.isin(runner.FOLDS)].reset_index(drop=True), \
            old[old.season.eq(2023)].reset_index(drop=True), {}, 'synthetic-target-hash'

    def test_cli_requires_reviewed_hash_empty_output_and_query_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol = root/'protocol.md'; protocol.write_text('frozen synthetic protocol')
            output = root/'output'
            for extra in [[], ['--approved-protocol-sha256', 'wrong']]:
                with patch.object(runner, 'PROTOCOL', protocol), \
                     patch('sys.argv', ['runner', '--output', str(output), *extra]), \
                     patch.object(runner, 'load_inputs') as load, redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit):
                    runner.main()
                load.assert_not_called(); self.assertFalse(output.exists())
            output.mkdir(); (output/'preserve.txt').write_text('keep')
            args = ['runner', '--output', str(output), '--approved-protocol-sha256', runner.sha256(protocol)]
            with patch.object(runner, 'PROTOCOL', protocol), patch('sys.argv', args), \
                 patch.object(runner, 'load_inputs') as load, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                runner.main()
            load.assert_not_called(); self.assertEqual((output/'preserve.txt').read_text(), 'keep')
            with patch('sys.argv', ['runner', '--predict-features', 'unopened.csv']), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()

    def run_synthetic_lifecycle(self, output, protocol, *, mutate=False):
        FakeResidual.output = output; FakeResidual.mutate_freeze = mutate; FakeResidual.calls = []
        inputs = self.synthetic_inputs()
        fit_years = iter([2020, 2021, 2022, 2023])
        def fake_control(spec, x, rows):
            year = next(fit_years)
            self.assertTrue(rows.season.lt(year).all())
            self.assertEqual((output/'pre2023_freeze.json').exists(), year == 2023)
            return FakeShared()
        args = ['runner', '--output', str(output), '--approved-protocol-sha256', runner.sha256(protocol)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'PROTOCOL', protocol))
            stack.enter_context(patch('sys.argv', args))
            stack.enter_context(patch.object(runner, 'load_inputs', return_value=inputs))
            stack.enter_context(patch.object(runner, 'fit_shared_control', side_effect=fake_control))
            stack.enter_context(patch.object(runner, 'ResidualStateModel', FakeResidual))
            stack.enter_context(patch.object(runner, 'accuracy_interval', return_value={'synthetic': True}))
            stack.enter_context(patch.object(runner, 'cluster_interval', return_value={'synthetic': True}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner.main()

    def test_all_oof_rows_freeze_before_2023_and_exact_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'run'; protocol = root/'p.md'; protocol.write_text('synthetic only')
            self.run_synthetic_lifecycle(output, protocol)
            oof = pd.read_csv(output/'oof_predictions.csv')
            self.assertEqual(oof.groupby('season').size().to_dict(), {2020:1214, 2021:2388, 2022:2528})
            self.assertEqual(len(pd.read_csv(output/'validation_predictions.csv')), 2588)
            freeze = json.loads((output/'pre2023_freeze.json').read_text())
            self.assertEqual(freeze['frozen_selection_sha256'], runner.sha256(output/'frozen_selection.json'))
            self.assertEqual(freeze['oof_predictions_sha256'], runner.sha256(output/'oof_predictions.csv'))
            bundle = joblib.load(output/'frozen_models.joblib')
            self.assertEqual(bundle['frozen_selection_sha256'], freeze['frozen_selection_sha256'])
            self.assertNotIn('a_won', bundle['query_metadata'])
            self.assertEqual(len(FakeResidual.calls), 8)
            self.assertEqual(bundle['frozen']['recommendation'], runner.REFERENCE)
            self.assertFalse(bundle['registry']['final_test_evaluated'])
            diagnostics = pd.read_csv(output/'optimizer_diagnostics.csv')
            self.assertEqual(diagnostics.groupby('stage').size().to_dict(),
                             {'global': 8, 'state_only': 8})
            self.assertTrue(diagnostics.success.all())
            self.assertTrue(diagnostics.iterations.eq(0).all())
            contributions = pd.read_csv(output/'residual_contributions.csv')
            self.assertEqual(len(contributions), 8)
            sizes = {2020:1214, 2021:2388, 2022:2528, 2023:2588}
            for record in contributions.itertuples():
                probability = .65 if record.model == 'shared_static_residual' else .7
                effect = np.log(probability/(1-probability))+.1
                self.assertAlmostEqual(record.mean_absolute_state_logit, effect)
                self.assertAlmostEqual(record.rms_state_logit, effect)
                self.assertAlmostEqual(record.max_absolute_state_logit, effect)
                self.assertAlmostEqual(record.global_state_rms, effect/2)
                self.assertEqual(record.classifications_changed_by_state, sizes[record.year])
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['state_only_solves'], 8)
            manifest = json.loads((output/'run_manifest.json').read_text())['artifact_sha256']
            for name in ('optimizer_diagnostics.csv', 'residual_contributions.csv'):
                self.assertEqual(manifest[name], runner.sha256(output/name))

    def test_tampering_with_frozen_selection_stops_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'run'; protocol = root/'p.md'; protocol.write_text('synthetic only')
            with self.assertRaisesRegex(ValueError, 'frozen choice changed'):
                self.run_synthetic_lifecycle(output, protocol, mutate=True)
            self.assertFalse((output/'validation_predictions.csv').exists())
            self.assertFalse((output/'frozen_models.joblib').exists())
        FakeResidual.mutate_freeze = False


if __name__ == '__main__':
    unittest.main()

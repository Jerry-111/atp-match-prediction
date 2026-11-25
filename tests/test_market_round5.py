"""Independent market-procedure tests using synthetic outcomes, never 2023 scores."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts import train_market_round5 as runner
from scripts.market_data import build_market_features
from tests import test_market_data


class MarketRoundFiveTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(457)
        n = 80
        market = rng.normal(size=n)
        history = expit(.6*market+rng.normal(size=n)*.4)
        outcomes = (market+rng.normal(size=n) > 0).astype(int)
        available = (np.arange(n) % 7 != 0).astype(float)
        features = pd.DataFrame({
            'market_open_logit_diff': market,
            'market_open_available': available,
            'market_open_overround': .03+np.arange(n)*.0001,
            'market_close_logit_diff': market*.8,
            'market_close_available': (np.arange(n) % 5 != 0).astype(float),
            'market_close_overround': .025,
            'market_movement_logit_diff': -.2*market,
            'market_both_available': available*(np.arange(n) % 5 != 0),
        }, index=100+np.arange(n)*2)
        return history, features, outcomes

    def test_twenty_registered_candidates_never_select_closing(self):
        specs = runner.candidate_specs()
        self.assertEqual(len(specs), 20)
        self.assertEqual(len({s['name'] for s in specs}), 20)
        self.assertEqual(json.loads(json.dumps(specs)), specs)
        self.assertNotIn('closing', json.dumps(specs))
        self.assertEqual({s['weight'] for s in specs if s['kind']=='blend'}, {.25, .5, .75})
        self.assertEqual(sum(s['kind']=='calibrated' for s in specs), 12)

    def test_2021h2_and_2022_date_purge_is_inclusive(self):
        rows = pd.DataFrame({
            'season': [2020, 2021, 2021, 2021, 2021, 2021, 2021, 2022, 2022, 2023],
            'tourney_date': ['2020-05-01', '2021-01-04', '2021-06-10', '2021-06-11',
                             '2021-07-01', '2021-12-06', '2021-12-07',
                             '2021-12-27', '2022-02-01', '2023-02-01'],
        }, index=np.arange(10)*3+2)
        train, valid = runner.market_masks(rows, '2021H2')
        self.assertEqual(np.flatnonzero(train).tolist(), [1, 2])
        self.assertEqual(np.flatnonzero(valid).tolist(), [4, 5, 6])
        train, valid = runner.market_masks(rows, '2022')
        self.assertEqual(np.flatnonzero(train).tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(np.flatnonzero(valid).tolist(), [7, 8])
        self.assertFalse((train & valid).any())
        with self.assertRaises(ValueError):
            runner.market_masks(rows, '2023')

    def test_selection_mask_loss_guard_tiebreak_and_extraneous_closing(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        specs = [dict(name=runner.REFERENCE, family='history'), dict(name='a', family='market'),
                 dict(name='b', family='market')]
        prediction = {runner.REFERENCE: np.array([.2, .8, .2, .8, .99, .01]),
                      'a': np.array([.1, .9, .1, .9, .99, .01]),
                      'b': np.array([.1, .9, .1, .9, .99, .01]),
                      'closing_market_descriptive': np.array([.001, .999, .001, .999, .001, .999])}
        mask = np.array([True]*4+[False]*2)
        chosen, families = runner.select_candidate(y, prediction, specs, mask)
        self.assertEqual(chosen, 'a')
        self.assertEqual(families['market'], 'a')
        y[~mask] = 1-y[~mask]
        for p in prediction.values():p[~mask] = 1-p[~mask]
        self.assertEqual((chosen, families), runner.select_candidate(y, prediction, specs, mask))
        y = np.array([0, 1, 0, 1])
        prediction = {runner.REFERENCE: np.array([.6, .4, .1, .9]), 'a': np.array([.49, .51, .49, .51])}
        self.assertGreater(runner.accuracy(y, prediction['a']), runner.accuracy(y, prediction[runner.REFERENCE]))
        chosen, _ = runner.select_candidate(y, prediction, specs[:2], np.ones(4, bool))
        self.assertEqual(chosen, runner.REFERENCE)  # accuracy alone is insufficient

    def test_gate_uses_2021_selection_and_2022_accuracy_and_loss(self):
        y = np.tile([0, 1], 200)
        labels = np.repeat(['2021H2', '2022'], 200)
        correct = np.tile(np.r_[np.ones(150, bool), np.zeros(50, bool)], 2)
        reference = np.where(correct, np.where(y==1, .6, .4), np.where(y==1, .4, .6))
        specs = [dict(name=runner.REFERENCE, family='history'), dict(name='new', family='new')]
        for mode, expected in [('pass', 'adopt_market_information'),
                               ('no_gain', 'retain_history'), ('bad_loss', 'retain_history')]:
            candidate = reference.copy()
            fixes = np.arange(150, 160)
            candidate[fixes] = np.where(y[fixes]==1, .6, .4)
            if mode != 'no_gain':
                fixes = np.arange(350, 352)
                candidate[fixes] = np.where(y[fixes]==1, .6, .4)
            if mode == 'bad_loss':
                candidate[352:] = np.where(y[352:]==1, .001, .999)
            prediction = {runner.REFERENCE: reference, 'new': candidate}
            gate = runner.market_gate(y, prediction, specs, labels)
            self.assertEqual(gate['early_selected'], 'new')
            self.assertEqual(gate['procedure'], expected)
            # Extra later outcomes and even perfect closing predictions cannot
            # become either the earlier selector or the 2022 adoption check.
            extended = {n: np.r_[p, .999, .001] for n, p in prediction.items()}
            extended['closing_market_descriptive'] = np.where(np.r_[y, 1, 0], .999, .001)
            self.assertEqual(gate, runner.market_gate(np.r_[y, 1, 0], extended, specs,
                                                     np.concatenate([labels, ['2023', '2024']])))

    def test_all_kinds_preserve_exact_missing_fallback_and_complement(self):
        history, features, y = self.fixture()
        missing = features.market_open_available.eq(0).to_numpy()
        reverse = features.copy()
        odd = [c for c in reverse if c.endswith('_diff')]
        reverse[odd] *= -1
        for spec in runner.candidate_specs():
            with self.subTest(name=spec['name']):
                model = runner.MarketModel(spec).fit(history, features, y)
                p = model.predict_proba(history, features)
                np.testing.assert_array_equal(p[missing], history[missing])
                np.testing.assert_allclose(p+model.predict_proba(1-history, reverse), 1., atol=3e-15, rtol=0)
                self.assertTrue(np.all((p>0)&(p<1)))
        self.assertEqual(runner.accuracy([1, 0], [.5, .49999]), 1.)

    def test_calibration_ignores_missing_labels_closing_and_source_results(self):
        history, features, y = self.fixture()
        missing = features.market_open_available.eq(0).to_numpy()
        for design in ('market', 'fusion', 'context'):
            spec = next(s for s in runner.candidate_specs() if s.get('design')==design)
            first = runner.MarketModel(spec).fit(history, features, y)
            altered_y = y.copy(); altered_y[missing] = 1-altered_y[missing]
            poisoned = features.assign(market_close_logit_diff='CLOSING SECRET',
                                       market_movement_logit_diff='CLOSING SECRET',
                                       vainqueur_id='SOURCE OUTCOME', score='SOURCE SCORE')
            second = runner.MarketModel(spec).fit(history, poisoned, altered_y)
            np.testing.assert_array_equal(first.predict_proba(history, features),
                                          second.predict_proba(history, poisoned))
            self.assertEqual(first.available_training_rows_, (~missing).sum())
        targets, source = test_market_data.MarketDataTests().fixture()
        plain, _ = build_market_features(targets, market_rows=source)
        poisoned, _ = build_market_features(targets.assign(a_won='IGNORED'),
            market_rows=source.assign(vainqueur_id='IGNORED', score='IGNORED', duree_min='IGNORED'))
        opening = runner.MarketModel(runner.candidate_specs()[1])
        np.testing.assert_array_equal(opening.predict_proba(np.full(4, .6), plain),
                                      opening.predict_proba(np.full(4, .6), poisoned))

    def test_nested_joblib_reload_and_descriptive_closing_fallback(self):
        history, features, y = self.fixture()
        models = {s['name']: runner.MarketModel(s).fit(history, features, y) for s in runner.candidate_specs()}
        bundle = {'models': models, 'base_model': 'frozen_R4', 'frozen': {'decision_threshold': .5}}
        expected = runner.predict_market_bundle(bundle, history, features)
        missing = features.market_close_available.eq(0).to_numpy()
        np.testing.assert_array_equal(expected.closing_market_descriptive.to_numpy()[missing], history[missing])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'market.joblib'
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            reordered = features[list(reversed(features.columns))].assign(score='POISON')
            pd.testing.assert_frame_equal(expected, runner.predict_market_bundle(loaded, history, reordered))
        self.assertTrue(expected.index.equals(features.index))
        reverse = features.copy()
        reverse[[c for c in features if c.endswith('_diff')]] *= -1
        np.testing.assert_allclose(expected+runner.predict_market_bundle(bundle, 1-history, reverse), 1., atol=3e-15, rtol=0)

    def test_invalid_prices_and_unusable_calibration_cannot_silently_fit(self):
        history, features, y = self.fixture()
        spec = next(s for s in runner.candidate_specs() if s['kind']=='calibrated')
        with self.assertRaisesRegex(ValueError, 'Too few'):
            runner.MarketModel(spec).fit(history, features.assign(market_open_available=0), y)
        with self.assertRaises(ValueError):
            runner.MarketModel(spec).fit(history, features, np.full(len(y), 2))
        with self.assertRaises(ValueError):
            runner.MarketModel(spec).fit(history, features.assign(market_open_available=.5), y)
        with self.assertRaises(ValueError):
            runner.MarketModel(spec).fit(np.ones(len(y)), features, y)

    def test_historical_prior_reads_audited_oof_not_fitted_training_predictions(self):
        rows = pd.DataFrame({'match_id': ['older', 'twenty', 'twentyone', 'twentytwo', 'later'],
                             'season': [2019, 2020, 2021, 2022, 2023],
                             'a_won': [0, 1, 0, 1, 0]}, index=[99, 7, 42, 3, 18])
        current = 'frozen_age_model'
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            oof = rows.iloc[1:4].reset_index(drop=True).assign(**{current: [.21, .32, .43]})
            val = rows.iloc[4:].reset_index(drop=True).assign(**{current: [.74]})
            oof.to_csv(directory/'oof_predictions.csv', index=False)
            val.to_csv(directory/'validation_predictions.csv', index=False)
            # A tempting in-sample file deliberately disagrees. The loader must
            # neither read it nor load/refit a model for the 2021/2022 inputs.
            rows.assign(**{current: .999}).to_csv(directory/'training_predictions.csv', index=False)
            (directory/'frozen_selection.json').write_text(json.dumps({'recommendation': current}))
            def audit():
                names = ['oof_predictions.csv', 'validation_predictions.csv', 'frozen_selection.json']
                (directory/'artifact_audit.json').write_text(json.dumps({
                    'artifact_sha256': {n: runner.sha256(directory/n) for n in names}}))
            audit()
            history, name = runner.load_history_predictions(rows, directory)
            self.assertEqual(name, current)
            self.assertTrue(history.index.equals(rows.index))
            self.assertTrue(np.isnan(history.loc[99]))
            np.testing.assert_array_equal(history.loc[[7, 42, 3, 18]], [.21, .32, .43, .74])
            val.iloc[0, val.columns.get_loc(current)] = .999
            val.to_csv(directory/'validation_predictions.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Audited'):
                runner.load_history_predictions(rows, directory)
            audit()
            oof.iloc[::-1].to_csv(directory/'oof_predictions.csv', index=False)
            audit()
            with self.assertRaisesRegex(ValueError, 'alignment'):
                runner.load_history_predictions(rows, directory)
            oof.iloc[:2].to_csv(directory/'oof_predictions.csv', index=False)
            audit()
            with self.assertRaisesRegex(ValueError, 'omit required'):
                runner.load_history_predictions(rows, directory)


if __name__ == '__main__':
    unittest.main()

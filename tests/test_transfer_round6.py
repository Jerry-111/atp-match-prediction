"""Temporal, matched-data and selection invariants for supervised transfer."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.train_transfer_round6 import (
    REFERENCE, R5_COMPARATOR, SPLINE_CORE, TransferClassifier, add_blends,
    blend_registry, build_transfer_features, build_transfer_targets,
    candidate_specs, choose_procedure, fold_masks, select_candidate, transfer_weights, verify_tour_targets,
)


def record(identifier, season=2020, day='2020-01-01', source='tour', won=1):
    return dict(match_id=identifier, season=season, tourney_date=day, source=source,
                player_a_id='A', player_b_id='B', rank_a=10, rank_b=30,
                surface='Hard', a_won=won)


class TransferTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(158)
        self.x = pd.DataFrame(rng.normal(size=(40, 4)), columns=SPLINE_CORE)
        self.x['context_mean'] = rng.normal(size=40)
        self.rows = pd.DataFrame([record(str(i), season=2019, day='2019-06-01',
            source='tour' if i < 20 else 'challenger', won=i % 2) for i in range(40)])

    def test_registry_and_fixed_convex_blends(self):
        specs = candidate_specs(self.x.columns)
        blends = blend_registry(specs)
        self.assertEqual(len(specs), 12)
        self.assertEqual(len(blends), 36)
        self.assertEqual(len(set(s['name'] for s in specs)), 12)
        p = {REFERENCE: np.array([.2, .7]), **{s['name']: np.array([.6, .1]) for s in specs}}
        combined = add_blends(p, blends)
        self.assertEqual(len(combined), 49)
        for name, setting in blends.items():
            np.testing.assert_allclose(combined[name], setting['model_weight'] * np.array([.6, .1])
                                       + (1-setting['model_weight']) * p[REFERENCE])
        with self.assertRaises(ValueError):
            add_blends(p, {'bad': {'model': specs[0]['name'], 'model_weight': 1.1}})

    def test_sealed_auxiliary_targets_and_preserved_tour(self):
        tour = pd.DataFrame([record('tour-original', season=2023, day='2023-05-01')])
        inventory = pd.DataFrame([record('challenger:x', source='challenger'),
            record('qualifying:x', source='qualifying'),
            record('challenger:2023', season=2023, day='2023-05-01', source='challenger', won='POISON'),
            record('challenger:2024', season=2024, day='BAD', source='challenger', won='POISON')])
        result = build_transfer_targets(tour, inventory)
        self.assertEqual(list(result.match_id), ['tour-original', 'challenger:x', 'qualifying:x'])
        self.assertEqual(result.loc[0, 'a_won'], tour.loc[0, 'a_won'])
        self.assertEqual(result.loc[0, 'tourney_date'], '2023-05-01')
        with self.assertRaises(ValueError):
            build_transfer_targets(tour.assign(season=2024), inventory)

    def test_feature_cutoff_poisoning_and_swap(self):
        target = pd.DataFrame([record('query', day='2020-02-01')])
        inventory = pd.DataFrame([record('tour:old', day='2020-01-11'),
                                 record('challenger:future', day='2020-01-12', source='challenger')])
        x = build_transfer_features(target, inventory)
        self.assertEqual(x.shape, (1, 48))
        self.assertGreater(x.ml_common_overall_diff.iloc[0], 0)
        changed = inventory.astype({'a_won': object}).copy()
        changed.loc[1, 'a_won'] = 'POISON'
        changed = pd.concat([changed, pd.DataFrame([record('tour:2024', season=2024, day='BAD', won='POISON')])])
        pd.testing.assert_frame_equal(x, build_transfer_features(target.assign(a_won='POISON'), changed))
        swapped = target.copy()
        swapped[['player_a_id', 'player_b_id']] = target[['player_b_id', 'player_a_id']].to_numpy()
        swapped[['rank_a', 'rank_b']] = target[['rank_b', 'rank_a']].to_numpy()
        sx = build_transfer_features(swapped, inventory)
        sign = np.array([-1 if c.endswith('_diff') else 1 for c in x])
        np.testing.assert_allclose(sx.to_numpy(), x.to_numpy() * sign, atol=1e-12)

    def test_weight_mass_and_zero_auxiliary_exact_control(self):
        for weight in (0., .1, .3, 1.):
            w = transfer_weights(self.rows, weight)
            self.assertAlmostEqual(w.sum(), 20)
        for kind in ('linear', 'spline'):
            spec = next(s for s in candidate_specs(self.x.columns) if s['kind'] == kind and s['auxiliary_weight'] == 0)
            mixed = self.x.copy()
            mixed.loc[20:] = np.nan
            model = TransferClassifier(spec).fit(mixed, self.rows)
            control = TransferClassifier(spec).fit(self.x.iloc[:20], self.rows.iloc[:20])
            np.testing.assert_array_equal(model.odd_scale_, control.odd_scale_)
            np.testing.assert_array_equal(model.predict_proba(self.x), control.predict_proba(self.x))

    def test_odd_model_and_serialization(self):
        swap = self.x.copy()
        swap[list(SPLINE_CORE)] *= -1
        for kind in ('linear', 'spline'):
            spec = next(s for s in candidate_specs(self.x.columns) if s['kind'] == kind and s['auxiliary_weight'] == .3)
            model = TransferClassifier(spec).fit(self.x, self.rows)
            p = model.predict_proba(self.x)
            np.testing.assert_allclose(model.predict_proba(swap), 1-p, atol=2e-15)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.joblib'
                joblib.dump(model, path)
                np.testing.assert_array_equal(joblib.load(path).predict_proba(self.x), p)

    def test_fit_seal_and_label_validation(self):
        spec = candidate_specs(self.x.columns)[1]
        with self.assertRaises(ValueError):
            TransferClassifier(spec).fit(self.x, self.rows.assign(season=2023))
        with self.assertRaises(ValueError):
            TransferClassifier(spec).fit(self.x, self.rows.assign(tourney_date='2024-01-01'))
        with self.assertRaises(ValueError):
            TransferClassifier(spec).fit(self.x, self.rows.assign(a_won=.2))

    def test_training_purge_and_no_current_year_auxiliary(self):
        rows = pd.DataFrame([record('old', season=2020, day='2020-12-11', source='challenger'),
            record('purged', season=2020, day='2020-12-12', source='qualifying'),
            record('valid', season=2021, day='2021-01-01'),
            record('current-aux', season=2021, day='2021-01-01', source='challenger')])
        train, valid = fold_masks(rows, 2021)
        self.assertEqual(list(rows.loc[train, 'match_id']), ['old'])
        self.assertEqual(list(rows.loc[valid, 'match_id']), ['valid'])

    def test_early_choice_gate_and_excluded_pooled_comparator(self):
        y = np.ones(6)
        seasons = np.array([2020, 2020, 2021, 2021, 2022, 2022])
        p = {REFERENCE: np.full(6, .499), 'early': np.full(6, .501),
             'late': np.array([.498]*4 + [.999]*2), R5_COMPARATOR: np.full(6, .999)}
        gate = choose_procedure(y, p, seasons)
        self.assertEqual(gate['early_selection']['choice'], 'early')
        self.assertEqual(gate['procedure'], 'adopt_transfer')
        p['early'][-2:] = .498
        gate = choose_procedure(y, p, seasons)
        self.assertEqual(gate['early_selection']['choice'], 'early')
        self.assertEqual(gate['procedure'], 'retain_round4')
        with self.assertRaises(ValueError):
            choose_procedure(y, p, np.array([2020, 2020, 2021, 2021, 2023, 2023]))

    def test_loss_guard_overrides_higher_accuracy(self):
        y = np.array([1, 1, 1, 0])
        p = {REFERENCE: np.array([.55, .55, .49, .1]),
             'high_acc_bad_loss': np.array([.51, .51, .51, .49])}
        selection = select_candidate(y, p, np.ones(4, dtype=bool))
        self.assertEqual(selection['choice'], REFERENCE)

    def test_cached_target_metadata_identity(self):
        tour = pd.DataFrame([record('original')])
        inventory = pd.DataFrame([record('tour:original')])
        verify_tour_targets(tour, inventory)
        for column, value in [('rank_a', 11), ('rank_b', 31), ('season', 2021),
                              ('tourney_date', '2020-01-02'), ('a_won', 0)]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                verify_tour_targets(tour.assign(**{column: value}), inventory)


if __name__ == '__main__':
    unittest.main()

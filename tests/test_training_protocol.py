"""Independent invariants for chronological probability-model selection."""

import unittest

import numpy as np
import pandas as pd

from scripts.train_models import (
    SymmetricClassifier,
    cluster_interval,
    fit_blend,
    mirror_features,
    temporal_masks,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_forward_fold_purges_recent_prior_season_labels(self):
        rows = pd.DataFrame({
            "season": [2018, 2019, 2019, 2020, 2020, 2021],
            "tourney_date": ["2018-08-01", "2019-12-16", "2019-12-17",
                             "2020-01-06", "2020-06-01", "2021-01-04"],
        }, index=[2, 5, 8, 11, 14, 17])
        train, valid = temporal_masks(rows, 2020)
        # The earliest target is Jan 6. Dec 16 is exactly T-21 and available;
        # Dec 17 belongs to an earlier season but its label is still embargoed.
        self.assertEqual(list(rows.index[train]), [2, 5])
        self.assertEqual(list(rows.index[valid]), [11, 14])
        self.assertFalse((train & valid).any())

    def test_forward_fold_requires_both_past_and_validation(self):
        rows = pd.DataFrame({"season": [2020], "tourney_date": ["2020-01-06"]})
        with self.assertRaisesRegex(ValueError, "No earlier training"):
            temporal_masks(rows, 2020)
        with self.assertRaisesRegex(ValueError, "No rows"):
            temporal_masks(rows, 2021)

    @staticmethod
    def fitted_model():
        x = pd.DataFrame({
            "strength_diff": [-2., -1.5, -1., -.5, .5, 1., 1.5, 2.],
            "context": [2., 7., 3., 8., 5., 9., 4., 10.],
        })
        spec = {"columns": list(x), "kind": "logistic", "params": {"C": .5}}
        model = SymmetricClassifier(spec).fit(x, np.array([0, 0, 1, 0, 1, 0, 1, 1]))
        return model, x

    def test_swapping_players_complements_probability_with_context(self):
        model, _ = self.fitted_model()
        x = pd.DataFrame({"strength_diff": [-10., -1., 0., 1.5, 10.],
                          "context": [100., -100., 15., 0., 90.]})
        original = x.copy(deep=True)
        p = model.predict_proba(x)
        q = model.predict_proba(mirror_features(x))
        np.testing.assert_allclose(p + q, 1., atol=1e-14)
        self.assertAlmostEqual(p[2], .5, places=14)
        self.assertTrue(np.all((p > 0) & (p < 1)))
        pd.testing.assert_frame_equal(x, original)

    def test_validation_values_do_not_refit_scaling(self):
        model, x = self.fitted_model()
        scaler = model.estimator.named_steps["standardscaler"]
        before = scaler.mean_.copy()
        # Even context is unchanged by exchanging players; its fitted mean
        # must therefore equal the training context mean, not a heldout mean.
        self.assertAlmostEqual(before[1], x.context.mean())
        model.predict_proba(pd.DataFrame({"strength_diff": [1e5], "context": [1e9]}))
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_unselected_target_outcomes_cannot_enter_predictions(self):
        model, x = self.fitted_model()
        clean = model.predict_proba(x)
        poisoned = x.assign(a_won=np.arange(len(x)) * 999, player_a_id="secret")
        np.testing.assert_array_equal(model.predict_proba(poisoned), clean)

    def test_convex_blend_does_not_reward_reversed_probabilities(self):
        y = np.array([0., 1., 0., 1.])
        good = np.array([.1, .9, .2, .8])
        weights = fit_blend(y, [good, 1 - good, np.full(4, .5)])
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertGreater(weights[0], .999)

    def test_paired_date_bootstrap_is_zero_for_identical_predictions(self):
        rows = pd.DataFrame({"a_won": [0, 1, 0, 1],
                             "tourney_date": ["2023-01-02", "2023-01-02",
                                              "2023-01-09", "2023-01-09"]})
        p = np.array([.1, .8, .2, .7])
        result = cluster_interval(rows, p, p, repeats=100)
        self.assertEqual(result["date_blocks"], 2)
        self.assertEqual(result["mean_log_loss_reduction"], 0.)
        self.assertEqual(result["ci95"], [0., 0.])

    def test_paired_date_bootstrap_positive_when_every_match_improves(self):
        rows = pd.DataFrame({"a_won": [0, 1, 0, 1],
                             "tourney_date": ["2023-01-02", "2023-01-02",
                                              "2023-01-09", "2023-01-09"]})
        result = cluster_interval(rows, np.full(4, .5),
                                  np.array([.1, .9, .2, .8]), repeats=100)
        self.assertGreater(result["ci95"][0], 0.)


if __name__ == "__main__":
    unittest.main()

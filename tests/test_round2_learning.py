"""Structural safeguards for the round-two odd spline learning family."""

import copy
import json
import unittest

import numpy as np
import pandas as pd

from scripts.round2_learning import candidate_specs, make_model
from scripts.train_models import mirror_features


class RoundTwoLearningTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(158)
        self.x = pd.DataFrame({
            "log_rank_diff": rng.normal(size=180),
            "elo_k32_diff": rng.normal(size=180) * 150,
            "surface_is_clay": rng.binomial(1, .3, 180),
            "surface_is_grass": rng.binomial(1, .1, 180),
            "log_rank_mean": rng.uniform(1, 5, 180),
            "constant_diff": np.zeros(180),
        })
        self.y = (rng.random(180) < 1 / (1 + np.exp(-self.x.log_rank_diff))).astype(int)
        self.seasons = np.tile([2018, 2019, 2020], 60)
        self.specs = candidate_specs(self.x.columns)

    def test_registry_is_bounded_serializable_and_ignores_labels(self):
        self.assertEqual(len(self.specs), 6)
        self.assertEqual(len({s["name"] for s in self.specs}), 6)
        self.assertEqual(json.loads(json.dumps(self.specs)), self.specs)
        self.assertEqual(candidate_specs([*self.x.columns, "a_won", "player_a_id"]), self.specs)

    def test_exact_exchange_symmetry_with_context_and_constant_columns(self):
        model = make_model(self.specs[2]).fit(self.x, self.y, self.seasons)
        p = model.predict_proba(self.x)
        q = model.predict_proba(mirror_features(self.x))
        np.testing.assert_allclose(p + q, 1., atol=1e-14)
        neutral = self.x.iloc[:2].copy()
        neutral.loc[:, [c for c in neutral if c.endswith("_diff")]] = 0
        np.testing.assert_allclose(model.predict_proba(neutral), .5, atol=1e-14)

    def test_player_orientation_in_training_does_not_change_fit(self):
        first = make_model(self.specs[2]).fit(self.x, self.y, self.seasons)
        second = make_model(self.specs[2]).fit(mirror_features(self.x), 1 - self.y, self.seasons)
        np.testing.assert_allclose(first.predict_proba(self.x), second.predict_proba(self.x), atol=1e-10)

    def test_prediction_cannot_refit_transform_or_inspect_outcomes(self):
        model = make_model(self.specs[2]).fit(self.x, self.y, self.seasons)
        before = (model.odd_scale_.copy(), model.spline_scale_.copy(), copy.deepcopy(model.context_mean_))
        p = model.predict_proba(self.x)
        extra = self.x.assign(a_won="POISON", season=2025)
        np.testing.assert_array_equal(p, model.predict_proba(extra))
        extreme = self.x.iloc[:4].copy()
        extreme["log_rank_diff"] = 1e9
        extreme["log_rank_mean"] = -1e8
        self.assertTrue(np.isfinite(model.predict_proba(extreme)).all())
        np.testing.assert_array_equal(model.odd_scale_, before[0])
        np.testing.assert_array_equal(model.spline_scale_, before[1])
        self.assertEqual(model.context_mean_, before[2])

    def test_recency_model_requires_aligned_training_seasons(self):
        model = make_model(self.specs[-1])
        with self.assertRaisesRegex(ValueError, "requires training seasons"):
            model.fit(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "aligned"):
            model.fit(self.x, self.y, self.seasons[:10])
        model.fit(self.x, self.y, self.seasons)
        self.assertTrue(np.isfinite(model.predict_proba(self.x)).all())

    def test_missing_or_nonfinite_required_features_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            make_model(self.specs[0]).fit(self.x.drop(columns="elo_k32_diff"), self.y)
        x = self.x.copy()
        x.loc[0, "elo_k32_diff"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            make_model(self.specs[0]).fit(x, self.y)


if __name__ == "__main__":
    unittest.main()

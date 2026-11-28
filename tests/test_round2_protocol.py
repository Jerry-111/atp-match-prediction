"""Independent checks of the second-round procedure gate and frozen inference."""

import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.train_round2 import choose_procedure, predict_round2
from scripts.train_models import mirror_features


class FixedSymmetricModel:
    def predict_proba(self, x):
        return expit(x.strength_diff.to_numpy())


class RoundTwoProtocolTest(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1, 0, 1])
        self.seasons = np.repeat([2020, 2021, 2022], 2)
        self.old = np.where(self.y == 1, .6, .4)

    def test_upgrade_requires_declared_minimum_gain(self):
        specs = [{"name": "new", "family": "only"}]
        for improvement, expected in [(.0005, "retain_round1"), (.002, "new_single")]:
            correct = .6 * np.exp(improvement)
            p = np.where(self.y == 1, correct, 1 - correct)
            result = choose_procedure(self.y, {"new": p}, specs, self.seasons, self.old)
            self.assertEqual(result["procedure"], expected)
            self.assertAlmostEqual(result["reference_2022_loss"] - result["new_single_2022_loss"], improvement)

    def test_bad_2022_performance_retains_first_round(self):
        specs = [{"name": "new", "family": "only"}]
        p = np.where(self.y == 1, .9, .1)
        p[-2:] = [.8, .2]
        result = choose_procedure(self.y, {"new": p}, specs, self.seasons, self.old)
        self.assertEqual(result["procedure"], "retain_round1")

    def test_family_choice_does_not_use_2022_outcomes(self):
        specs = [{"name": "early_best", "family": "shared"}, {"name": "late_best", "family": "shared"}]
        early = np.where(self.y == 1, .9, .1)
        early[-2:] = [.8, .2]
        late = np.where(self.y == 1, .55, .45)
        late[-2:] = [.01, .99]
        result = choose_procedure(self.y, {"early_best": early, "late_best": late}, specs, self.seasons, self.old)
        self.assertEqual(result["early_selected"], {"shared": "early_best"})
        self.assertEqual(result["procedure"], "retain_round1")

    def test_adding_2023_poison_cannot_change_gate(self):
        specs = [{"name": "first", "family": "shared"}, {"name": "second", "family": "shared"}]
        oof = {"first": np.where(self.y == 1, .8, .2), "second": np.where(self.y == 1, .55, .45)}
        before = choose_procedure(self.y, oof, specs, self.seasons, self.old)
        after = choose_procedure(np.r_[self.y, [1, 0]],
            {"first": np.r_[oof["first"], [0., 1.]], "second": np.r_[oof["second"], [1., 0.]]},
            specs, np.r_[self.seasons, [2023, 2023]], np.r_[self.old, [0., 1.]])
        self.assertEqual(before, after)

    def test_complementary_families_can_upgrade_via_blend(self):
        specs = [{"name": "one", "family": "one"}, {"name": "two", "family": "two"}]
        oof = {"one": np.tile([.05, .55], 3), "two": np.tile([.45, .95], 3)}
        result = choose_procedure(self.y, oof, specs, self.seasons, self.old)
        self.assertEqual(result["procedure"], "new_blend")
        self.assertLess(result["new_blend_2022_loss"], result["new_single_2022_loss"])

    @staticmethod
    def bundle():
        return {"feature_names": ["strength_diff", "elo_k32_diff"],
                "models": {"learned": FixedSymmetricModel()},
                "raw_columns": {"raw": "elo_k32_diff"},
                "blends": {"blend": {"components": ["learned", "raw"], "weights": [.4, .6]}}}

    def test_frozen_round2_inference_is_symmetric_and_label_independent(self):
        x = pd.DataFrame({"elo_k32_diff": [-100., 30., 140.], "strength_diff": [-1., 0., 2.]}, index=[8, 2, 5])
        actual = predict_round2(self.bundle(), x)
        poisoned = predict_round2(self.bundle(), x.assign(a_won="POISON"))
        pd.testing.assert_frame_equal(actual, poisoned)
        self.assertEqual(list(actual.index), [8, 2, 5])
        np.testing.assert_allclose(actual + predict_round2(self.bundle(), mirror_features(x)), 1., atol=1e-14)
        self.assertTrue(np.all((actual.to_numpy() > 0) & (actual.to_numpy() < 1)))

    def test_frozen_inference_rejects_missing_and_nonfinite_features(self):
        with self.assertRaisesRegex(ValueError, "Missing frozen"):
            predict_round2(self.bundle(), pd.DataFrame({"strength_diff": [1.]}))
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_round2(self.bundle(), pd.DataFrame({"strength_diff": [np.nan], "elo_k32_diff": [1.]}))


if __name__ == "__main__":
    unittest.main()

"""Check the reference Glicko calculation and temporal feature guarantees."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.dynamic_features import (
    DynamicHistory, DynamicRatings, build_dynamic_features, inflated_rd, update_glicko,
)


class DynamicFeaturesTest(unittest.TestCase):
    day = date(2023, 7, 3)

    def match(self, number, days_ago, a="A", b="B", won=1, surface="Hard"):
        played = self.day - timedelta(days=days_ago)
        return {"match_id": f"tour:T:{number}", "season": played.year, "source": "tour",
                "tourney_date": played.strftime("%Y%m%d"), "surface": surface,
                "player_a_id": a, "player_b_id": b, "a_won": won}

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.day.isoformat(),
                              "player_a_id": "A", "player_b_id": "B", "surface": "Hard", "a_won": 1}], index=[17])

    def test_reference_glickman_worked_example(self):
        # Original Glicko PDF pages 4--5, not Glicko-2's volatility example.
        rating, rd = update_glicko(1500, 200, [(1400, 30, 1), (1550, 100, 0), (1700, 300, 0)])
        self.assertAlmostEqual(rating, 1464.1, delta=.1)
        self.assertAlmostEqual(rd, 151.4, delta=.1)

    def test_uncertainty_grows_with_inactivity_without_changing_skill(self):
        history = DynamicHistory()
        history.append(self.day, 1700, 60)
        short = history.at(self.day + timedelta(days=7))
        long = history.at(self.day + timedelta(days=364))
        self.assertEqual(short[0], long[0])
        self.assertGreater(long[1], short[1])
        self.assertEqual(inflated_rd(350, 10000), 350)
        with self.assertRaisesRegex(ValueError, "backward"):
            inflated_rd(60, -1)
        # Posterior after the cutoff must not enter even when forecast is later.
        history.append(self.day + timedelta(days=10), 1000, 30)
        self.assertEqual(history.at(self.day, self.day + timedelta(days=20))[0], 1700)

    def test_same_date_batch_and_input_order_invariance(self):
        inv = pd.DataFrame([self.match(1, 21), self.match(2, 21, b="C")])
        state = DynamicRatings(inv)
        cutoff = self.day - timedelta(days=21)
        expected = update_glicko(1500, 350, [(1500, 350, 1), (1500, 350, 1)])
        self.assertEqual(state.overall["A"].at(cutoff), expected)
        self.assertEqual(state.overall["A"].at(cutoff-timedelta(days=1)), (1500, 350))
        pd.testing.assert_frame_equal(build_dynamic_features(self.targets(), inv),
                                      build_dynamic_features(self.targets(), inv.iloc[::-1]))

    def test_future_target_and_holdout_poisoning_cannot_change_features(self):
        inv = pd.DataFrame([self.match(1, 21), self.match(2, 20, won=0), self.match(3, 0, won=0)])
        expected = build_dynamic_features(self.targets(), inv)
        inv.loc[[1, 2], "a_won"] = 1
        inv = pd.concat([inv, pd.DataFrame([
            {"source": "tour", "season": 2024, "a_won": "SEALED"},
            {"source": "tour", "season": 2025, "a_won": "SEALED"},
            {"source": "challenger", "season": 2023, "a_won": "IGNORED"},
        ])], ignore_index=True)
        targets = self.targets()
        targets["a_won"] = "NOT AN INPUT"
        pd.testing.assert_frame_equal(expected, build_dynamic_features(targets, inv))

    def test_exact_player_exchange_and_no_surface_history_fallback(self):
        inv = pd.DataFrame([self.match(1, 100), self.match(2, 90, b="C"), self.match(3, 21, a="B", b="C", won=0)])
        targets = self.targets()
        targets["surface"] = "Grass"
        expected = build_dynamic_features(targets, inv)
        self.assertEqual(expected.loc[17, "glicko_surface_weight_sum"], 0)
        self.assertEqual(expected.loc[17, "pooled_glicko_rating_diff"], expected.loc[17, "glicko_rating_diff"])
        targets[["player_a_id", "player_b_id"]] = targets[["player_b_id", "player_a_id"]].to_numpy()
        odd = [c for c in expected if c.endswith("_diff")]
        expected[odd] = -expected[odd]
        pd.testing.assert_frame_equal(expected, build_dynamic_features(targets, inv))

    def test_unseen_players_finite_and_locked_targets(self):
        inv = pd.DataFrame([self.match(1, 21)])
        targets = self.targets()
        targets[["player_a_id", "player_b_id"]] = [["X", "Y"]]
        result = build_dynamic_features(targets, inv)
        self.assertEqual(result.shape, (1, 17))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(result.loc[17, "glicko_adjusted_diff"], 0)
        self.assertEqual(result.loc[17, "glicko_rd_sum"], 700)
        targets["season"] = 2024
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            build_dynamic_features(targets, inv)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            DynamicRatings(pd.concat([inv, inv]))


if __name__ == "__main__":
    unittest.main()

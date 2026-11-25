"""Cross-level ratings: numerical controls, temporal isolation and exchangeability."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.multilevel_features import (
    RatingBank, build_multilevel_features, elo_expectation,
    historical_records, surface_blend,
)


class MultilevelFeaturesTest(unittest.TestCase):
    day = date(2023, 7, 3)

    def match(self, number, ago=21, a="A", b="B", won=1, source="challenger", surface="Hard"):
        played = self.day - timedelta(days=ago)
        return {"match_id": f"{source}:T:{number}", "season": played.year,
                "source": source, "tourney_date": played.isoformat(),
                "player_a_id": a, "player_b_id": b, "a_won": won, "surface": surface}

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.day.isoformat(),
                              "player_a_id": "A", "player_b_id": "B", "surface": "Hard",
                              "a_won": "UNUSED"}], index=[41])

    def test_hand_calculation_common_and_level_priors(self):
        self.assertAlmostEqual(elo_expectation(1900, 1500), 10/11)
        inv = pd.DataFrame([self.match(1)])
        f = build_multilevel_features(self.targets(), inv).iloc[0]
        self.assertEqual(f.ml_common_overall_diff, 32)
        self.assertEqual(f.ml_level_overall_diff, 32)
        self.assertEqual(f.ml_tour_overall_diff, 0)
        self.assertEqual(f.ml_common_overall_mean, 1500)
        self.assertEqual(f.ml_level_overall_mean, 1300)
        self.assertEqual(f.ml_level_trend90_diff, 0)
        self.assertAlmostEqual(f.ml_challenger_recent364_log_count_sum, 2*np.log(2))
        self.assertEqual(surface_blend(1600, 1400, 20), (1500, .5))

    def test_same_date_prestate_and_first_level_tie(self):
        inv = pd.DataFrame([self.match(1), self.match(2, b="C", source="tour")])
        bank = RatingBank(historical_records(inv), level_aware=True)
        cutoff = self.day - timedelta(days=21)
        # A's first observed day includes Tour, so 1500; B only Challenger, 1300.
        expected = 1500 + 32*(1-elo_expectation(1500, 1300)) + 16
        self.assertAlmostEqual(bank.overall["A"].at(cutoff)[0], expected)
        self.assertEqual(bank.overall["A"].at(cutoff)[1], 2)
        self.assertEqual(bank.overall["A"].at(cutoff-timedelta(days=1)), (1500, 0))
        pd.testing.assert_frame_equal(build_multilevel_features(self.targets(), inv),
                                      build_multilevel_features(self.targets(), inv.iloc[::-1]))

    def test_later_tour_appearance_does_not_change_initial_level(self):
        inv = pd.DataFrame([self.match(1, ago=100), self.match(2, ago=20, b="C", source="tour")])
        bank = RatingBank(historical_records(inv), level_aware=True)
        self.assertEqual(bank.overall["A"].at(self.day-timedelta(days=21)), (1316, 1))
        # An early target queried alongside a later target still cannot use the
        # latter target's roster, first-level information, outcomes or counts.
        targets = self.targets()
        later = targets.copy()
        later["tourney_date"] = (self.day+timedelta(days=70)).isoformat()
        both = pd.concat([targets, later], ignore_index=True)
        first = build_multilevel_features(both, inv).iloc[0]
        mutated = inv.copy()
        mutated.loc[1, ["player_b_id", "a_won"]] = ["NEW_FUTURE_PLAYER", 0]
        np.testing.assert_array_equal(first, build_multilevel_features(both, mutated).iloc[0])

    def test_inclusive_cutoff_and_sealed_labels(self):
        inv = pd.DataFrame([self.match(1, ago=21), self.match(2, ago=20, won="FUTURE")])
        expected = build_multilevel_features(self.targets(), inv.iloc[:1])
        inv = pd.concat([inv, pd.DataFrame([
            {"season": 2024, "source": "tour", "a_won": "SEALED"},
            {"season": 2025, "source": "challenger", "a_won": "SEALED"},
            {"season": 2024, "source": "qualifying", "a_won": "SEALED"},
        ])], ignore_index=True)
        pd.testing.assert_frame_equal(expected, build_multilevel_features(self.targets(), inv))

    def test_player_exchange_and_history_orientation(self):
        inv = pd.DataFrame([self.match(1, ago=100), self.match(2, ago=40, a="B", b="C", source="tour"),
                            self.match(3, ago=21, b="C", source="qualifying", surface="Clay")])
        target = self.targets()
        result = build_multilevel_features(target, inv)
        swapped = target.copy()
        swapped[["player_a_id", "player_b_id"]] = swapped[["player_b_id", "player_a_id"]].to_numpy()
        actual = build_multilevel_features(swapped, inv)
        parity = np.array([-1 if c.endswith("_diff") else 1 for c in result])
        np.testing.assert_allclose(actual, result.to_numpy()*parity, atol=1e-12, rtol=0)
        reverse = inv.copy()
        reverse[["player_a_id", "player_b_id"]] = reverse[["player_b_id", "player_a_id"]].to_numpy()
        reverse["a_won"] = 1-reverse.a_won
        np.testing.assert_allclose(result, build_multilevel_features(target, reverse), atol=1e-12, rtol=0)

    def test_lower_results_propagate_to_tour_opponents(self):
        inv = pd.DataFrame([self.match(1, ago=100), self.match(2, ago=21, a="A", b="C", source="tour")])
        altered = inv.copy()
        altered.loc[0, "a_won"] = 0
        target = self.targets()
        target[["player_a_id", "player_b_id"]] = [["C", "D"]]
        first, second = [build_multilevel_features(target, h).iloc[0] for h in (inv, altered)]
        self.assertEqual(first.ml_tour_overall_diff, second.ml_tour_overall_diff)
        self.assertNotEqual(first.ml_common_overall_diff, second.ml_common_overall_diff)

    def test_no_surface_history_and_unseen_priors(self):
        inv = pd.DataFrame([self.match(1)])
        target = self.targets()
        target["surface"] = "Grass"
        f = build_multilevel_features(target, inv)
        self.assertEqual(f.shape, (1, 37))
        self.assertEqual(f.index.tolist(), [41])
        for name in ("tour", "common", "level"):
            self.assertEqual(f.iloc[0][f"ml_{name}_blend_diff"], f.iloc[0][f"ml_{name}_overall_diff"])
        target[["player_a_id", "player_b_id"]] = [["X", "Y"]]
        unseen = build_multilevel_features(target, inv).iloc[0]
        self.assertEqual(unseen.ml_level_overall_mean, 1500)
        self.assertEqual(unseen.ml_unseen_sum, 2)
        self.assertTrue(np.isfinite(unseen).all())

    def test_invalid_inputs_and_final_targets_rejected(self):
        inv = pd.DataFrame([self.match(1)])
        for year in (2008, 2024, 2025):
            target = self.targets()
            target["season"] = year
            with self.assertRaisesRegex(ValueError, "holdout is locked"):
                build_multilevel_features(target, inv)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_multilevel_features(self.targets(), pd.concat([inv, inv]))
        broken = inv.copy()
        broken["player_a_id"] = None
        with self.assertRaisesRegex(ValueError, "Missing"):
            build_multilevel_features(self.targets(), broken)
        broken = inv.copy()
        broken["a_won"] = 2
        with self.assertRaisesRegex(ValueError, "Invalid"):
            build_multilevel_features(self.targets(), broken)


if __name__ == "__main__":
    unittest.main()

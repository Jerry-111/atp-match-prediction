"""Style rates use valid opportunities and historical attributes, never target stats."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.style_features import build_style_features, technical_pair


class StyleFeaturesTest(unittest.TestCase):
    day = date(2023, 7, 3)

    def match(self, number=1, ago=21):
        day = self.day-timedelta(days=ago)
        inv = {"match_id": f"tour:T:{number}", "season": day.year, "source": "tour",
               "tourney_date": day.isoformat(), "surface": "Hard", "player_a_id": "A",
               "player_b_id": "B", "a_won": "LABEL NOT USED"}
        raw = {"tourney_id": "T", "match_num": number, "tourney_date": day.isoformat(),
               "surface": "Hard", "winner_id": "A", "loser_id": "B",
               "winner_ht": "195", "loser_ht": "180", "winner_hand": "L", "loser_hand": "R",
               "w_ace": "10", "w_df": "4", "w_svpt": "100", "w_1stIn": "60", "w_1stWon": "45", "w_2ndWon": "20",
               "l_ace": "2", "l_df": "3", "l_svpt": "80", "l_1stIn": "50", "l_1stWon": "30", "l_2ndWon": "12"}
        return inv, (day.year, raw)

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.day.isoformat(),
                              "player_a_id": "A", "player_b_id": "B", "a_won": "UNUSED"}], index=[91])

    def build(self, matches, target=None):
        inventory, raw = zip(*matches)
        return build_style_features(self.targets() if target is None else target,
                                    pd.DataFrame(inventory), raw_rows=raw)

    def test_hand_opportunity_calculation(self):
        f = self.build([self.match()]).iloc[0]
        self.assertAlmostEqual(f.style_ace_diff, 16/200-8/180)
        self.assertAlmostEqual(f.style_double_fault_diff, 12/140-11/130)
        self.assertAlmostEqual(f.style_first_in_diff, 120/200-110/180)
        self.assertAlmostEqual(f.style_first_won_diff, 115/160-100/150)
        self.assertAlmostEqual(f.style_second_won_diff, 70/140-62/130)
        self.assertAlmostEqual(f.style_return_first_won_diff, 50/150-45/160)
        self.assertAlmostEqual(f.style_return_second_won_diff, 68/130-70/140)
        self.assertEqual(f.style_height_diff, 15)
        self.assertEqual(f.style_left_diff, 1)
        self.assertEqual(f.style_mixed_hands, 1)

    def test_inclusive_window_and_target_future_poison(self):
        matches = [self.match(1, 21), self.match(2, 364), self.match(3, 20), self.match(4, 0)]
        expected = self.build(matches)
        for _, (_, raw) in matches[2:]:
            raw.update(w_svpt="TARGET SECRET", winner_ht="230", winner_hand="R")
        pd.testing.assert_frame_equal(expected, self.build(matches))
        old = self.match(5, 365)
        self.assertAlmostEqual(expected.iloc[0].style_ace_diff, 26/300-10/260)
        # Old technical counts fall outside the window; newer attrs take priority.
        pd.testing.assert_frame_equal(expected, self.build([old, *matches]), check_flags=False)

    def test_earlier_target_unchanged_when_later_queries_added(self):
        matches = [self.match(1, 21), self.match(2, 0)]
        t = self.targets()
        later = t.copy()
        later["tourney_date"] = (self.day+timedelta(days=70)).isoformat()
        both = pd.concat([t, later], ignore_index=True)
        original = self.build(matches, both)
        matches[1][1][1].update(w_ace="30", winner_ht="220", winner_hand="R")
        updated = self.build(matches, both)
        np.testing.assert_array_equal(original.iloc[0], updated.iloc[0])
        self.assertNotEqual(original.iloc[1].style_height_diff, updated.iloc[1].style_height_diff)

    def test_invalid_pair_rejected_and_attribute_conflict_safe(self):
        first = self.match(1, 22)
        second = self.match(2, 21)
        third = self.match(3, 21)
        second[1][1]["winner_ht"] = "200"
        third[1][1]["winner_ht"] = "210"
        f = self.build([first, second, third])
        self.assertEqual(f.iloc[0].style_height_diff, 15)  # earlier known height
        self.assertEqual(f.attrs["conflicting_same_day_attributes"], 1)
        pd.testing.assert_frame_equal(f, self.build([third, first, second]))
        for key, value in (("w_ace", "90"), ("l_1stIn", "1000"), ("w_df", "30"), ("w_svpt", "NaN")):
            match = self.match()
            match[1][1][key] = value
            self.assertIsNone(technical_pair(match[1][1]))
            row = self.build([match]).iloc[0]
            self.assertEqual(row.style_ace_diff, 0)
            self.assertAlmostEqual(row.style_missing_log_sum, 2*np.log(2))

    def test_swap_symmetry_and_missing_neutral_values(self):
        f = self.build([self.match()])
        target = self.targets()
        target[["player_a_id", "player_b_id"]] = [["B", "A"]]
        swapped = self.build([self.match()], target)
        parity = np.array([-1 if c.endswith("_diff") else 1 for c in f])
        np.testing.assert_allclose(swapped, f.to_numpy()*parity, atol=1e-12, rtol=0)
        target[["player_a_id", "player_b_id"]] = [["X", "Y"]]
        unseen = self.build([self.match()], target)
        self.assertEqual(unseen.shape, (1, 30))
        self.assertEqual(unseen.index.tolist(), [91])
        self.assertTrue(np.isfinite(unseen.to_numpy()).all())
        self.assertEqual(unseen.iloc[0].style_height_mean, 185)
        self.assertEqual(unseen.iloc[0].style_hand_missing_sum, 2)
        self.assertEqual(unseen.iloc[0].style_height_missing_sum, 2)

    def test_locked_seasons_labels_and_identity_checks(self):
        inv, raw = self.match()
        inventory = pd.concat([pd.DataFrame([inv]), pd.DataFrame([{"source": "tour", "season": 2024}])])
        base = self.build([(inv, raw)])
        pd.testing.assert_frame_equal(base, build_style_features(self.targets(), inventory,
                                                               raw_rows=[raw, (2024, {"SECRET": True})]))
        target = self.targets()
        target["season"] = 2025
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            self.build([(inv, raw)], target)
        wrong = self.match()
        wrong[1][1]["winner_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.build([wrong])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build([(inv, raw), (inv, raw)])


if __name__ == "__main__":
    unittest.main()

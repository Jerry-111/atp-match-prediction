"""Break-point opportunity arithmetic and forbidden-information guards."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.breakpoint_features import breakpoint_pair, build_breakpoint_features, iid_hold_probability


class BreakpointFeaturesTest(unittest.TestCase):
    day = date(2023, 7, 3)

    def match(self, number=1, ago=21, surface="Hard"):
        day = self.day-timedelta(days=ago)
        inv = {"match_id": f"tour:T:{number}", "season": day.year, "source": "tour",
               "tourney_date": day.isoformat(), "surface": surface, "player_a_id": "A",
               "player_b_id": "B", "a_won": "NOT USED"}
        raw = {"tourney_id": "T", "match_num": number, "tourney_date": day.isoformat(),
               "surface": surface, "winner_id": "A", "loser_id": "B",
               "w_bpSaved": "3", "w_bpFaced": "5", "w_SvGms": "10", "w_svpt": "60", "w_1stWon": "30", "w_2ndWon": "10",
               "l_bpSaved": "2", "l_bpFaced": "6", "l_SvGms": "10", "l_svpt": "70", "l_1stWon": "30", "l_2ndWon": "10"}
        return inv, (day.year, raw)

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.day.isoformat(), "surface": "Hard",
                              "player_a_id": "A", "player_b_id": "B", "a_won": "SECRET"}], index=[91])

    def build(self, matches, target=None):
        return build_breakpoint_features(self.targets() if target is None else target,
                                         pd.DataFrame([m[0] for m in matches]), raw_rows=[m[1] for m in matches])

    def test_hand_counts_shrinkage_and_residuals(self):
        pair = breakpoint_pair(self.match()[1][1])
        self.assertEqual(pair[0], (3, 5, 4, 6, 8, 10, 4, 10, 40, 60, 30, 70))
        self.assertEqual(pair[1], (2, 6, 2, 5, 6, 10, 2, 10, 40, 70, 20, 60))
        f = self.build([self.match()]).iloc[0]
        self.assertAlmostEqual(f.bp_overall_save_diff, 16/25-15/26)
        self.assertAlmostEqual(f.bp_overall_convert_diff, 11/26-9/25)
        self.assertAlmostEqual(f.bp_overall_hold_diff, 2/30)
        self.assertAlmostEqual(f.bp_overall_break_diff, 2/30)
        self.assertAlmostEqual(f.bp_overall_hold_residual_diff,
                               2/30-iid_hold_probability(105/160)+iid_hold_probability(105/170))
        self.assertAlmostEqual(f.bp_overall_break_residual_diff,
                               2/30+iid_hold_probability(1-65/170)-iid_hold_probability(1-55/160))
        self.assertAlmostEqual(f.bp_overall_exposure_log_sum, 2*np.log(12))

    def test_iid_formula_by_finite_paths_and_deuce_solution(self):
        # Independent recursion enumerates pre-deuce paths, using the exact
        # two-point deuce recurrence as a terminal boundary condition.
        def enumerate_game(p, a=0, b=0):
            if a >= 4 and a-b >= 2:
                return 1.
            if b >= 4 and b-a >= 2:
                return 0.
            if a == b == 3:
                return p*p/(1-2*p*(1-p))
            return p*enumerate_game(p, a+1, b)+(1-p)*enumerate_game(p, a, b+1)
        for p in (0., .2, .5, .65, .9, 1.):
            self.assertAlmostEqual(iid_hold_probability(p), enumerate_game(p))
            self.assertAlmostEqual(iid_hold_probability(p)+iid_hold_probability(1-p), 1.)

    def test_inclusive_window_and_future_target_statistics_poison(self):
        matches = [self.match(), self.match(2, 364), self.match(3, 365), self.match(4, 20), self.match(5, 0)]
        expected = self.build(matches)
        self.assertAlmostEqual(expected.iloc[0].bp_overall_valid_log_sum, 2*np.log(3))
        for i in (2, 3, 4):
            matches[i][1][1].update(w_bpSaved="SECRET", w_svpt="SECRET")
        pd.testing.assert_frame_equal(expected, self.build(matches))
        later = self.targets().assign(tourney_date=(self.day+timedelta(days=70)).isoformat())
        both = pd.concat([self.targets(), later], ignore_index=True)
        matches = [self.match(), self.match(2, 0)]
        original = self.build(matches, both)
        matches[1][1][1]["w_bpSaved"] = "1"
        updated = self.build(matches, both)
        np.testing.assert_array_equal(original.iloc[0], updated.iloc[0])
        self.assertNotEqual(original.iloc[1].bp_overall_save_diff, updated.iloc[1].bp_overall_save_diff)

    def test_invalid_marginals_are_missing_and_zero_bp_valid(self):
        for key, value in (("w_bpSaved", "6"), ("w_bpFaced", "14"), ("l_SvGms", "0"),
                           ("w_svpt", "NaN"), ("l_1stWon", "100"), ("w_bpFaced", "1.5")):
            with self.subTest(key=key):
                match = self.match()
                match[1][1][key] = value
                self.assertIsNone(breakpoint_pair(match[1][1]))
                f = self.build([match]).iloc[0]
                self.assertEqual(f.bp_overall_save_diff, 0.)
                self.assertAlmostEqual(f.bp_overall_missing_log_sum, 2*np.log(2))
        match = self.match()
        match[1][1].update(w_bpSaved="0", w_bpFaced="0")
        self.assertIsNotNone(breakpoint_pair(match[1][1]))
        for changes in ({"w_bpSaved": "60", "w_bpFaced": "65"},
                        {"w_bpSaved": "45", "w_bpFaced": "50"},
                        {"w_bpSaved": "0", "w_bpFaced": "10", "w_1stWon": "49", "w_2ndWon": "10"}):
            with self.subTest(changes=changes):
                self.assertIsNone(breakpoint_pair({**self.match()[1][1], **changes}))

    def test_same_date_permutation_surface_swap_and_unseen(self):
        matches = [self.match(), self.match(2, 21, "Clay")]
        f = self.build(matches)
        pd.testing.assert_frame_equal(f, self.build(matches[::-1]))
        self.assertAlmostEqual(f.iloc[0].bp_surface_valid_log_sum, 2*np.log(2))
        self.assertAlmostEqual(f.iloc[0].bp_overall_valid_log_sum, 2*np.log(3))
        swapped = self.targets()
        swapped[["player_a_id", "player_b_id"]] = [["B", "A"]]
        np.testing.assert_allclose(self.build(matches, swapped), f.to_numpy()*
                                   [-1 if c.endswith("_diff") else 1 for c in f], atol=1e-12, rtol=0)
        swapped[["player_a_id", "player_b_id"]] = [["X", "Y"]]
        neutral = self.build(matches, swapped)
        self.assertEqual(neutral.shape, (1, 20))
        self.assertEqual(neutral.index.tolist(), [91])
        np.testing.assert_array_equal(neutral, np.zeros((1, 20)))

    def test_holdout_labels_source_identity_and_duplicates(self):
        match = self.match()
        inventory = pd.DataFrame([match[0], {"season": 2024, "source": "tour", "a_won": "SECRET"}])
        f = build_breakpoint_features(self.targets(), inventory, raw_rows=[match[1], (2024, {"SECRET": True})])
        pd.testing.assert_frame_equal(f, self.build([match]))
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            self.build([match], self.targets().assign(season=2024))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build([match, match])
        match[1][1]["winner_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.build([match])


if __name__ == "__main__":
    unittest.main()

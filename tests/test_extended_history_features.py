"""Historical information expansion must preserve time, source and parity."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.extended_history_features import (
    OfficialRatingBank, build_extended_history_features, extra_event_reason,
)
from scripts.paper_margin_features import experience_k


class ExtendedHistoryTest(unittest.TestCase):
    day = date(2023, 7, 3)

    def match(self, number=1, ago=21, extra=False, winner="A", loser="B"):
        day = self.day-timedelta(days=ago)
        raw = {"tourney_id": "T", "match_num": number, "tourney_date": day.isoformat(),
               "tourney_name": "Davis Cup WG R1: USA vs FRA" if extra else "Test Open",
               "tourney_level": "D" if extra else "250", "best_of": "3",
               "surface": "Hard", "winner_id": winner, "loser_id": loser,
               "winner_rank": "10", "loser_rank": "20", "score": "6-4 6-4"}
        inv = {"match_id": f"tour:T:{number}", "season": day.year, "source": "tour",
               "tourney_date": day.isoformat(), "surface": "Hard", "player_a_id": winner,
               "player_b_id": loser, "a_won": 1}
        return None if extra else inv, (day.year, raw)

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.day.isoformat(), "surface": "Hard",
                              "player_a_id": "A", "player_b_id": "B", "a_won": "UNUSED"}], index=[91])

    def build(self, matches, target=None):
        inv = [m[0] for m in matches if m[0] is not None]
        empty = ["match_id", "season", "source", "tourney_date", "surface", "player_a_id", "player_b_id", "a_won"]
        return build_extended_history_features(self.targets() if target is None else target,
                                              pd.DataFrame(inv, columns=empty), raw_rows=[m[1] for m in matches])

    def test_hand_update_and_matched_controls(self):
        only_tour = self.build([self.match()])
        f = only_tour.iloc[0]
        self.assertEqual(f.official_tour_k32_overall_diff, 32.)
        self.assertAlmostEqual(f.official_tour_adaptive_overall_diff, experience_k(0))
        for kind in ("k32", "adaptive"):
            for scope in ("overall", "surface", "blend"):
                self.assertEqual(f[f"official_tour_{kind}_{scope}_diff"], f[f"official_extended_{kind}_{scope}_diff"])
        # Both same-day outcomes use the same pre-date state, even for A.
        f = self.build([self.match(), self.match(2, extra=True, loser="C")]).iloc[0]
        self.assertEqual(f.official_tour_k32_overall_diff, 32.)
        self.assertEqual(f.official_extended_k32_overall_diff, 48.)
        self.assertEqual(f.official_added_k32_overall_diff, 16.)
        self.assertAlmostEqual(f.official_extra_exposure_log_diff, np.log(2))

    def test_explicit_event_and_standard_format_eligibility(self):
        raw = self.match(extra=True)[1][1]
        for level, name in (("D", "Davis Cup WG F: USA vs FRA"), ("F", "Tour Finals"),
                            ("F", "ATP Finals"), ("O", "Tokyo Olympics"), ("A", "United Cup"),
                            ("A", "Atp Cup"), ("A", "World Team Championship")):
            self.assertIsNone(extra_event_reason({**raw, "tourney_level": level, "tourney_name": name}))
        for changes in ({"tourney_name": "Unidentified D event"},
                        {"tourney_level": "A", "tourney_name": "Laver Cup"},
                        {"tourney_level": "A", "tourney_name": "Next Gen Finals"},
                        {"score": "4-2 4-2 4-2", "best_of": "5"},
                        {"score": "6-4 1-0 RET"}, {"score": "6-4 4-6 6-4 6-4"},
                        {"score": "4-6 4-6"}, {"winner_rank": ""}, {"surface": "Carpet"}):
            with self.subTest(changes=changes):
                self.assertIsNotNone(extra_event_reason({**raw, **changes}))

    def test_inclusive_cutoff_target_and_future_poison(self):
        matches = [self.match(), self.match(2, 20, extra=True), self.match(3, 0, extra=True)]
        expected = self.build(matches)
        self.assertEqual(expected.iloc[0].official_extended_k32_overall_diff, 32.)
        for _, (_, raw) in matches[1:]:
            raw.update(winner_id="SECRET", loser_id="SECRET", score="SECRET", winner_rank="SECRET")
        pd.testing.assert_frame_equal(expected, self.build(matches))
        early = self.targets()
        later = early.assign(tourney_date=(self.day+timedelta(days=70)).isoformat())
        both = pd.concat([early, later], ignore_index=True)
        original = [self.match(), self.match(2, 0, extra=True)]
        f = self.build(original, both)
        original[1][1][1].update(winner_id="B", loser_id="A")
        updated = self.build(original, both)
        np.testing.assert_array_equal(f.iloc[0], updated.iloc[0])
        self.assertNotEqual(f.iloc[1].official_extended_k32_overall_diff, updated.iloc[1].official_extended_k32_overall_diff)

    def test_permutation_swap_and_unseen_neutral_prior(self):
        matches = [self.match(), self.match(2, extra=True, loser="C"), self.match(3, 40, extra=True)]
        f = self.build(matches)
        pd.testing.assert_frame_equal(f, self.build(list(reversed(matches))))
        swapped = self.targets()
        swapped[["player_a_id", "player_b_id"]] = [["B", "A"]]
        parity = [-1 if c.endswith("_diff") else 1 for c in f]
        np.testing.assert_allclose(self.build(matches, swapped), f.to_numpy()*parity, atol=1e-12, rtol=0)
        unknown = swapped.copy()
        unknown[["player_a_id", "player_b_id"]] = [["X", "Y"]]
        neutral = self.build(matches, unknown)
        self.assertEqual(neutral.shape, (1, 20))
        self.assertEqual(neutral.index.tolist(), [91])
        self.assertTrue((neutral[neutral.attrs["odd_columns"]] == 0).all().all())
        self.assertTrue(np.isfinite(neutral.to_numpy()).all())

    def test_surface_uses_predate_state_and_not_future_roster(self):
        day = self.day
        bank = OfficialRatingBank([(day, "1", "extra", "A", "B", 1., "Hard"),
                                   (day, "2", "extra", "A", "C", 1., "Clay")], extended=True, adaptive=False)
        self.assertEqual(bank.at("A", "Hard", day)["overall"], 1532.)
        self.assertEqual(bank.at("A", "Hard", day)["surface"], 1516.)
        self.assertEqual(bank.at("A", "Hard", day-timedelta(days=1))["overall"], 1500.)
        self.assertEqual(bank.at("A", "Grass", day)["surface"], 1532.)

    def test_locked_holdout_and_duplicate_identity_checks(self):
        match = self.match()
        baseline = self.build([match])
        inventory = pd.DataFrame([match[0], {"season": 2024, "source": "tour", "a_won": "SECRET"}])
        f = build_extended_history_features(self.targets(), inventory, raw_rows=[match[1], (2024, {"SECRET": True})])
        pd.testing.assert_frame_equal(baseline, f)
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            self.build([match], self.targets().assign(season=2024))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build([match, self.match(2, extra=True), self.match(2, extra=True)])
        match[1][1]["winner_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.build([match])


if __name__ == "__main__":
    unittest.main()

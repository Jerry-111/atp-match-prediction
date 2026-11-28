"""Temporal poisoning, player symmetry and same-date rating guarantees."""

import unittest
from datetime import date, timedelta

import pandas as pd

from scripts.advanced_features import (
    HistoricalState, build_advanced_features, elo_probability, mirror_features,
)


class AdvancedFeaturesTest(unittest.TestCase):
    start = date(2023, 7, 3)

    def match(self, number, ago, a="A", b="B", won=1, surface="Hard", source="tour"):
        day = self.start - timedelta(days=ago)
        return {
            "match_id": f"{source}:T:{number}", "source": source, "season": day.year,
            "tourney_date": day.strftime("%Y%m%d"), "player_a_id": a,
            "player_b_id": b, "a_won": won, "surface": surface,
        }

    def targets(self):
        return pd.DataFrame([{
            "match_id": "target", "season": 2023, "tourney_date": self.start.isoformat(),
            "player_a_id": "A", "player_b_id": "B", "rank_a": 7, "rank_b": 25,
            "surface": "Hard", "a_won": 1,
        }], index=[42])

    def test_cutoff_is_inclusive_and_recent_results_cannot_poison(self):
        past = self.match(1, 21)
        too_recent = self.match(2, 20, won=0)
        current = self.match(3, 0, won=0)
        old = self.match(4, 91, b="C", won=0)
        inv = pd.DataFrame([past, too_recent, current, old])
        baseline = build_advanced_features(self.targets(), inv)
        self.assertEqual(list(baseline.index), [42])
        self.assertEqual(baseline.loc[42, "career_min_count"], 1)
        self.assertAlmostEqual(baseline.loc[42, "form_90_rate_diff"], 1 / 3)
        # Alter all outcomes newer than cutoff, including the target's own
        # outcome, and append held-out rows whose labels cannot be converted.
        inv.loc[[1, 2], "a_won"] = 1
        inv = pd.concat([inv, pd.DataFrame([
            {"source": "tour", "season": 2024, "a_won": "POISON"},
            {"source": "tour", "season": 2025, "a_won": "POISON"},
            {"source": "challenger", "season": 2023, "a_won": "POISON"},
        ])], ignore_index=True)
        targets = self.targets()
        targets["a_won"] = 0
        pd.testing.assert_frame_equal(baseline, build_advanced_features(targets, inv))

    def test_player_exchange_exactly_matches_documented_signs(self):
        inv = pd.DataFrame([
            self.match(1, 365, won=0), self.match(2, 364, a="A", b="C"),
            self.match(3, 90, a="B", b="C", surface="Grass"),
            self.match(4, 21, won=1), self.match(5, 21, a="A", b="D"),
        ])
        targets = self.targets()
        actual = build_advanced_features(targets, inv)
        swapped = targets.copy()
        swapped[["player_a_id", "player_b_id"]] = targets[["player_b_id", "player_a_id"]].to_numpy()
        swapped[["rank_a", "rank_b"]] = targets[["rank_b", "rank_a"]].to_numpy()
        swapped["a_won"] = 1 - targets["a_won"]
        pd.testing.assert_frame_equal(build_advanced_features(swapped, inv), mirror_features(actual))

    def test_same_date_rating_updates_are_batched_and_order_invariant(self):
        # Each starts this date at 1500: two wins give A +32, B and C -16.
        inv = pd.DataFrame([self.match(1, 21), self.match(2, 21, b="C")])
        state = HistoricalState(inv)
        cutoff = self.start - timedelta(days=21)
        self.assertEqual(state.ratings[32]["A"].at(cutoff), 1532)
        self.assertEqual(state.ratings[32]["B"].at(cutoff), 1484)
        self.assertEqual(state.surface_ratings[32]["A", "Hard"].at(cutoff), 1532)
        self.assertEqual(state.ratings[32]["A"].at(cutoff - timedelta(days=1)), 1500)
        self.assertAlmostEqual(elo_probability(400), 10 / 11)
        pd.testing.assert_frame_equal(
            build_advanced_features(self.targets(), inv),
            build_advanced_features(self.targets(), inv.iloc[::-1]),
        )

    def test_unknown_player_has_finite_neutral_history(self):
        targets = self.targets()
        targets["player_a_id"] = "UNSEEN1"
        targets["player_b_id"] = "UNSEEN2"
        frame = build_advanced_features(targets, pd.DataFrame([self.match(1, 21)]))
        self.assertFalse(frame.isna().any().any())
        self.assertEqual(frame.loc[42, "elo_k32_diff"], 0)
        self.assertEqual(frame.loc[42, "form_90_rate_diff"], 0)
        self.assertEqual(frame.loc[42, "h2h_career_rate_diff"], 0)
        self.assertEqual(frame.loc[42, "overall_unseen_sum"], 2)

    def test_holdout_targets_are_locked_and_duplicate_history_rejected(self):
        targets = self.targets()
        targets["season"] = 2024
        inv = pd.DataFrame([self.match(1, 21)])
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            build_advanced_features(targets, inv)
        with self.assertRaisesRegex(ValueError, "Duplicate history"):
            HistoricalState(pd.concat([inv, inv]))


if __name__ == "__main__":
    unittest.main()

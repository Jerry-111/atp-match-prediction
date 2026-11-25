"""Opponent adjustment, date batching, exchange symmetry and temporal guards."""

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.adjusted_point_features import (
    AdjustedPointState, Skill, _joined_history, build_adjusted_point_features, hold_probability,
)
from scripts.advanced_features import mirror_features


class AdjustedPointFeaturesTest(unittest.TestCase):
    start = date(2023, 7, 3)

    def match(self, number, ago, a="A", b="B", serve_a=60, total_a=100,
              serve_b=40, total_b=80, surface="Hard"):
        day = self.start - timedelta(days=ago)
        inv = {"match_id": f"tour:T:{number}", "source": "tour", "season": day.year,
               "tourney_date": day.strftime("%Y%m%d"), "player_a_id": a, "player_b_id": b,
               "a_won": 1, "surface": surface}
        raw = {"tourney_id": "T", "match_num": number, "tourney_date": inv["tourney_date"],
               "winner_id": a, "loser_id": b, "surface": surface,
               "w_1stWon": str(serve_a), "w_2ndWon": "0", "w_svpt": str(total_a),
               "l_1stWon": str(serve_b), "l_2ndWon": "0", "l_svpt": str(total_b)}
        return inv, (day.year, raw)

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.start.isoformat(),
                              "player_a_id": "A", "player_b_id": "B", "surface": "Hard",
                              "a_won": 1}], index=[42])

    def build(self, matches, targets=None):
        inv, raw = zip(*matches)
        return build_adjusted_point_features(self.targets() if targets is None else targets,
                                            pd.DataFrame(inv), raw_rows=raw)

    def test_opponent_strength_changes_identical_observed_rates(self):
        # X previously returned very well, Y poorly, against interchangeable servers.
        matches = []
        for k in range(12):
            matches.extend([
                self.match(2*k, 250-k*7, a="X", b=f"S{k}", serve_a=65, serve_b=35, total_b=100),
                self.match(2*k+1, 250-k*7, a="Y", b=f"Q{k}", serve_a=65, serve_b=85, total_b=100),
            ])
        matches.extend([
            self.match(30, 21, a="A", b="X", serve_a=65, serve_b=65, total_b=100),
            self.match(31, 21, a="B", b="Y", serve_a=65, serve_b=65, total_b=100),
        ])
        row = self.build(matches).loc[42]
        self.assertGreater(row["adjpoints_h365_overall_serve_skill_diff"], 0)
        self.assertGreater(row["adjpoints_h730_overall_serve_skill_diff"], 0)

    def test_date_batched_update_and_input_permutation(self):
        matches = [self.match(1, 21), self.match(2, 21, b="C")]
        inv, raw = zip(*matches)
        records, _ = _joined_history(pd.DataFrame(inv), None, raw)
        state = AdjustedPointState(records, 365)
        cutoff = self.start - timedelta(days=21)
        a = state.overall["A", "serve"].at(cutoff, 365)
        # Both games used p=.65 before either game updated A's skill.
        self.assertAlmostEqual(a.mean, .5 * (120 - 200*.65) / (25 + 200*.65*.35))
        self.assertEqual(a.points, 200)
        pd.testing.assert_frame_equal(self.build(matches), self.build(list(reversed(matches))))

    def test_cutoff_poisoning_and_heldout_source_not_inspected(self):
        matches = [self.match(1, 21), self.match(2, 20), self.match(3, 0)]
        before = self.build(matches)
        self.assertEqual(before.loc[42, "adjpoints_h365_exposure_min"], 180)
        for _, (_, raw) in matches[1:]:
            raw["w_1stWon"] = "99"
            raw["l_1stWon"] = "1"
        targets = self.targets()
        targets["a_won"] = "POISON"
        after = self.build(matches, targets)
        pd.testing.assert_frame_equal(before, after)
        inv, raw = zip(*matches)
        inv = pd.concat([pd.DataFrame(inv), pd.DataFrame([
            {"source": "tour", "season": 2024, "a_won": "POISON"},
            {"source": "challenger", "season": 2023, "a_won": "POISON"}])])
        poisoned = build_adjusted_point_features(targets, inv, raw_rows=[*raw, (2025, {"POISON": True})])
        pd.testing.assert_frame_equal(after, poisoned)

    def test_symmetry_unknown_players_and_finite_outputs(self):
        matches = [self.match(1, 35), self.match(2, 21, b="C", surface="Clay")]
        before = self.build(matches)
        self.assertEqual(before.shape, (1, 34))
        targets = self.targets()
        targets[["player_a_id", "player_b_id"]] = targets[["player_b_id", "player_a_id"]].to_numpy()
        pd.testing.assert_frame_equal(self.build(matches, targets), mirror_features(before))
        targets[["player_a_id", "player_b_id"]] = ["NEW_A", "NEW_B"]
        unseen = self.build(matches, targets)
        self.assertTrue(np.isfinite(unseen.to_numpy()).all())
        self.assertEqual(unseen.loc[42, "adjpoints_h365_unseen_sum"], 2)
        self.assertTrue((unseen.filter(like="_diff").to_numpy() == 0).all())

    def test_invalid_stats_and_identity_mismatch(self):
        inv, raw = self.match(1, 21)
        raw[1]["l_svpt"] = ""
        frame = self.build([(inv, raw)])
        self.assertEqual(frame.attrs["missing_statistics_matches"], 1)
        self.assertEqual(frame.loc[42, "adjpoints_h365_unseen_sum"], 2)
        raw[1]["winner_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.build([(inv, raw)])

    def test_holdout_guard(self):
        targets = self.targets()
        targets["season"] = 2024
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            self.build([self.match(1, 21)], targets)

    def test_discounting_and_exact_game_probability(self):
        old = Skill(mean=1, precision=50, points=100, day=self.start)
        new = old.at(self.start + timedelta(days=365), 365)
        self.assertAlmostEqual(new.mean, 2/3)
        self.assertEqual(new.precision, 37.5)
        self.assertEqual(new.points, 50)
        self.assertAlmostEqual(hold_probability(.5), .5)
        self.assertEqual(hold_probability(0), 0)
        self.assertEqual(hold_probability(1), 1)
        for p in (.1, .3, .65, .9):
            self.assertAlmostEqual(hold_probability(p) + hold_probability(1-p), 1)


if __name__ == "__main__":
    unittest.main()

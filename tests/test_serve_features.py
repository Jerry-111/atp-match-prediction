"""Point-rate calculations, missingness, symmetry and future-stat poisoning."""

import unittest
from datetime import date, timedelta

import pandas as pd

from scripts.advanced_features import mirror_features
from scripts.serve_features import build_serve_features, development_entries


class ServeFeaturesTest(unittest.TestCase):
    start = date(2023, 7, 3)

    def match(self, number, ago, surface="Hard", bad=False):
        day = self.start - timedelta(days=ago)
        inv = {"match_id": f"tour:T:{number}", "source": "tour", "season": day.year,
               "tourney_date": day.strftime("%Y%m%d"), "player_a_id": "A", "player_b_id": "B",
               "a_won": 1, "surface": surface}
        raw = {"tourney_id": "T", "match_num": number, "tourney_date": inv["tourney_date"],
               "winner_id": "A", "loser_id": "B", "surface": surface,
               "w_1stWon": "40", "w_2ndWon": "20", "w_svpt": "100",
               "l_1stWon": "30", "l_2ndWon": "10", "l_svpt": "80"}
        if bad:
            raw["l_svpt"] = ""
        return inv, (day.year, raw)

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.start.isoformat(),
                              "player_a_id": "A", "player_b_id": "B", "surface": "Hard",
                              "a_won": 1}], index=[42])

    def build(self, matches, targets=None):
        inv, raw = zip(*matches)
        return build_serve_features(self.targets() if targets is None else targets,
                                    pd.DataFrame(inv), raw_rows=raw)

    def test_opportunity_weighting_and_inclusive_cutoff(self):
        frame = self.build([self.match(1, 21), self.match(2, 364), self.match(3, 20),
                            self.match(4, 365)])
        self.assertEqual(list(frame.index), [42])
        row = frame.loc[42]
        self.assertEqual(row["points_overall_364_valid_matches_min"], 2)
        self.assertEqual(row["points_overall_728_valid_matches_min"], 3)
        self.assertAlmostEqual(row["points_overall_364_serve_rate_diff"],
                               (120 + 65) / 300 - (80 + 65) / 260)
        self.assertAlmostEqual(row["points_overall_364_return_rate_diff"],
                               (80 + 35) / 260 - (80 + 35) / 300)

    def test_invalid_pair_missingness_and_surface(self):
        frame = self.build([self.match(1, 21, bad=True), self.match(2, 22, surface="Clay")])
        row = frame.loc[42]
        self.assertEqual(row["points_surface_364_valid_matches_min"], 0)
        self.assertEqual(row["points_surface_364_missing_matches_min"], 1)
        self.assertEqual(row["points_surface_364_unseen_sum"], 2)
        self.assertEqual(row["points_surface_364_serve_rate_diff"], 0)
        self.assertEqual(row["points_overall_364_valid_matches_min"], 1)

    def test_future_statistics_cannot_poison_and_holdout_fields_are_not_read(self):
        matches = [self.match(1, 21), self.match(2, 20), self.match(3, 0)]
        first = self.build(matches)
        for _, (_, raw) in matches[1:]:
            raw["w_1stWon"] = "9999999"
        targets = self.targets()
        targets["a_won"] = "POISON"
        second = self.build(matches, targets)
        # Provenance valid/missing tallies can change; feature rows cannot.
        first.attrs.clear()
        second.attrs.clear()
        pd.testing.assert_frame_equal(first, second)
        inv, raw = zip(*matches)
        inventory = pd.concat([pd.DataFrame(inv), pd.DataFrame([
            {"source": "tour", "season": 2024, "a_won": "POISON"}])])
        poisoned = build_serve_features(targets, inventory, raw_rows=[*raw, (2024, {"POISON": True})])
        poisoned.attrs.clear()
        pd.testing.assert_frame_equal(second, poisoned)

    def test_player_exchange_and_unknown_players(self):
        matches = [self.match(1, 21), self.match(2, 25, surface="Clay")]
        frame = self.build(matches)
        targets = self.targets()
        targets[["player_a_id", "player_b_id"]] = targets[["player_b_id", "player_a_id"]].to_numpy()
        pd.testing.assert_frame_equal(self.build(matches, targets), mirror_features(frame))
        targets[["player_a_id", "player_b_id"]] = ["NEW1", "NEW2"]
        unseen = self.build(matches, targets)
        self.assertFalse(unseen.isna().any().any())
        self.assertEqual(unseen.loc[42, "points_overall_364_unseen_sum"], 2)

    def test_holdout_guard_and_manifest_filter(self):
        self.assertEqual([e["season"] for e in development_entries()], list(range(2006, 2024)))
        targets = self.targets()
        targets["season"] = 2024
        with self.assertRaisesRegex(ValueError, "holdout is locked"):
            self.build([self.match(1, 21)], targets)

    def test_identity_mismatch_and_invalid_counts(self):
        inv, raw = self.match(1, 21)
        raw[1]["winner_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.build([(inv, raw)])
        for invalid in ("nan", "-1", "1000", "1.2"):
            inv, raw = self.match(1, 21)
            raw[1]["w_1stWon"] = invalid
            frame = self.build([(inv, raw)])
            self.assertEqual(frame.loc[42, "points_overall_364_missing_matches_min"], 1)


if __name__ == "__main__":
    unittest.main()

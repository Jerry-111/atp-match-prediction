"""Historical-cohort expansion without changing original rows or feature schema."""

import unittest

import pandas as pd

from scripts.advanced_features import HistoricalState, build_advanced_features
from scripts.adjusted_point_features import build_adjusted_point_features
from scripts.dynamic_features import build_dynamic_features
from scripts.expanded_training import BASE, _target_row, build_expanded_features, build_expanded_targets
from scripts.serve_features import build_serve_features


class ExpandedTrainingTest(unittest.TestCase):
    def fixtures(self):
        inventory, raw = [], []
        for season, day in ((2008, "20081201"), (2009, "20090105"), (2015, "20150105"),
                            (2016, "20160104"), (2023, "20230102")):
            slam = season in (2015, 2023)
            record = {"match_id": f"tour:{season}-T:1", "source": "tour", "season": season,
                      "tourney_id": f"{season}-T", "tourney_date": day, "match_num": 1,
                      "player_a_id": "A", "player_b_id": "B", "player_a_name": "Alpha",
                      "player_b_name": "Beta", "rank_a": 10, "rank_b": 20, "a_won": 1,
                      "round": "F", "surface": "Hard", "tourney_level": "G" if slam else "250"}
            inventory.append(record)
            raw.append((season, {"tourney_id": f"{season}-T", "match_num": 1, "tourney_date": day,
                                "winner_id": "A", "loser_id": "B", "surface": "Hard",
                                "w_1stWon": "60", "w_2ndWon": "0", "w_svpt": "100",
                                "l_1stWon": "40", "l_2ndWon": "0", "l_svpt": "80",
                                "best_of": "5" if slam else "3"}))
        inventory = pd.DataFrame(inventory)
        state = HistoricalState(inventory)
        original = pd.DataFrame([_target_row(row, state) for row in inventory.to_dict("records")
                                 if row["season"] >= 2016])
        return original, inventory, raw

    def test_expansion_preserves_original_values_and_actual_seasons(self):
        original, inventory, _ = self.fixtures()
        expanded = build_expanded_targets(original, inventory)
        self.assertEqual(list(expanded.season), [2009, 2015, 2016, 2023])
        self.assertEqual(list(expanded.split), ["train", "train", "train", "validation"])
        pd.testing.assert_frame_equal(expanded.iloc[-2:].reset_index(drop=True), original, check_exact=True)
        self.assertEqual(expanded.attrs["added_training_rows"], 2)
        # The prior 2008 match contributes one win at the 2009 target cutoff.
        self.assertAlmostEqual(expanded.iloc[0].overall_rate_diff, 1/3)
        self.assertAlmostEqual(expanded.iloc[0].surface_rate_diff, 1/3)
        self.assertAlmostEqual(expanded.iloc[0].h2h_rate_diff, 1/3)

    def test_canonical_player_order_is_outcome_independent(self):
        original, inventory, _ = self.fixtures()
        idx = inventory.season.eq(2009)
        inventory.loc[idx, ["player_a_id", "player_b_id"]] = ["B", "A"]
        inventory.loc[idx, ["player_a_name", "player_b_name"]] = ["Beta", "Alpha"]
        inventory.loc[idx, ["rank_a", "rank_b"]] = [20, 10]
        inventory.loc[idx, "a_won"] = 0
        expanded = build_expanded_targets(original, inventory)
        row = expanded.iloc[0]
        self.assertEqual((row.player_a_id, row.player_b_id, row.a_won), ("A", "B", 1))
        self.assertEqual((row.rank_a, row.rank_b), (10, 20))

    def test_original_feature_corruption_is_rejected(self):
        original, inventory, _ = self.fixtures()
        original.loc[0, "surface_rate_diff"] += .01
        with self.assertRaisesRegex(ValueError, "feature mismatch"):
            build_expanded_targets(original, inventory)

    def test_all_default_gates_and_explicit_bounds(self):
        original, inventory, raw = self.fixtures()
        expanded = build_expanded_targets(original, inventory)
        for fn in (build_advanced_features, build_serve_features, build_dynamic_features,
                   build_adjusted_point_features):
            options = {"raw_rows": raw} if fn in (build_serve_features, build_adjusted_point_features) else {}
            with self.assertRaisesRegex(ValueError, "holdout is locked"):
                fn(expanded, inventory, **options)
            actual = fn(expanded, inventory, first_target_season=2009, **options)
            self.assertEqual(len(actual), 4)
            for invalid in (2008, 2017, 2024):
                with self.assertRaisesRegex(ValueError, "first_target_season"):
                    fn(expanded, inventory, first_target_season=invalid, **options)
            heldout = original.copy()
            heldout["season"] = 2024
            with self.assertRaisesRegex(ValueError, "holdout is locked"):
                fn(heldout, inventory, first_target_season=2009, **options)

    def test_198_original_features_exactly_preserved_and_format_appended(self):
        original, inventory, raw = self.fixtures()
        expanded = build_expanded_targets(original, inventory)
        x = build_expanded_features(expanded, inventory, raw_rows=raw)
        old = pd.concat([original[BASE], build_advanced_features(original, inventory),
                         build_serve_features(original, inventory, raw_rows=raw),
                         build_dynamic_features(original, inventory),
                         build_adjusted_point_features(original, inventory, raw_rows=raw)], axis=1)
        self.assertEqual(x.shape, (4, 202))
        tail = x.iloc[-2:, :198].reset_index(drop=True)
        tail.attrs.clear()
        old.attrs.clear()
        pd.testing.assert_frame_equal(tail, old, check_exact=True)
        self.assertEqual(list(x.format_best_of_five), [0, 1, 0, 1])
        self.assertEqual(x.loc[1, "format_log_rank_diff"], expanded.loc[1, "log_rank_diff"])
        self.assertEqual(x.loc[0, "format_elo_diff"], 0)

    def test_holdout_inventory_poison_and_actual_raw_format(self):
        original, inventory, raw = self.fixtures()
        expected = build_expanded_targets(original, inventory)
        inventory = pd.concat([inventory, pd.DataFrame([
            {"source": "tour", "season": 2024, "a_won": "POISON"},
            {"source": "challenger", "season": 2012, "a_won": "POISON"}])])
        pd.testing.assert_frame_equal(build_expanded_targets(original, inventory), expected)
        # Real format metadata is authoritative even if a fixture's level disagrees.
        raw[1][1]["best_of"] = "5"
        x = build_expanded_features(expected, inventory, raw_rows=[*raw, (2025, {"POISON": True})])
        self.assertEqual(x.loc[0, "format_best_of_five"], 1)
        raw[1][1]["best_of"] = ""
        with self.assertRaisesRegex(ValueError, "best_of"):
            build_expanded_features(expected, inventory, raw_rows=raw)


if __name__ == "__main__":
    unittest.main()

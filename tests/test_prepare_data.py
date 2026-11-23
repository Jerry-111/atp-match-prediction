"""Regression checks for information available at the tournament start."""

import unittest
from datetime import date, timedelta

from scripts.prepare_data import build_model_rows


class ModelInputTest(unittest.TestCase):
    def test_window_endpoints_and_player_orientation(self):
        target_date = date(2016, 1, 4)

        def match(number, days_ago, winner="A100", loser="Z900", surface="Hard"):
            start = target_date - timedelta(days=days_ago)
            return {
                "season": start.year, "tourney_date": start, "tourney_id": "T",
                "match_num": number, "round": "R32", "surface": surface,
                "winner_id": winner, "loser_id": loser, "winner_name": winner,
                "loser_name": loser, "winner_rank": 5, "loser_rank": 9,
            }

        records = [
            match(1, 365), match(2, 364), match(3, 21, "Z900", "A100"),
            match(4, 20), match(5, 30, surface="Clay"), match(6, 0),
            match(7, 0, winner="Z900", loser="A100"),
            {"season": 2024},  # Test outcomes must never be read.
        ]
        rows = build_model_rows(reversed(records))
        self.assertEqual([row["match_id"] for row in rows], ["T:6", "T:7"])
        for row in rows:
            self.assertEqual((row["player_a_id"], row["player_b_id"]), ("A100", "Z900"))
            for side in ("a", "b"):
                self.assertEqual(row[f"surface_matches_{side}"], 2)
                self.assertEqual(row[f"surface_wins_{side}"], 1)
                self.assertEqual(row[f"surface_rate_{side}"], 0.5)
                self.assertEqual(row[f"latest_history_date_{side}"], "2015-12-14")
        self.assertEqual((rows[0]["a_won"], rows[1]["a_won"]), (1, 0))
        self.assertGreater(rows[0]["log_rank_diff"], 0)
        self.assertLess(rows[1]["log_rank_diff"], 0)


if __name__ == "__main__":
    unittest.main()

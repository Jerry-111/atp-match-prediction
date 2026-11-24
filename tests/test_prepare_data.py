"""Regression checks for information available at the tournament start."""

import tempfile
import unittest
from datetime import date, timedelta

from scripts.prepare_data import build_model_rows, write_model_csv


class ModelInputTest(unittest.TestCase):
    def test_historical_cohort_warmup_and_export(self):
        def match(season, number, start):
            return {
                "season": season, "tourney_date": start, "tourney_id": "T",
                "match_num": number, "round": "R32", "surface": "Hard",
                "winner_id": "A100", "loser_id": "Z900", "winner_name": "A",
                "loser_name": "Z", "winner_rank": 5, "loser_rank": 9,
            }

        matches = [
            {"season": 1999},  # Outside the chosen cohort.
            match(2000, 1, date(2000, 12, 1)),
            match(2001, 2, date(2001, 1, 5)),
            match(2015, 3, date(2015, 10, 1)),
            match(2023, 4, date(2023, 8, 1)),
            {"season": 2024},
            {"season": 2025},
        ]
        rows = build_model_rows(matches, first_season=2000)
        self.assertEqual([row["split"] for row in rows], ["train", "train", "validation"])
        self.assertEqual([row["season"] for row in rows], [2001, 2015, 2023])
        self.assertEqual(rows[0]["surface_wins_a"], 1)
        with tempfile.TemporaryDirectory() as directory:
            base = write_model_csv(build_model_rows(matches), directory)
            base_contents = base.read_bytes()
            extended = write_model_csv(rows, directory, first_season=2000)
            self.assertEqual(base.name, "model_input.csv")
            self.assertEqual(extended.name, "model_input_2000.csv")
            self.assertEqual(base.read_bytes(), base_contents)

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

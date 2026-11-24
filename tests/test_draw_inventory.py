"""Regression checks for draw identity and held-out labels in the local inventory."""

import csv
import io
import unittest

from scripts.audit_draw_coverage import Inventory


class DrawInventoryTest(unittest.TestCase):
    def test_distinct_draws_round_robin_and_held_out_outcomes(self):
        output = io.StringIO()
        inventory = Inventory(output)
        match = {
            "tourney_id": "T", "tourney_name": "Example", "tourney_level": "C",
            "tourney_date": "20221205", "surface": "Hard", "round": "RR", "match_num": "1",
            "winner_id": "Z9", "winner_name": "Z", "winner_rank": "2",
            "loser_id": "A1", "loser_name": "A", "loser_rank": "5",
        }
        inventory.write(match, "challenger", 2022)
        inventory.write(dict(match, match_num="2"), "challenger", 2022)
        inventory.write(dict(match, tourney_level="250", round="Q1"), "qualifying", 2022)
        inventory.write(dict(match, tourney_id="H", tourney_date="20241205"), "challenger", 2024)
        output.seek(0)
        rows = list(csv.DictReader(output))
        self.assertEqual([r["match_id"] for r in rows], [
            "challenger:T:1", "challenger:T:2", "qualifying:T:1", "challenger:H:1",
        ])
        self.assertTrue(all(r["player_a_id"] == "A1" and r["rank_a"] == "5" for r in rows))
        self.assertEqual([r["a_won"] for r in rows], ["0", "0", "0", ""])
        self.assertEqual([r["stage"] for r in rows], ["main", "main", "qualifying", "main"])
        self.assertEqual(rows[0]["tourney_level"], "C")


if __name__ == "__main__":
    unittest.main()

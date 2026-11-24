"""Checks for fixed pre-start rankings used by audited ATP main draws."""

import unittest

from scripts.model_event_metadata import EVENT_METADATA, RANKS_BY_WEEK, corrected_tour_row


class ModelEventMetadataTest(unittest.TestCase):
    def test_same_weekly_ranks_independent_of_round_and_winner(self):
        row = {
            "tourney_id": "2019-418", "tourney_date": "20190727", "round": "R64",
            "winner_id": "KE17", "winner_rank": "52", "loser_id": "MM58", "loser_rank": "10",
        }
        original = row.copy()
        effective = corrected_tour_row(row)
        self.assertEqual(effective["tourney_date"], "20190729")
        self.assertEqual((effective["winner_rank"], effective["loser_rank"]), ("48", "9"))
        reversed_row = dict(row, round="F", winner_id="MM58", loser_id="KE17")
        result = corrected_tour_row(reversed_row)
        self.assertEqual((result["winner_rank"], result["loser_rank"]), ("9", "48"))
        self.assertEqual(row, original)

    def test_only_named_main_draws_change(self):
        row = {"tourney_id": "2019-418", "round": "Q2"}
        self.assertIs(corrected_tour_row(row), row)
        unrelated = {"tourney_id": "2018-418", "round": "QF"}
        self.assertIs(corrected_tour_row(unrelated), unrelated)
        with self.assertRaisesRegex(ValueError, "Missing verified 20190722 rank"):
            corrected_tour_row(dict(row, round="R32", winner_id="UNKNOWN", loser_id="MM58"))

    def test_snapshot_coverage_and_dates(self):
        self.assertTrue(all(week < start for start, week in EVENT_METADATA.values()))
        self.assertEqual({week: len(ranks) for week, ranks in RANKS_BY_WEEK.items()},
                         {"20190722": 48, "20240812": 48, "20241028": 56})
        self.assertEqual(RANKS_BY_WEEK["20241028"]["Y09V"], 74)


if __name__ == "__main__":
    unittest.main()

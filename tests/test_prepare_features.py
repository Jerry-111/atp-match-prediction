"""Check historical boundaries and lagged matchups against a small explicit history."""

import unittest
from datetime import date, timedelta

from scripts.prepare_features import build_feature_rows


class FourGroupFeaturesTest(unittest.TestCase):
    def test_overall_window_h2h_history_and_target_order(self):
        start = date(2016, 1, 4)

        def match(number, days_ago, a="A", b="B", a_won="1", source="tour"):
            played = start - timedelta(days=days_ago)
            return {
                "match_id": f"{source}:T:{number}", "source": source, "season": str(played.year),
                "tourney_date": played.strftime("%Y%m%d"), "player_a_id": a, "player_b_id": b,
                "a_won": a_won,
            }

        current = match(7, 0, a_won="0")
        first_meeting = match(8, 0, a="A", b="D")
        inventory = [
            match(1, 91), match(2, 90, a_won="0"), match(3, 21, b="C"),
            match(4, 21, a="B", b="C"), match(5, 20), match(9, 150),
            match(6, 22, source="challenger"), current, first_meeting,
            {"source": "tour", "season": "2024"},  # Held-out labels are never required.
        ]
        targets = [
            {**m, "tourney_date": start.isoformat(), "match_id": m["match_id"][5:], "split": "train"}
            for m in (first_meeting, current)
        ]
        new_pair, known_pair = build_feature_rows(targets, reversed(inventory))
        self.assertEqual(new_pair["match_id"], "T:8")
        self.assertEqual((new_pair["h2h_matches"], new_pair["h2h_rate_diff"]), (0, 0))
        self.assertEqual(new_pair["overall_matches_b"], 0)
        self.assertEqual((known_pair["overall_matches_a"], known_pair["overall_wins_a"]), (2, 1))
        self.assertEqual((known_pair["overall_matches_b"], known_pair["overall_wins_b"]), (2, 2))
        self.assertEqual(known_pair["overall_rate_diff"], -0.25)
        self.assertEqual((known_pair["h2h_matches"], known_pair["h2h_wins_a"]), (3, 2))
        self.assertAlmostEqual(known_pair["h2h_rate_diff"], 0.2)

        with self.assertRaisesRegex(ValueError, "does not match"):
            list(build_feature_rows([{**targets[0], "player_b_id": "B"}], inventory))
        with self.assertRaisesRegex(ValueError, "missing 1 eligible"):
            list(build_feature_rows(targets[:1], inventory))

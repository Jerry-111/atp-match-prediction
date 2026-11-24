"""Exercise the conservative identity crosswalk independently of annual files."""

import unittest
from datetime import date

from scripts.supplemental_draws_2006 import crosswalk_players


class CrosswalkTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "tourney_date": "20060115",
            "winner_id": "101", "winner_name": "André Martín", "winner_age": "20.0",
            "loser_id": "102", "loser_name": "Alex Smith", "loser_age": "20.0",
        }
        self.birth = date(2006, 1, 15).toordinal() - round(20 * 365.25)

    def test_candidate_requires_unique_name_and_age_match(self):
        index = {
            "andremartin": {"M001": {self.birth + 45}, "M002": {self.birth + 46}},
            "alexsmith": {"S001": {self.birth}, "S002": {self.birth}},
        }
        self.assertEqual(crosswalk_players([self.row], index), {"101": "M001"})
        self.assertEqual(crosswalk_players([dict(self.row, winner_age="")], index), {})

    def test_two_native_ids_cannot_share_an_atp_id(self):
        index = {"andremartin": {"M001": {self.birth}}}
        other = dict(self.row, winner_id="103")
        self.assertEqual(crosswalk_players([self.row, other], index), {})


if __name__ == "__main__":
    unittest.main()

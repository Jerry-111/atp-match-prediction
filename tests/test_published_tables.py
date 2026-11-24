"""Cross-check the two published CSVs using only their shipped contents."""

import csv
import unittest
from datetime import datetime
from pathlib import Path


PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SHARED_FIELDS = (
    "season", "split", "tourney_id", "match_num", "round", "surface",
    "player_a_id", "player_b_id", "player_a_name", "player_b_name", "rank_a", "rank_b", "a_won",
)


class PublishedTablesTest(unittest.TestCase):
    def test_model_table_matches_the_labeled_tour_cohort(self):
        periods = {year: "train" for year in range(2007, 2023)}
        periods.update({2006: "warmup", 2023: "validation", 2024: "test", 2025: "test"})
        seen = set()
        inventory_seasons = set()
        expected = {}
        with (PROCESSED / "mens_singles_2006_2025.csv").open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                key = row["match_id"]
                self.assertEqual(key, f"{row['source']}:{row['tourney_id']}:{row['match_num']}")
                if key in seen:
                    self.fail(f"Duplicate inventory match {key}")
                seen.add(key)
                year = int(row["season"])
                inventory_seasons.add(year)
                self.assertEqual(row["split"], periods.get(year), key)
                if row["split"] == "test":
                    self.assertEqual(row["a_won"], "", key)
                else:
                    self.assertIn(row["a_won"], ("0", "1"), key)

                if row["source"] == "tour" and 2016 <= year <= 2023:
                    self.assertEqual(row["stage"], "main", key)
                    self.assertIn(row["tourney_level"], ("250", "500", "M", "G"), key)
                    self.assertLess(row["player_a_id"], row["player_b_id"], key)
                    date = datetime.strptime(row["tourney_date"], "%Y%m%d").date().isoformat()
                    expected[key] = (date, tuple(row[field] for field in SHARED_FIELDS))
        self.assertEqual(inventory_seasons, set(periods))

        model_seasons = set()
        with (PROCESSED / "model_input.csv").open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                key = "tour:" + row["match_id"]
                reference = expected.pop(key, None)
                self.assertIsNotNone(reference, f"Unexpected or repeated model match {key}")
                fields = tuple(row[field] for field in SHARED_FIELDS)
                self.assertEqual((row["tourney_date"], fields), reference, key)
                model_seasons.add(int(row["season"]))
        if expected:
            self.fail(f"Model table omits {len(expected)} inventory matches, including {next(iter(expected))}")
        self.assertEqual(model_seasons, set(range(2016, 2024)))


if __name__ == "__main__":
    unittest.main()

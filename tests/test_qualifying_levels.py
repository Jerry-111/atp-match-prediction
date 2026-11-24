"""Tests for the narrowly scoped 2010 qualifying metadata recovery."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.qualifying_levels import effective_qualifying_row, load_2010_event_metadata


class QualifyingMetadataTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "tourney_id": "2010-540", "tourney_name": "Wimbledon", "tourney_date": "20100621",
            "tourney_level": "", "surface": "", "round": "Q1",
        }
        self.metadata = {"2010-540": ("Wimbledon", "20100621", "G", "Grass")}

    def test_restores_both_fields_with_provenance_without_changing_source(self):
        original = self.row.copy()
        effective, source = effective_qualifying_row(self.row, 2010, self.metadata)
        self.assertEqual(effective, dict(original, tourney_level="G", surface="Grass"))
        self.assertEqual(source, "tour:2010-540")
        self.assertEqual(self.row, original)
        self.assertIsNot(effective, self.row)

    def test_leaves_nonmatching_and_already_reported_fields_untouched(self):
        extra = dict(self.metadata, unrelated=("Wimbledon", "20100621", "G", "Grass"))
        cases = [
            (2011, self.row, self.metadata),
            (2010, dict(self.row, tourney_id="unrelated"), extra),
            (2010, self.row, {}),
            (2010, dict(self.row, tourney_name="Other name"), self.metadata),
            (2010, dict(self.row, tourney_date="20100620"), self.metadata),
            (2010, dict(self.row, tourney_level="G"), self.metadata),
            (2010, dict(self.row, surface="Hard"), self.metadata),
        ]
        for year, row, metadata in cases:
            with self.subTest(year=year, row=row, metadata=metadata):
                original = row.copy()
                effective, source = effective_qualifying_row(row, year, metadata)
                self.assertIs(effective, row)
                self.assertEqual(row, original)
                self.assertEqual(source, "")

    def test_conflicting_main_draw_metadata_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.csv"
            source.write_text(
                "tourney_id,tourney_name,tourney_date,tourney_level,surface\n"
                "2010-540,Wimbledon,20100621,G,Grass\n"
                "2010-540,Wimbledon,20100621,G,Hard\n"
                "2010-414,Hamburg,20100719,500,Clay\n"
                "2010-414,Hamburg,20100719,500,Clay\n",
                encoding="utf-8",
            )
            with patch("scripts.qualifying_levels.fetch_csv", return_value=source):
                metadata = load_2010_event_metadata(tmp)
        self.assertEqual(metadata, {"2010-414": ("Hamburg", "20100719", "500", "Clay")})


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.inventory_event_metadata import corrected_inventory_row


class InventoryEventMetadataTest(unittest.TestCase):
    # Source draw, original date/surface, official tournament-week date/surface.
    CASES = (
        ("qualifying", "2025-8996", "Santiago", "250", "20250224", "Hard", "20250224", "Clay"),
        ("challenger", "2024-7740", "Nottingham", "C", "20240211", "Grass", "20240610", "Grass"),
        ("challenger", "2021-444", "Bratislava", "C", "20210613", "Clay", "20211108", "Hard"),
        ("challenger", "2023-444", "Bratislava 1", "C", "20230618", "Hard", "20231009", "Hard"),
        ("challenger", "2024-444", "Bratislava 1", "C", "20240616", "Hard", "20241028", "Hard"),
    )

    def test_verified_metadata_and_idempotence_for_each_edition(self):
        for source, event, name, level, date, surface, new_date, new_surface in self.CASES:
            with self.subTest(source=source, event=event):
                year = int(event[:4])
                row = dict(tourney_id=event, tourney_name=name, tourney_level=level,
                           tourney_date=date, surface=surface, winner_rank="37", loser_rank="84")
                original = row.copy()
                revised = corrected_inventory_row(row, source, year)
                self.assertEqual(revised, dict(original, tourney_date=new_date, surface=new_surface))
                self.assertEqual(row, original)
                self.assertIs(corrected_inventory_row(revised, source, year), revised)

    def test_checks_the_edition_and_original_metadata_together(self):
        row = dict(tourney_id="2021-444", tourney_name="Bratislava", tourney_level="C",
                   tourney_date="20210613", surface="Clay")
        for changes in ({"tourney_name": "Bratislava 1"}, {"tourney_level": "250"},
                        {"tourney_date": "20210614"}, {"surface": "Hard"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "2021-444"):
                corrected_inventory_row(dict(row, **changes), "challenger", 2021)
        with self.assertRaisesRegex(ValueError, "2021-444"):
            corrected_inventory_row(row, "challenger", 2024)

    def test_preserves_other_draws_with_the_same_city_or_event_id(self):
        row = dict(tourney_id="2025-8996", tourney_name="Santiago", tourney_level="250",
                   tourney_date="20250224", surface="Clay")
        self.assertIs(corrected_inventory_row(row, "tour", 2025), row)
        other = dict(row, tourney_id="2024-8996", tourney_date="20240226")
        self.assertIs(corrected_inventory_row(other, "qualifying", 2024), other)


if __name__ == "__main__":
    unittest.main()

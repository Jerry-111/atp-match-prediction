"""Regression coverage for reconciled Tour training labels and warmup history."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.baseline import load_matches
from scripts.prepare_data import build_model_rows
from scripts.reconcile_matches import DrawReconciliation


def example(event, number, round_name, winner, loser, score):
    return {
        "tourney_id": event, "tourney_level": "250", "tourney_date": event[:4] + "0701",
        "surface": "Clay", "round": round_name, "match_num": str(number),
        "winner_id": winner, "loser_id": loser, "winner_name": winner, "loser_name": loser,
        "winner_rank": "10", "loser_rank": "20", "score": score,
    }


class TourReconciliationTest(unittest.TestCase):
    def test_normal_loader_removes_bad_labels_and_history_before_modeling(self):
        false_first = example("2015-316", 1, "R32", "PD31", "R419", "7-6(7) 6-3")
        true_first = example("2015-316", 4, "R32", "PD31", "E688", "7-6(4) 6-3")
        true_final = dict(false_first, match_num="28", round="F")
        next_clay = dict(example("2016-999", 1, "R32", "PD31", "ZZ99", "6-1 6-1"),
                         tourney_date="20160301")
        rome = example("2021-416", 14, "R64", "P1", "P2", "6-2 6-2")
        stuttgart = example("2021-321", 5, "R32", "P3", "P4", "6-3 6-1")
        rows = {
            2015: [false_first, true_first, true_final],
            2016: [next_clay],
            2021: [rome, dict(rome, match_num="25"), stuttgart, dict(stuttgart, match_num="13")],
        }
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            seasons = range(2015, 2026)
            manifest = cache / "manifest.json"
            manifest.write_text(json.dumps({"files": [{"season": y} for y in seasons]}))
            for season in seasons:
                with (cache / f"{season}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=false_first)
                    writer.writeheader()
                    writer.writerows(rows.get(season, []))
            with patch("scripts.baseline.MANIFEST", manifest), patch(
                "scripts.baseline.fetch_csv", side_effect=lambda root, entry: root / f"{entry['season']}.csv"
            ):
                matches, audit = load_matches(cache)
                later_matches, _ = load_matches(cache, first_season=2016)
        self.assertEqual([audit[2015][k] for k in ("raw", "eligible_source", "spurious", "eligible")],
                         [3, 3, 1, 2])
        self.assertEqual([audit[2021][k] for k in ("raw", "eligible_source", "duplicate", "eligible")],
                         [4, 4, 2, 2])
        self.assertEqual([(m["tourney_id"], m["match_num"]) for m in later_matches],
                         [("2016-999", 1), ("2021-416", 14), ("2021-321", 5)])
        model_rows = build_model_rows(matches)
        self.assertEqual(len(model_rows), 3)
        self.assertEqual(model_rows[0]["player_a_id"], "PD31")
        self.assertEqual(model_rows[0]["surface_matches_a"], 2)

    def test_changed_duplicate_evidence_stops_reconciliation(self):
        verifier = DrawReconciliation()
        row = example("2021-416", 14, "R64", "P1", "P2", "6-2 6-2")
        verifier.reason("tour:2021-416:14", row)
        verifier.reason("tour:2021-416:25", dict(row, match_num="25", score="6-3 6-3"))
        with self.assertRaisesRegex(ValueError, "no longer verify as duplicates"):
            verifier.verify_season("tour", 2021)
        with self.assertRaisesRegex(ValueError, "official 2015 Båstad draw"):
            verifier.verify_season("tour", 2015)


if __name__ == "__main__":
    unittest.main()

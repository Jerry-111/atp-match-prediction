import math
import unittest

from scripts.metrics import score_probabilities


class ProbabilityMetricsTest(unittest.TestCase):
    def test_scores_match_ids_in_different_orders_and_tie_picks_a(self):
        scores = score_probabilities([("m1", "1"), ("m2", "0")], [("m2", 0.9), ("m1", 0.5)])
        self.assertEqual((scores["matches"], scores["correct"], scores["accuracy"]), (2, 1, 0.5))
        self.assertAlmostEqual(scores["mean_log_loss"], (-math.log(0.5) - math.log(0.1)) / 2)

    def test_certain_predictions(self):
        correct = score_probabilities([("a", 1), ("b", 0)], [("a", 1), ("b", 0)])
        self.assertEqual(correct["mean_log_loss"], 0)
        for outcome, forecast in [(0, 1), (1, 0)]:
            with self.subTest(outcome=outcome):
                scores = score_probabilities([("a", outcome)], [("a", forecast)])
                self.assertEqual(scores["mean_log_loss"], math.inf)

    def test_invalid_comparisons(self):
        cases = [
            ([], []),
            ([("a", 1)], [("b", 0.5)]),
            ([("a", 1)], [("a", 0.5), ("b", 0.5)]),
            ([("a", 1)], [("a", 0.5), ("a", 0.4)]),
            ([("a", 1), ("a", 1)], [("a", 0.5)]),
            ([("a", "2")], [("a", 0.5)]),
            ([("a", 1)], [("a", math.nan)]),
            ([("a", 1)], [("a", 1.01)]),
        ]
        for truth, forecast in cases:
            with self.subTest(truth=truth, forecast=forecast), self.assertRaises(ValueError):
                score_probabilities(truth, forecast)

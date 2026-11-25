"""Primary worked example plus independent point/game-path enumerations."""

from collections import defaultdict
from functools import lru_cache
import unittest

import numpy as np
import pandas as pd

from scripts.advanced_features import mirror_features
from scripts.paper_point_model import (build_paper_point_features, game_probability,
                                      match_probability, set_probability, tiebreak_probability)


def forward_tiebreak(a, b, first):
    """Independent forward point-mass propagation, including long deuce tails."""
    active = {(0, 0): 1.}
    won = 0.
    for point_number in range(300):
        next_active = defaultdict(float)
        server = first ^ (((point_number + 1) // 2) % 2)
        probability = a if server == 0 else 1 - b
        for (aw, bw), mass in active.items():
            for win, chance in ((True, probability), (False, 1 - probability)):
                aa, bb = aw + int(win), bw + int(not win)
                part = mass * chance
                if max(aa, bb) >= 7 and abs(aa - bb) >= 2:
                    if aa > bb:
                        won += part
                else:
                    next_active[aa, bb] += part
        active = next_active
        if sum(active.values()) < 1e-15:
            break
    if sum(active.values()) >= 1e-12:
        raise AssertionError("Independent tiebreak truncation did not converge")
    return won


def enumerate_set_paths(a, b, first):
    """Enumerate individual game-outcome paths, retaining next-set server."""
    result = defaultdict(float)
    holds = (game_probability(a), game_probability(b))
    tie = forward_tiebreak(a, b, first)

    def walk(aw, bw, server, mass):
        if aw == bw == 6:
            result[0, 1-first] += mass * tie
            result[1, 1-first] += mass * (1-tie)
        elif max(aw, bw) >= 6 and abs(aw-bw) >= 2:
            result[0 if aw > bw else 1, server] += mass
        else:
            chance = holds[0] if server == 0 else 1-holds[1]
            walk(aw+1, bw, 1-server, mass*chance)
            walk(aw, bw+1, 1-server, mass*(1-chance))

    walk(0, 0, first, 1.)
    return result


class PaperPointModelTest(unittest.TestCase):
    def test_primary_omalley_worked_example(self):
        # O'Malley2007 NESSIS slide18: serve .65, return .37; best-of3=.5985.
        self.assertAlmostEqual(match_probability(.65, .63, 3), .5985, delta=.00005)
        self.assertAlmostEqual(game_probability(.65), .8296446444954129, places=14)

    def test_neutral_swap_and_match_length(self):
        for p in (.2, .5, .65, .9):
            self.assertAlmostEqual(match_probability(p, p, 3), .5, places=13)
            self.assertAlmostEqual(match_probability(p, p, 5), .5, places=13)
        a = np.array([.52, .6, .7, .85])
        b = np.array([.64, .55, .6, .7])
        for best in (3, 5):
            np.testing.assert_allclose(match_probability(a, b, best) + match_probability(b, a, best), 1., atol=1e-13)
        self.assertGreater(match_probability(.7, .6, 5), match_probability(.7, .6, 3))
        self.assertLess(match_probability(.6, .7, 5), match_probability(.6, .7, 3))

    def test_monotonicity_endpoints_and_invalid_inputs(self):
        values = match_probability(np.linspace(.05, .95, 20), .65, 3)
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertTrue(0 <= match_probability(.99, .01, 5) <= 1)
        self.assertGreater(match_probability(.99, .01, 5), .999999)
        self.assertLess(match_probability(.01, .99, 5), 1e-8)
        self.assertEqual(game_probability(0), 0)
        self.assertEqual(game_probability(1), 1)
        for invalid in (0, 1, np.nan, np.inf, -.1):
            with self.assertRaises(ValueError):
                match_probability(invalid, .6, 3)
        with self.assertRaises(ValueError):
            match_probability(.6, .6, 4)

    def test_tiebreak_matches_independent_forward_point_chain(self):
        for a, b in ((.65, .63), (.45, .7), (.8, .75)):
            for first in (0, 1):
                self.assertAlmostEqual(tiebreak_probability(a, b, first), forward_tiebreak(a, b, first), places=13)

    def test_match_matches_independent_game_path_enumeration(self):
        a, b = .59, .67
        transitions = {first: enumerate_set_paths(a, b, first) for first in (0, 1)}
        self.assertAlmostEqual(sum(transitions[0].values()), 1., places=12)
        expected_set = .5 * sum(mass for first in (0, 1) for (winner, _), mass in transitions[first].items() if winner == 0)
        self.assertAlmostEqual(set_probability(a, b), expected_set, places=12)
        for best in (3, 5):
            needed = best//2 + 1

            @lru_cache(None)
            def match(aw, bw, server):
                if aw == needed:
                    return 1.
                if bw == needed:
                    return 0.
                return sum(mass * match(aw + int(winner == 0), bw + int(winner == 1), next_server)
                           for (winner, next_server), mass in transitions[server].items())

            expected = .5 * (match(0, 0, 0) + match(0, 0, 1))
            self.assertAlmostEqual(match_probability(a, b, best), expected, places=12)

    @staticmethod
    def fixture():
        target = pd.DataFrame({"season": [2010, 2023], "a_won": ["UNUSED", "UNUSED"]}, index=[7, 2])
        x = pd.DataFrame({"format_best_of_five": [0., 1.]}, index=target.index)
        for half in (365, 730):
            for scope in ("overall", "surface"):
                stem = f"adjpoints_h{half}_{scope}_serve_prob"
                x[stem + "_sum"] = [1.28, 1.30]
                x[stem + "_diff"] = [.02, -.10]
        return target, x

    def test_feature_exchange_format_and_unused_outcome_poisoning(self):
        target, x = self.fixture()
        p = build_paper_point_features(target, x)
        self.assertEqual(p.shape, (2, 12))
        self.assertTrue(all(c.endswith("_diff") for c in p))
        swapped = build_paper_point_features(target, mirror_features(x))
        np.testing.assert_allclose(p.to_numpy(), -swapped.to_numpy(), atol=1e-12)
        poisoned = target.assign(a_won=[0, 1], score="POISON")
        pd.testing.assert_frame_equal(p, build_paper_point_features(poisoned, x))
        self.assertAlmostEqual(p.iloc[0, 0], np.log(.5985315484924838 / (1-.5985315484924838)), places=12)
        with self.assertRaisesRegex(ValueError, "holdout"):
            build_paper_point_features(target.assign(season=2024), x)
        with self.assertRaisesRegex(ValueError, "indices"):
            build_paper_point_features(target, x.iloc[::-1])


if __name__ == "__main__":
    unittest.main()

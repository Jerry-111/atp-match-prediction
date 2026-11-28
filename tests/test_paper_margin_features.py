"""Primary-paper equation examples, chronology, source checks and symmetry."""

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.advanced_features import mirror_features
from scripts.paper_margin_features import (
    WeightedRatingState, _source_records, build_paper_margin_features, experience_k,
    paper_update, score_weights,
)


class PaperMarginFeaturesTest(unittest.TestCase):
    start = date(2023, 7, 3)

    def match(self, n, ago, winner="A", loser="B", score="6-0 6-0", surface="Hard"):
        day = self.start - timedelta(days=ago)
        a, b = sorted((winner, loser))
        inv = {"match_id": f"tour:T:{n}", "source": "tour", "season": day.year,
               "tourney_date": day.strftime("%Y%m%d"), "player_a_id": a, "player_b_id": b,
               "a_won": int(a==winner), "surface": surface}
        raw = {"tourney_id": "T", "match_num": n, "tourney_date": inv["tourney_date"],
               "winner_id": winner, "loser_id": loser, "surface": surface,
               "score": score, "best_of": "3"}
        return inv, (day.year, raw)

    def targets(self):
        return pd.DataFrame([{"season": 2023, "tourney_date": self.start.isoformat(),
                              "player_a_id": "A", "player_b_id": "B", "surface": "Hard",
                              "a_won": 1}], index=[42])

    def build(self, matches, targets=None):
        inv, raw = zip(*matches)
        return build_paper_margin_features(self.targets() if targets is None else targets,
                                           pd.DataFrame(inv), raw_rows=raw)

    def test_paper_table1_game_weights_and_equation3(self):
        # Angelini et al. accepted manuscript Table1, p6. Exact weights, not
        # the table's two-decimal display rounding. Case3 winner has fewer games.
        for score, p, weight in (("6-0 6-0", .6, 1), ("7-6 7-6", .6, 14/26),
                                  ("0-6 7-6 7-6", .95, 14/32)):
            actual, _ = score_weights(score, 3)
            self.assertAlmostEqual(actual, weight)
            # Choose ratings to produce the paper's listed probability exactly.
            diff = 400*np.log10(p/(1-p))
            delta = paper_update(1500+diff, 1500, 1, 30, actual)
            self.assertAlmostEqual(delta/experience_k(30), (1-p)*weight)
        # Table1 case4 is a defeat1-6,1-6; raw score is winner-oriented6-1,6-1.
        weight, _ = score_weights("6-1 6-1", 3)
        diff = 400*np.log10(.75/.25)
        self.assertAlmostEqual(paper_update(1500+diff,1500,0,30,weight)/experience_k(30), -.75*12/14)

    def test_set_appendix_tiebreak_and_long_advantage_scores(self):
        self.assertEqual(score_weights("6-4 4-6 6-3", 3)[1], 2/3)
        self.assertEqual(score_weights("6-4 4-6 6-3 6-2", 5)[1], 3/4)
        self.assertEqual(score_weights("7-6(5) 7-6(12)",3)[0],14/26)
        self.assertEqual(score_weights("6-4 3-6 6-7(7) 7-6(3) 70-68",5)[0],92/183)
        self.assertEqual(score_weights("6-4 4-6 6-3 2-6 13-12(3)",5)[1],3/5)
        for score in ("6-4 RET", "6-5 6-4", "6-4", "6-4 6-4 0-6", "", "0-6 0-6"):
            with self.assertRaises(ValueError):
                score_weights(score,3)

    def test_date_batch_freezes_ratings_and_experience(self):
        matches = [self.match(1,21),self.match(2,21,loser="C",score="7-6 7-6")]
        inv, raw = zip(*matches)
        records,_ = _source_records(pd.DataFrame(inv),None,raw)
        state = WeightedRatingState(records)
        cutoff = self.start-timedelta(days=21)
        actual,count=state.banks["overall","welo_games"]["A"].at(cutoff)
        self.assertAlmostEqual(actual,1500+experience_k(0)*.5*(1+14/26))
        self.assertEqual(count,2)
        self.assertEqual(experience_k(0),experience_k(1))
        self.assertLess(experience_k(2),experience_k(1))
        pd.testing.assert_frame_equal(self.build(matches),self.build(matches[::-1]))

    def test_cutoff_poisoning_and_target_label_independence(self):
        matches=[self.match(1,21),self.match(2,20),self.match(3,0)]
        before=self.build(matches)
        for _,(_,raw) in matches[1:]:raw["score"]="0-6 7-6 7-6"
        targets=self.targets();targets["a_won"]="POISON"
        pd.testing.assert_frame_equal(before,self.build(matches,targets))
        inv,raw=zip(*matches)
        inv=pd.concat([pd.DataFrame(inv),pd.DataFrame([
            {"source":"tour","season":2024,"a_won":"POISON"},
            {"source":"challenger","season":2023,"a_won":"POISON"}])])
        after=build_paper_margin_features(targets,inv,raw_rows=[*raw,(2025,{"POISON":True})])
        pd.testing.assert_frame_equal(before,after)

    def test_exchange_symmetry_and_unknown_players(self):
        matches=[self.match(1,40),self.match(2,21,winner="B",loser="C",score="6-4 6-4")]
        before=self.build(matches)
        self.assertEqual(before.shape,(1,22))
        targets=self.targets()
        targets[["player_a_id","player_b_id"]]=targets[["player_b_id","player_a_id"]].to_numpy()
        pd.testing.assert_frame_equal(self.build(matches,targets),mirror_features(before))
        targets[["player_a_id","player_b_id"]]=["NEW_A","NEW_B"]
        unknown=self.build(matches,targets)
        self.assertTrue(np.isfinite(unknown.to_numpy()).all())
        self.assertTrue((unknown.filter(like="_diff").to_numpy()==0).all())

    def test_source_winner_and_holdout_guards(self):
        inv,raw=self.match(1,21)
        inv["a_won"]=0
        with self.assertRaisesRegex(ValueError,"winner mismatch"):
            self.build([(inv,raw)])
        for season in (2008,2024,2025):
            targets=self.targets();targets["season"]=season
            with self.assertRaisesRegex(ValueError,"holdout is locked"):
                self.build([self.match(1,21)],targets)


if __name__=="__main__":
    unittest.main()

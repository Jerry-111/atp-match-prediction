"""Hand-check graph orientation and temporal protection for the paper family."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.paper_graph_features import (
    TemporalGraph, bonacich_scores, build_paper_graph_features, hyperbolic_weights,
)


class PaperGraphFeaturesTest(unittest.TestCase):
    day = date(2023,7,3)

    def match(self, number, ago, a="A", b="B", won=1, surface="Hard"):
        played=self.day-timedelta(days=ago)
        return {"match_id":f"tour:T:{number}","source":"tour","season":played.year,
                "tourney_date":played.strftime("%Y%m%d"),"player_a_id":a,"player_b_id":b,
                "a_won":won,"surface":surface}

    def targets(self):
        return pd.DataFrame([{"season":2023,"tourney_date":self.day.isoformat(),"player_a_id":"A",
                              "player_b_id":"B","surface":"Hard","a_won":1}],index=[42])

    def test_exact_core_two_player_hand_calculation(self):
        # W[loser,winner]: B defeats A four times; A defeats B once.
        # W.T*[1,2] = 2*[1,2]; B must receive the larger score.
        score=bonacich_scores([[0,4],[1,0]],regularization=0)
        np.testing.assert_allclose(score,np.array([1.,2.])/np.sqrt(5),atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(score)),1)
        np.testing.assert_allclose(hyperbolic_weights([0,365,730]),[1,.5,1/3])

    def test_regularized_chain_preserves_incoming_orientation_and_indirect_credit(self):
        # Both A and C have one win, but A beat B, who also defeated D.
        # Vertex order A B C D; credit must flow D->B->A and D->C.
        matrix=np.zeros((4,4))
        matrix[1,0]=1
        matrix[3,1]=1
        matrix[3,2]=1
        score=bonacich_scores(matrix)
        self.assertGreater(score[0],score[2])
        self.assertTrue(np.all(score>0))
        self.assertAlmostEqual(np.linalg.norm(score),1)
        permutation=[2,0,3,1]
        permuted=bonacich_scores(matrix[np.ix_(permutation,permutation)])
        np.testing.assert_allclose(permuted,score[permutation],atol=1e-12)

    def test_cutoff_inclusive_and_future_labels_roster_and_target_poisoning(self):
        inv=pd.DataFrame([self.match(1,21),self.match(2,20,won=0),self.match(3,0,won=0)])
        expected=build_paper_graph_features(self.targets(),inv)
        self.assertGreater(expected.loc[42,"paper_graph_bscore_diff"],0)
        self.assertAlmostEqual(expected.loc[42,"paper_graph_win_strength_log_diff"],np.log(2))
        inv.loc[[1,2],"a_won"]=1
        inv.loc[[1,2],"player_b_id"]="FUTURE_NEW_PLAYER"
        inv=pd.concat([inv,pd.DataFrame([
            {"source":"tour","season":2024,"a_won":"SEALED"},
            {"source":"tour","season":2025,"a_won":"SEALED"},
            {"source":"challenger","season":2023,"a_won":"IGNORE"},
        ])],ignore_index=True)
        targets=self.targets();targets["a_won"]="NOT_A_FEATURE"
        pd.testing.assert_frame_equal(expected,build_paper_graph_features(targets,inv))

    def test_same_date_permutation_surface_separation_and_player_swap(self):
        inv=pd.DataFrame([self.match(1,21),self.match(2,21,b="C"),
                          self.match(3,40,a="B",b="C",won=0,surface="Grass")])
        targets=self.targets()
        expected=build_paper_graph_features(targets,inv)
        pd.testing.assert_frame_equal(expected,build_paper_graph_features(targets,inv.iloc[::-1]))
        targets[["player_a_id","player_b_id"]]=targets[["player_b_id","player_a_id"]].to_numpy()
        odd=[c for c in expected if c.endswith("_diff")]
        expected[odd]=-expected[odd]
        pd.testing.assert_frame_equal(expected,build_paper_graph_features(targets,inv))
        cutoff=self.day.toordinal()-21
        state=TemporalGraph(inv)
        grass=state.snapshot(cutoff,"Grass")
        self.assertNotIn("A",grass.players)
        self.assertEqual(set(grass.players),{"B","C"})

    def test_unseen_players_have_neutral_finite_prior(self):
        inv=pd.DataFrame([self.match(1,21)])
        targets=self.targets();targets[["player_a_id","player_b_id"]]=[["X","Y"]]
        features=build_paper_graph_features(targets,inv)
        self.assertEqual(features.shape,(1,12))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        np.testing.assert_array_equal(features.to_numpy(),np.zeros((1,12)))
        targets["surface"]="Grass"
        features=build_paper_graph_features(targets,inv)
        self.assertEqual(features.loc[42,"paper_surface_graph_observed_sum"],0)

    def test_holdout_duplicate_and_invalid_inputs_rejected(self):
        inv=pd.DataFrame([self.match(1,21)])
        targets=self.targets();targets["season"]=2024
        with self.assertRaisesRegex(ValueError,"holdout is locked"):
            build_paper_graph_features(targets,inv)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            TemporalGraph(pd.concat([inv,inv]))
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            hyperbolic_weights([-1])
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            bonacich_scores([[0,-1],[1,0]])


if __name__=="__main__":
    unittest.main()

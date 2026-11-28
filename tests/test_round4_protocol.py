"""Independent controls for the paper round's selection and frozen inference."""

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.train_models import SymmetricClassifier, mirror_features
from scripts.train_round4 import (
    candidate_specs, choose_procedure, predict_round4, raw_probability,
)


class RoundFourProtocolTests(unittest.TestCase):
    def setUp(self):
        self.y = np.tile([0, 1], 3)
        self.seasons = np.repeat([2020, 2021, 2022], 2)

    def test_gate_selects_on_early_years_and_ignores_later_outcomes(self):
        specs = [{"name": "early", "family": "one"},
                 {"name": "late", "family": "two"}]
        predictions = {"early": np.array([.1, .9, .1, .9, .8, .2]),
                       "late": np.array([.45, .55, .45, .55, .01, .99])}
        reference = np.tile([.4, .6], 3)
        expected = choose_procedure(self.y, predictions, specs, self.seasons, reference)
        self.assertEqual(expected["early_selected"], "early")
        self.assertEqual(expected["procedure"], "retain_round3")
        poisoned = choose_procedure(np.r_[self.y, [1, 0, 1, 0]],
            {"early": np.r_[predictions["early"], [0, 1, 0, 1]],
             "late": np.r_[predictions["late"], [1, 0, 1, 0]]},
            specs, np.r_[self.seasons, [2023, 2023, 2024, 2025]],
            np.r_[reference, [0, 1, 0, 1]])
        self.assertEqual(expected, poisoned)

    def test_gate_uses_supplied_current_reference_and_threshold(self):
        specs = [{"name": "only", "family": "one"}]
        reference = np.tile([.4, .6], 3)
        for gain, decision in [(.000999, "retain_round3"), (.001001, "adopt_paper_model")]:
            correct = .6 * np.exp(gain)
            p = np.where(self.y == 1, correct, 1-correct)
            result = choose_procedure(self.y, {"only": p}, specs, self.seasons, reference)
            self.assertEqual(result["procedure"], decision)
            self.assertAlmostEqual(result["absolute_improvement"], gain)
        # A candidate can beat an obsolete weaker baseline while losing to the
        # current round-three reference. The latter must control the decision.
        candidate = np.tile([.3, .7], 3)
        current = np.tile([.25, .75], 3)
        obsolete = np.tile([.4, .6], 3)
        self.assertEqual(choose_procedure(self.y, {"only": candidate}, specs,
            self.seasons, current)["procedure"], "retain_round3")
        self.assertEqual(choose_procedure(self.y, {"only": candidate}, specs,
            self.seasons, obsolete)["procedure"], "adopt_paper_model")
        changed_early = current.copy()
        changed_early[:4] = [0, 1, 0, 1]
        self.assertEqual(choose_procedure(self.y, {"only": candidate}, specs,
            self.seasons, current), choose_procedure(self.y, {"only": candidate},
            specs, self.seasons, changed_early))

    def test_raw_elo_and_point_logits_have_distinct_physical_scales(self):
        x = pd.DataFrame({"elo_diff": [-400., 0., 400.],
                          "point_logit_diff": [-np.log(9.), 0., np.log(9.)]})
        elo = {"kind": "raw_elo", "columns": ["elo_diff"]}
        point = {"kind": "raw_logit", "columns": ["point_logit_diff"]}
        np.testing.assert_allclose(raw_probability(elo, x), [1/11, .5, 10/11], atol=1e-14)
        np.testing.assert_allclose(raw_probability(point, x), [.1, .5, .9], atol=1e-14)
        for spec in (elo, point):
            np.testing.assert_allclose(raw_probability(spec, x) +
                                       raw_probability(spec, mirror_features(x)), 1.)

    def test_declared_family_comparisons_preserve_schema_and_raw_controls(self):
        groups = {
            "base": ["log_rank_diff", "elo_k32_diff", "log_rank_mean",
                     "format_best_of_five", "format_log_rank_diff", "surface_is_clay"],
            "margin": ["paper_overall_welo_games_diff", "paper_overall_welo_sets_diff",
                       "paper_overall_elo_diff", "paper_mean"],
            "graph": ["paper_graph_bscore_diff", "paper_graph_observed_sum"],
            "point": [f"paperpoint_h{half}_{scope}_match_logit_diff"
                      for half in (365, 730) for scope in ("overall", "surface")],
            "profile": ["profile_age30_diff", "profile_age_mean", "profile_points_diff",
                        "profile_log_points_diff", "profile_points_missing_diff"],
        }
        x = pd.DataFrame(columns=sum(groups.values(), []))
        specs = candidate_specs(x, groups)
        self.assertEqual(len(specs), 51)
        self.assertEqual(len({s["name"] for s in specs}), 51)
        self.assertEqual(sum(s["kind"] == "raw_elo" for s in specs), 3)
        self.assertEqual(sum(s["kind"] == "raw_logit" for s in specs), 4)
        for spec in specs:
            self.assertEqual(len(spec["columns"]), len(set(spec["columns"])))
            self.assertTrue(set(spec["columns"]).issubset(x))
            if spec["kind"] == "logistic":
                self.assertTrue(all(c.endswith("_diff") for c in spec["columns"]))
            for group in ("margin", "graph", "point", "profile"):
                if spec["family"] == "add_" + group:
                    self.assertTrue(set(spec["columns"]).issubset(groups["base"] + groups[group]))
                if spec["family"] == "without_" + group:
                    self.assertFalse(set(spec["columns"]) & set(groups[group]))
            if spec["family"] in ("add_age", "add_points"):
                profile_columns = set(spec["columns"]) & set(groups["profile"])
                self.assertTrue(profile_columns)
                self.assertTrue(all(c.startswith("profile_age") == (spec["family"] == "add_age")
                                    for c in profile_columns))
        for family in ("add_age", "add_points"):
            self.assertEqual(sum(s["family"] == family for s in specs), 4)

    def _bundle(self):
        x = pd.DataFrame({"log_rank_diff": [-2., -1., -.3, .3, 1., 2.],
                          "elo_k32_diff": [-400., -100., -10., 10., 100., 400.],
                          "point_logit_diff": [-2., -1., -.2, .2, 1., 2.]},
                         index=[9, 4, 2, 7, 1, 3])
        spec = {"kind": "logistic", "columns": ["log_rank_diff"], "params": {"C": .1}}
        model = SymmetricClassifier(spec).fit(x, np.array([0, 0, 1, 0, 1, 1]))
        first = {"feature_names": ["log_rank_diff"], "models": {"linear": model},
                 "raw_elo_columns": [], "frozen": {"selected_by_family": {"linear": "linear"},
                 "ensemble_components": ["linear"], "ensemble_weights": [1.],
                 "advanced_champion": "linear"}}
        third = {"feature_names": ["log_rank_diff"], "models": {"prior": model},
                 "round1_bundle": first, "round2_bundle": None,
                 "round1_aliases": {"older": "linear"}}
        bundle = {"feature_names": list(x), "models": {"paper": model},
                  "raw_specs": {"elo": {"kind": "raw_elo", "columns": ["elo_k32_diff"]},
                                "point": {"kind": "raw_logit", "columns": ["point_logit_diff"]}},
                  "round3_bundle": third, "previous_aliases": {"round3_recommended": "prior"}}
        return bundle, x

    def test_nested_frozen_serialization_preserves_order_symmetry_and_labels_are_ignored(self):
        bundle, x = self._bundle()
        expected = predict_round4(bundle, x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            reordered = x[list(reversed(x.columns))].assign(a_won="POISON", score="POISON")
            pd.testing.assert_frame_equal(expected, predict_round4(loaded, reordered))
            np.testing.assert_allclose(expected + predict_round4(loaded, mirror_features(x)),
                                       1., atol=1e-14)
        self.assertEqual(list(expected.index), list(x.index))
        self.assertEqual(list(expected), ["paper", "elo", "point", "round3_recommended"])

    def test_frozen_inference_rejects_dropped_duplicate_or_nonfinite_features(self):
        bundle, x = self._bundle()
        invalid = [x.iloc[:0], x.drop(columns="point_logit_diff"),
                   pd.concat([x, x[["log_rank_diff"]]], axis=1),
                   x.assign(point_logit_diff=np.nan), x.assign(point_logit_diff=np.inf)]
        for frame in invalid:
            with self.subTest(columns=list(frame), rows=len(frame)):
                with self.assertRaises(ValueError):
                    predict_round4(bundle, frame)


if __name__ == "__main__":
    unittest.main()

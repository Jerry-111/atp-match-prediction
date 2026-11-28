"""Training breadth and format controls preserve the evaluation experiment."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.train_models import mirror_features
from scripts.train_round3 import (candidate_specs, choose_procedure, predict_round3,
                                 training_masks, verify_unchanged_cohort)


class FixedModel:
    def predict_proba(self, x):
        return expit(x.log_rank_diff.to_numpy())


class RoundThreeProtocolTest(unittest.TestCase):
    def test_registry_has_matched_start_learner_and_format_cells(self):
        specs = candidate_specs(["log_rank_diff", "elo_k32_diff", "log_rank_mean",
                                 "format_best_of_five", "format_log_rank_diff"])
        self.assertEqual(len(specs), 32)
        self.assertEqual(len({s["name"] for s in specs}), 32)
        for start in (2009, 2016):
            for kind in ("logistic", "odd_spline"):
                for flag in (False, True):
                    cell = [s for s in specs if s["train_start"] == start and s["kind"] == kind and s["include_format"] == flag]
                    self.assertEqual(len(cell), 4)
                    for spec in cell:
                        self.assertEqual("format_log_rank_diff" in spec["columns"], flag)

    def test_training_breadth_changes_fit_rows_but_not_scored_rows(self):
        rows = pd.DataFrame({"season": [2009, 2015, 2016, 2019, 2020, 2020, 2021],
                             "tourney_date": ["2009-01-01", "2015-01-01", "2016-01-01", "2019-12-20",
                                              "2020-01-06", "2020-05-04", "2021-01-04"]})
        long, valid_long = training_masks(rows, 2020, 2009)
        short, valid_short = training_masks(rows, 2020, 2016)
        self.assertEqual(list(rows.index[long]), [0, 1, 2])
        self.assertEqual(list(rows.index[short]), [2])
        pd.testing.assert_series_equal(valid_long, valid_short)
        self.assertEqual(list(rows.index[valid_long]), [4, 5])

    def test_gate_uses_early_choice_not_the_best_2022_candidate(self):
        y = np.tile([0, 1], 3)
        seasons = np.repeat([2020, 2021, 2022], 2)
        specs = [{"name": "early", "family": "one"}, {"name": "late", "family": "two"}]
        predictions = {"early": np.array([.1, .9, .1, .9, .8, .2]),
                       "late": np.array([.45, .55, .45, .55, .01, .99])}
        reference = np.tile([.4, .6], 3)
        result = choose_procedure(y, predictions, specs, seasons, reference)
        self.assertEqual(result["early_selected"], "early")
        self.assertEqual(result["procedure"], "retain_round1")
        poisoned = choose_procedure(np.r_[y, [1, 0]],
            {"early": np.r_[predictions["early"], [0, 1]], "late": np.r_[predictions["late"], [1, 0]]},
            specs, np.r_[seasons, [2023, 2023]], np.r_[reference, [0, 1]])
        self.assertEqual(result, poisoned)

    def test_minimum_improvement_gate(self):
        y = np.tile([0, 1], 3)
        seasons = np.repeat([2020, 2021, 2022], 2)
        specs = [{"name": "only", "family": "one"}]
        reference = np.tile([.4, .6], 3)
        for gain, decision in [(.0005, "retain_round1"), (.002, "adopt_history_model")]:
            correct = .6 * np.exp(gain)
            p = np.where(y == 1, correct, 1 - correct)
            result = choose_procedure(y, {"only": p}, specs, seasons, reference)
            self.assertEqual(result["procedure"], decision)

    def test_existing_values_and_order_are_verified_before_cache_reuse(self):
        original = pd.DataFrame({"match_id": ["a", "b"], "a_won": [0, 1], "season": [2020, 2023]})
        expanded = pd.concat([pd.DataFrame({"match_id": ["older"], "a_won": [1], "season": [2009]}), original], ignore_index=True)
        features = pd.DataFrame({"log_rank_diff": [9., 1., 2.], "format_best_of_five": [0., 1., 0.]})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            features.loc[[1, 2], ["log_rank_diff"]].to_csv(folder / "development_features.csv", index=False)
            self.assertEqual(verify_unchanged_cohort(original, expanded, features, folder), 0.)
            changed = features.copy()
            changed.loc[2, "log_rank_diff"] += .01
            with self.assertRaisesRegex(ValueError, "feature values changed"):
                verify_unchanged_cohort(original, expanded, changed, folder)

    def test_frozen_predictions_are_symmetric_and_ignore_labels(self):
        first = {"feature_names": ["log_rank_diff", "elo_k32_diff"], "models": {"linear": FixedModel()},
                 "raw_elo_columns": ["elo_k32_diff"],
                 "frozen": {"selected_by_family": {"linear": "linear", "raw": "raw_elo_k32_diff"},
                            "ensemble_components": ["linear", "raw_elo_k32_diff"], "ensemble_weights": [.5, .5],
                            "advanced_champion": "oof_convex_ensemble"}}
        bundle = {"feature_names": ["log_rank_diff", "elo_k32_diff"], "models": {"new": FixedModel()},
                  "round1_bundle": first, "round2_bundle": None,
                  "round1_aliases": {"round1_ensemble": "oof_convex_ensemble"}}
        x = pd.DataFrame({"log_rank_diff": [-1., 0., 2.], "elo_k32_diff": [-100., 30., 140.]}, index=[8, 2, 5])
        p = predict_round3(bundle, x)
        pd.testing.assert_frame_equal(p, predict_round3(bundle, x.assign(a_won="POISON")))
        np.testing.assert_allclose(p + predict_round3(bundle, mirror_features(x)), 1., atol=1e-14)
        self.assertEqual(list(p.index), [8, 2, 5])


if __name__ == "__main__":
    unittest.main()

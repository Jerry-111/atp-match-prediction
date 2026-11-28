"""Accuracy-first selection still requires time separation and frozen weights."""
import copy
import json
from math import comb
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.train_models import mirror_features
from scripts.train_round5 import (
    REFERENCE, accuracy, accuracy_interval, candidate_specs, choose_procedure,
    load_previous, predict_round5, select_accuracy, selection_probability, simplex_weights,
)


class ColumnProbability:
    def __init__(self, column):
        self.column = column

    def predict_proba(self, x):
        return expit(x[self.column].to_numpy())


def specification(name, family=None, method=None):
    return {"name": name, "family": family or name, "method": method or name}


class RoundFiveProtocolTests(unittest.TestCase):
    def test_simplex_grid_is_complete_nonnegative_and_includes_vertices(self):
        for n in (1, 2, 4):
            grid = np.array(list(simplex_weights(n)))
            self.assertEqual(len(grid), comb(10+n-1, n-1))
            self.assertEqual(len({tuple(w) for w in grid}), len(grid))
            self.assertTrue(np.all(grid >= 0))
            np.testing.assert_allclose(grid.sum(axis=1), 1., atol=1e-15)
            for vertex in np.eye(n):
                self.assertTrue(np.any(np.all(grid == vertex, axis=1)))
        with self.assertRaises(ValueError):
            list(simplex_weights(0))

    def test_accuracy_winner_outside_probability_guard_is_ineligible(self):
        y = np.array([0, 1, 0, 1])
        oof = {REFERENCE: np.array([.6, .4, .1, .9]),
               "overconfident_errors": np.array([.49, .51, .49, .51])}
        specs = [specification(name) for name in oof]
        self.assertGreater(accuracy(y, oof["overconfident_errors"]), accuracy(y, oof[REFERENCE]))
        selected = select_accuracy(y, oof, specs, np.ones(4, bool))
        self.assertEqual(selected["choice"], REFERENCE)
        self.assertEqual(selected["eligible_models"], 1)
        self.assertNotIn("overconfident_errors", selected["selected_by_family"])

    def test_accuracy_ties_use_loss_then_stable_name(self):
        y = np.array([0, 1, 0, 1])
        oof = {REFERENCE: np.array([.2, .8, .2, .8]),
               "b": np.array([.1, .9, .1, .9]), "a": np.array([.1, .9, .1, .9])}
        specs = [specification(name, method="one") for name in oof]
        result = select_accuracy(y, oof, specs, np.ones(4, bool))
        self.assertEqual(result["best_single"], "a")
        self.assertEqual(result["choice"], "a")

    def test_interior_blend_can_win_and_selection_ignores_unselected_rows(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        oof = {REFERENCE: np.array([.6, .9, .2, .9, .2, .9]),
               "other": np.array([.2, .9, .6, .9, .7, .3])}
        specs = [specification(name) for name in oof]
        mask = np.array([True]*4+[False]*2)
        result = select_accuracy(y, oof, specs, mask)
        self.assertEqual(result["choice"], "accuracy_grid_blend")
        np.testing.assert_allclose(result["weights"], [.5, .5])
        self.assertEqual(result["selected_accuracy"], 1.)
        changed_y = y.copy(); changed_y[4:] = 1-changed_y[4:]
        changed = {name: p.copy() for name, p in oof.items()}
        for p in changed.values(): p[4:] = 1-p[4:]
        self.assertEqual(result, select_accuracy(changed_y, changed, specs, mask))

    def test_temporal_gate_requires_both_accuracy_and_log_loss_conditions(self):
        y = np.tile([0, 1], 300)
        seasons = np.repeat([2020, 2021, 2022], 200)
        correct = np.tile(np.r_[np.ones(150, bool), np.zeros(50, bool)], 3)
        ref = np.where(correct, np.where(y == 1, .6, .4), np.where(y == 1, .4, .6))
        for mode, expected in (("pass", "adopt_accuracy_model"),
                               ("no_accuracy_gain", "retain_round4"),
                               ("bad_loss", "retain_round4")):
            candidate = ref.copy()
            fixes = np.r_[np.arange(150, 160), np.arange(350, 360)]
            candidate[fixes] = np.where(y[fixes] == 1, .6, .4)
            if mode != "no_accuracy_gain":
                fixes = np.arange(550, 552)
                candidate[fixes] = np.where(y[fixes] == 1, .6, .4)
            if mode == "bad_loss":
                wrong = np.arange(552, 600)
                candidate[wrong] = np.where(y[wrong] == 1, .001, .999)
            oof = {REFERENCE: ref, "new": candidate}
            specs = [specification(name) for name in oof]
            result = choose_procedure(y, oof, specs, seasons)
            self.assertEqual(result["early_selection"]["components"], ["new"])
            self.assertEqual(result["procedure"], expected)
            extended = choose_procedure(np.r_[y, [1, 0]],
                {name: np.r_[p, [0., 1.]] for name, p in oof.items()}, specs,
                np.r_[seasons, [2023, 2024]])
            self.assertEqual(result, extended)

    def test_registry_contains_75_new_settings_with_matched_information_groups(self):
        groups = {"core_age": ["log_rank_diff", "elo_k32_diff", "log_rank_mean",
                                "profile_age_diff", "profile_age_mean"],
                  "multilevel": ["multilevel_strength_diff", "multilevel_count_sum"],
                  "style": ["style_first_serve_diff", "style_count_sum"]}
        columns = sum(groups.values(), [])+["old_paper_graph_diff"]
        specs = candidate_specs(pd.DataFrame(columns=columns), groups)
        self.assertEqual(len(specs), 75)
        self.assertEqual(len({s["name"] for s in specs}), 75)
        self.assertEqual(json.loads(json.dumps(specs)), specs)
        self.assertEqual(sum(s["kind"].startswith("r5_") for s in specs), 24)
        for spec in specs:
            self.assertEqual(len(spec["columns"]), len(set(spec["columns"])))
            self.assertTrue(set(spec["columns"]).issubset(columns))
            if spec["family"].startswith("core_"):
                self.assertTrue(set(spec["columns"]).issubset(groups["core_age"]))
            if spec["family"].startswith("multilevel_"):
                self.assertFalse(set(spec["columns"]) & set(groups["style"]))
            if spec["family"].startswith("style_"):
                self.assertFalse(set(spec["columns"]) & set(groups["multilevel"]))

    def test_previous_oof_maps_to_current_frozen_names_and_checks_identity(self):
        rows = pd.DataFrame({"match_id": ["a", "b", "c", "later"],
                             "season": [2020, 2021, 2022, 2023], "a_won": [0, 1, 0, 1]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = root/"results/round4"; previous.mkdir(parents=True)
            third = root/"results/round3"; third.mkdir(parents=True)
            (previous/"frozen_selection.json").write_text(json.dumps({"recommendation": "r4_selected"}))
            (third/"frozen_selection.json").write_text(json.dumps({
                "selected_by_family": {"history2009_linear_format1": "r3_linear"}}))
            fourth_oof = rows.iloc[:3].assign(r4_selected=[.2, .7, .4], without_graph_spline_C0_01=0.)
            fourth_oof = fourth_oof.drop(columns="without_graph_spline_C0_01")
            fourth_oof["without_graph_spline_C0.01"] = [.3, .6, .2]
            fourth_oof.to_csv(previous/"oof_predictions.csv", index=False)
            third_oof = rows.iloc[:3].assign(r3_linear=[.4, .8, .3])
            third_oof.to_csv(third/"oof_predictions.csv", index=False)
            with patch("scripts.train_round5.ROOT", root):
                predictions, aliases, specs = load_previous(rows, previous)
                np.testing.assert_array_equal(predictions[REFERENCE], [.2, .7, .4])
                np.testing.assert_array_equal(predictions["previous_accuracy_comparator"], [.4, .8, .3])
                self.assertEqual(aliases[REFERENCE], "r4_selected")
                self.assertEqual(aliases["previous_accuracy_comparator"], "round3_linear")
                self.assertEqual(len(specs), 3)
                third_oof.iloc[::-1].to_csv(third/"oof_predictions.csv", index=False)
                with self.assertRaisesRegex(ValueError, "identity"):
                    load_previous(rows, previous)

    def bundle(self):
        model = ColumnProbability("score_diff")
        first = {"feature_names": ["score_diff"], "models": {"linear": model}, "raw_elo_columns": [],
                 "frozen": {"selected_by_family": {"linear": "linear"},
                            "ensemble_components": ["linear"], "ensemble_weights": [1.],
                            "advanced_champion": "linear"}}
        third = {"feature_names": ["score_diff"], "models": {"prior": model},
                 "round1_bundle": first, "round2_bundle": None,
                 "round1_aliases": {"round1_best_single": "linear", "rank_reference": "linear"}}
        fourth = {"feature_names": ["score_diff"], "models": {"r4_selected": model}, "raw_specs": {},
                  "round3_bundle": third, "previous_aliases": {"round3_linear": "prior"}}
        fifth = {"feature_names": ["score_diff", "new_score_diff"],
                 "models": {"new": ColumnProbability("new_score_diff")}, "previous_bundle": fourth,
                 "previous_aliases": {REFERENCE: "r4_selected", "old_linear": "round3_linear"},
                 "frozen": {"pooled_selection": {"components": ["new", REFERENCE], "weights": [.4, .6]}}}
        x = pd.DataFrame({"score_diff": [-1., .3, 1.4], "new_score_diff": [.2, -.6, 2.]}, index=[9, 4, 7])
        return fifth, x

    def test_nested_inference_reload_preserves_simplex_orientation_and_row_order(self):
        bundle, x = self.bundle()
        expected = predict_round5(bundle, x)
        np.testing.assert_allclose(expected.accuracy_grid_blend,
                                   .4*expit(x.new_score_diff)+.6*expit(x.score_diff))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"frozen.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            reordered = x[list(reversed(x.columns))].assign(a_won="POISON", score="POISON")
            pd.testing.assert_frame_equal(expected, predict_round5(loaded, reordered))
            np.testing.assert_allclose(expected+predict_round5(loaded, mirror_features(x)), 1., atol=1e-14)
        self.assertEqual(list(expected.index), [9, 4, 7])

    def test_malformed_frozen_weights_are_rejected_even_if_probabilities_look_valid(self):
        probabilities = {"a": np.array([.4, .6]), "b": np.array([.45, .55])}
        cases = [(["a", "b"], [1.2, -.2]), (["a", "b"], [.2, .3]),
                 (["a", "b"], [1.]), (["a", "a"], [.5, .5]),
                 (["missing"], [1.]), (["a"], [np.nan]), ([], [])]
        for components, weights in cases:
            with self.subTest(components=components, weights=weights):
                with self.assertRaises(ValueError):
                    selection_probability({"components": components, "weights": weights}, probabilities)
        bundle, x = self.bundle()
        bundle["frozen"]["pooled_selection"] = {"components": ["new"], "weights": [.5]}
        with self.assertRaises(ValueError):
            predict_round5(bundle, x)

    def test_accuracy_change_counts_new_errors_as_well_as_corrections(self):
        rows = pd.DataFrame({"a_won": [0, 1, 0, 1], "tourney_date": ["2022-01-01"]*2+["2022-02-01"]*2})
        before = np.array([.6, .8, .2, .4])
        after = np.array([.4, .8, .8, .6])
        interval = accuracy_interval(rows, before, after, repeats=100)
        self.assertEqual(interval["corrected"], 2)
        self.assertEqual(interval["newly_wrong"], 1)
        self.assertEqual(interval["net_correct"], 1)
        self.assertEqual(interval["accuracy_improvement"], .25)
        self.assertEqual(interval["date_blocks"], 2)


if __name__ == "__main__":
    unittest.main()

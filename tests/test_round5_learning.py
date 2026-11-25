"""Check the new learning objective without fitting or scoring 2023 targets."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.round5_learning import (
    AccuracyMarginClassifier, OddFeatureMap, calibration_masks, candidate_specs,
    make_model, positive_calibration_slope,
)
from scripts.train_models import mirror_features


class RecordingClassifier(AccuracyMarginClassifier):
    def _fit_margin(self, x, y, seasons):
        fitted = super()._fit_margin(x, y, seasons)
        if not hasattr(self, "fit_calls"):
            self.fit_calls = []
        self.fit_calls.append({"index": list(x.index), "coefficients": fitted[1].coef_.copy(),
                               "feature_map": fitted[0]})
        return fitted


class RoundFiveLearningTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(2026)
        n = 90
        rank = rng.normal(size=n)
        x = pd.DataFrame({
            "log_rank_diff": rank, "elo_k32_diff": 100*rank+rng.normal(size=n)*40,
            "profile_age_diff": rng.normal(size=n)*5,
            "profile_age30_diff": rng.normal(size=n)*3,
            "profile_age_interval_diff": rng.normal(size=n)*2,
            "profile_age_mean": 28+rng.normal(size=n)*3,
            "log_rank_mean": 4+rng.normal(size=n)*.5,
            "overall_inactivity_log_diff": rng.normal(size=n),
            "overall_inactivity_log_sum": 6+rng.normal(size=n),
            "surface_is_clay": (np.arange(n)%3 == 0).astype(float),
            "format_best_of_five": (np.arange(n)%4 == 0).astype(float),
            "new_history_diff": rng.normal(size=n),
        }, index=np.arange(n)*2+1)
        y = (rank+rng.normal(size=n)*.8 > 0).astype(int)
        seasons = np.repeat([2017, 2018, 2019], 30)
        dates = np.repeat(["2017-06-01", "2018-06-01", "2019-01-07"], 30)
        dates[59] = "2018-12-30"
        return x, y, seasons, dates

    def spec(self, columns, design="linear", head="logistic", width=0):
        return next(s for s in candidate_specs(columns)
                    if s.get("design") == design and s["kind"] == "r5_"+head
                    and s["params"]["components"] == width)

    def test_registry_is_bounded_serializable_and_contains_matched_objectives(self):
        x, _, _, _ = self.fixture()
        specs = candidate_specs(x.columns)
        self.assertEqual(len(specs), 26)
        self.assertEqual(len({s["name"] for s in specs}), 26)
        self.assertEqual(json.loads(json.dumps(specs)), specs)
        self.assertEqual(sum(s["kind"] == "odd_spline" for s in specs), 2)
        for head in ("r5_logistic", "r5_squared_hinge"):
            self.assertEqual(sum(s["kind"] == head for s in specs), 12)
        for spec in specs:
            self.assertTrue(set(spec["columns"]).issubset(x.columns))
            self.assertEqual(len(spec["columns"]), len(set(spec["columns"])))
            self.assertIn("new_history_diff", spec["columns"])
            if spec.get("design") == "rff":
                self.assertNotIn("new_history_diff", spec["kernel_columns"])
        with self.assertRaises(ValueError):
            candidate_specs(["not_odd"])

    def test_all_designs_are_exactly_odd_and_fit_statistics_ignore_prediction_rows(self):
        x, _, _, _ = self.fixture()
        for design, width in (("linear", 0), ("context", 0), ("rff", 256), ("rff", 512)):
            spec = self.spec(x.columns, design=design, width=width)
            mapping = OddFeatureMap(spec).fit(x.iloc[:60])
            expected = mapping.transform(x.iloc[60:])
            np.testing.assert_allclose(mapping.transform(mirror_features(x.iloc[60:])),
                                       -expected, atol=1e-13, rtol=1e-13)
            before = mapping.odd_scale_.copy()
            mapping.transform(x.iloc[60:]*1e6)
            np.testing.assert_array_equal(mapping.odd_scale_, before)
            np.testing.assert_array_equal(mapping.transform(x.iloc[60:]), expected)
            # Exchange fixes an identical-player input; every learned design
            # column must vanish, including nonlinear and context columns.
            tied = x.iloc[[0]].copy()
            tied.loc[:, [c for c in tied if c.endswith("_diff")]] = 0.
            np.testing.assert_allclose(mapping.transform(tied), 0., atol=1e-14)

    def test_inner_calibration_split_is_chronological_and_purged(self):
        _, _, seasons, dates = self.fixture()
        earlier, later, meta = calibration_masks(seasons, dates)
        self.assertEqual(meta["calibration_year"], 2019)
        self.assertEqual(np.flatnonzero(later).tolist(), list(range(60, 90)))
        self.assertEqual(np.flatnonzero(earlier).tolist(), list(range(59)))
        self.assertEqual(meta["inner_training_cutoff"], "2018-12-17T00:00:00")
        embargo, _, info = calibration_masks(seasons)
        self.assertEqual(np.flatnonzero(embargo).tolist(), list(range(30)))
        self.assertIn("whole-year", info["policy"])
        with self.assertRaises(ValueError):
            calibration_masks(np.repeat(2019, 3))

    def test_hinge_calibration_uses_scores_from_earlier_fit_then_final_refits(self):
        x, y, seasons, dates = self.fixture()
        spec = self.spec(x.columns, design="context", head="squared_hinge")
        model = RecordingClassifier(spec).fit(x, y, seasons, dates=dates)
        self.assertEqual(len(model.fit_calls), 2)
        self.assertEqual(model.fit_calls[0]["index"], list(x.index[:59]))
        self.assertEqual(model.fit_calls[1]["index"], list(x.index))
        changed = y.copy(); changed[60:] = 1-changed[60:]
        other = RecordingClassifier(spec).fit(x, changed, seasons, dates=dates)
        # Later calibration labels cannot change the model that supplied their
        # out-of-time margins; they may change calibration and final refitting.
        np.testing.assert_array_equal(model.fit_calls[0]["coefficients"],
                                      other.fit_calls[0]["coefficients"])
        self.assertGreater(model.calibration_slope_, 0)
        self.assertEqual(model.calibration_metadata_["calibration_rows"], 30)
        np.testing.assert_array_equal(model.predict_proba(x) > .5, model.decision_function(x) > 0)

    def test_positive_calibration_cannot_reverse_classifier_and_respects_zero(self):
        margins = np.array([-2., -.5, 0., .5, 2.])
        for y in (np.array([0, 0, 0, 1, 1]), np.array([1, 1, 1, 0, 0])):
            slope = positive_calibration_slope(margins, y)
            self.assertTrue(.05 <= slope <= 20)
            np.testing.assert_array_equal(slope*margins > 0, margins > 0)
        with self.assertRaises(ValueError):
            positive_calibration_slope([np.nan], [0])

    def test_probabilities_reload_and_exchange_exactly_for_both_objectives(self):
        x, y, seasons, dates = self.fixture()
        for head in ("logistic", "squared_hinge"):
            model = make_model(self.spec(x.columns, design="rff", head=head, width=256))
            model.fit(x, y, seasons, dates=dates)
            probability = model.predict_proba(x)
            self.assertTrue(np.all((probability > 0) & (probability < 1)))
            np.testing.assert_allclose(probability+model.predict_proba(mirror_features(x)),
                                       1., atol=1e-13)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"model.joblib"
                joblib.dump(model, path)
                loaded = joblib.load(path)
                reordered = x[list(reversed(x.columns))].assign(a_won="POISON", score="POISON")
                np.testing.assert_array_equal(probability, loaded.predict_proba(reordered))

    def test_invalid_or_final_holdout_training_is_rejected(self):
        x, y, seasons, dates = self.fixture()
        spec = self.spec(x.columns)
        for changed in (np.repeat(2023, len(x)), np.repeat(np.nan, len(x)), seasons+.5):
            with self.assertRaises(ValueError):
                make_model(spec).fit(x, y, changed, dates=dates)
        for changed_dates in (np.repeat("2024-01-01", len(x)), dates[:-1]):
            with self.assertRaises(ValueError):
                make_model(spec).fit(x, y, seasons, dates=changed_dates)
        model = make_model(spec).fit(x, y, seasons)
        for bad in (x.drop(columns="log_rank_diff"), x.assign(log_rank_diff=np.nan),
                    pd.concat([x, x[["log_rank_diff"]]], axis=1)):
            with self.assertRaises(ValueError):
                model.predict_proba(bad)


if __name__ == "__main__":
    unittest.main()

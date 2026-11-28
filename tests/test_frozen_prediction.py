"""Inference artifacts retain predictions without reading outcomes or fitting."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.predict_frozen import ROOT, predict_bundle
from scripts.train_models import SymmetricClassifier, mirror_features


class FrozenPredictionTest(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({"strength_diff": [-2., -1., -.5, .5, 1., 2.],
                               "elo_k32_diff": [-150., -70., -20., 30., 80., 160.]})
        spec = {"columns": ["strength_diff"], "kind": "logistic", "params": {"C": .1}}
        model = SymmetricClassifier(spec).fit(self.x, np.array([0, 0, 1, 0, 1, 1]))
        self.bundle = {
            "models": {"linear": model}, "feature_names": list(self.x),
            "raw_elo_columns": ["elo_k32_diff"],
            "frozen": {"selected_by_family": {"advanced_linear": "linear", "raw_elo": "raw_elo_k32_diff"},
                       "ensemble_components": ["linear", "raw_elo_k32_diff"],
                       "ensemble_weights": [.6, .4], "advanced_champion": "oof_convex_ensemble"},
        }

    def test_predictions_survive_label_poison_and_column_reordering(self):
        expected = predict_bundle(self.bundle, self.x)
        poisoned = self.x.iloc[:, ::-1].assign(a_won="DO_NOT_READ", match_id="opaque")
        pd.testing.assert_frame_equal(expected, predict_bundle(self.bundle, poisoned))
        np.testing.assert_array_equal(expected.linear, self.bundle["models"]["linear"].predict_proba(self.x))

    def test_input_order_and_swap_symmetry(self):
        x = self.x.iloc[[4, 0, 3]]
        predictions = predict_bundle(self.bundle, x)
        self.assertEqual(list(predictions.index), [4, 0, 3])
        swapped = predict_bundle(self.bundle, mirror_features(x))
        np.testing.assert_allclose(predictions.to_numpy() + swapped.to_numpy(), 1., atol=1e-14)

    def test_invalid_required_features_fail_before_model_prediction(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            predict_bundle(self.bundle, self.x.drop(columns="strength_diff"))
        for value in (np.nan, np.inf, "SECRET"):
            x = self.x.astype(object)
            x.loc[0, "strength_diff"] = value
            with self.assertRaisesRegex(ValueError, "numeric and finite"):
                predict_bundle(self.bundle, x)

    def test_invalid_frozen_weights_are_rejected(self):
        for weights in ([1., -1.], [0., 0.], [np.nan, 1.], [1.]):
            bundle = copy.deepcopy(self.bundle)
            bundle["frozen"]["ensemble_weights"] = weights
            with self.assertRaisesRegex(ValueError, "convex"):
                predict_bundle(bundle, self.x)

    def test_cli_round_trip_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            joblib.dump(self.bundle, folder / "bundle.joblib")
            features = self.x.assign(match_id=[f"m{i}" for i in range(len(self.x))], a_won="unused")
            features.to_csv(folder / "input.csv", index=False)
            result = subprocess.run([sys.executable, "-m", "scripts.predict_frozen",
                "--model", str(folder / "bundle.joblib"), "--features", str(folder / "input.csv"),
                "--output", str(folder / "output.csv")], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("oof_convex_ensemble", result.stdout)
            actual = pd.read_csv(folder / "output.csv")
            self.assertEqual(actual.match_id.tolist(), features.match_id.tolist())
            np.testing.assert_allclose(actual.drop(columns="match_id"),
                                       predict_bundle(self.bundle, self.x), atol=1e-14)


if __name__ == "__main__":
    unittest.main()

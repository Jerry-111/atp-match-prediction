"""Predict from an already fitted, trusted local model bundle without refitting.

Feature creation remains a separate chronological step. This module accepts no
outcomes and does not update ratings, choose hyperparameters or calibrate scores.
Only load joblib artifacts you trust: pickle-based formats can execute code.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]


def predict_bundle(bundle, features):
    """Return selected-model and frozen-ensemble probabilities in input order.

    Extra columns (including outcomes and identifiers) are ignored. Required
    feature columns are selected in the order stored in the bundle. The caller
    is responsible for creating these features from information available at
    the prediction time using the frozen T-21 rule.
    """
    required = list(bundle["feature_names"])
    if not required or len(set(required)) != len(required):
        raise ValueError("Model bundle must declare unique feature columns")
    if features.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names")
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if features.empty:
        raise ValueError("At least one feature row is required")
    try:
        x = features.loc[:, required].astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError("Required features must be numeric and finite") from error
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Required features must be numeric and finite")

    frozen = bundle["frozen"]
    selected = list(dict.fromkeys(frozen["selected_by_family"].values()))
    probabilities = {}
    for name in selected:
        if name.startswith("raw_"):
            column = name[4:]
            if column not in bundle["raw_elo_columns"]:
                raise ValueError(f"Unrecognized frozen raw Elo column: {column}")
            p = expit(np.log(10) / 400 * x[column].to_numpy())
        else:
            if name not in bundle["models"]:
                raise ValueError(f"Missing fitted model: {name}")
            p = np.asarray(bundle["models"][name].predict_proba(x), dtype=float)
        if p.shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid probabilities from model: {name}")
        probabilities[name] = p

    components = frozen["ensemble_components"]
    weights = np.asarray(frozen["ensemble_weights"], dtype=float)
    if (not components or weights.shape != (len(components),) or
            not np.isfinite(weights).all() or np.any(weights < 0) or
            not np.isclose(weights.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError("Frozen ensemble weights must be a convex combination")
    if any(name not in probabilities for name in components):
        raise ValueError("Frozen ensemble references an unavailable component")
    probabilities["oof_convex_ensemble"] = np.column_stack(
        [probabilities[name] for name in components]) @ weights
    if frozen["advanced_champion"] not in probabilities:
        raise ValueError("Frozen recommendation is not available in the bundle")
    return pd.DataFrame(probabilities, index=features.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "results/modeling/frozen_models.joblib",
                        help="Trusted local fitted bundle; no training is performed")
    parser.add_argument("--features", required=True, type=Path,
                        help="CSV with the complete frozen numeric feature schema")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    features = pd.read_csv(args.features)
    predictions = predict_bundle(bundle, features)
    if "match_id" in features:
        predictions.insert(0, "match_id", features.match_id.to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predicted {len(predictions):,} rows; frozen recommendation: "
          f"{bundle['frozen']['advanced_champion']}; wrote {args.output}")


if __name__ == "__main__":
    main()

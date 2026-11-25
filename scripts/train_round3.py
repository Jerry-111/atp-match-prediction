"""Bounded history-breadth experiment with unchanged scoring cohorts.

Only training start and a fixed linear/spline regularization grid vary. The
2020--21 global winner is tested against the original early-selected ensemble
on 2022. A minimum 0.001 log-loss gain is required for adopting the new procedure.
2023 has already been seen in prior rounds and remains exploratory here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.predict_frozen import predict_bundle
from scripts.round2_learning import candidate_specs as spline_specs
from scripts.round2_learning import make_model as make_spline
from scripts.train_models import (FOLDS, ROOT, SEED, SymmetricClassifier,
                                 cluster_interval, mean_loss, sha256, temporal_masks)
from scripts.train_round2 import load_round1, predict_round2


GATE_IMPROVEMENT = .001
START_SEASONS = (2009, 2016)


def _history_specs(columns):
    """Sixteen matched history/learner settings for one format condition."""
    columns = list(columns)
    odd = [c for c in columns if c.endswith("_diff")]
    if not odd:
        raise ValueError("At least one difference feature is required")
    spline = [s for s in spline_specs(columns) if s["interactions"]]
    # In a small synthetic schema there may be no available interactions; select
    # by the structural candidate name, not by whether context happens to exist.
    if len(spline) != 4:
        spline = [s for s in spline_specs(columns) if "_context1_" in s["name"]]
    result = []
    for start in START_SEASONS:
        for c in (.003, .01):
            for half in (None, 5.):
                result.append({"name": f"history{start}_linear_C{c:g}_half{half}",
                               "family": f"history{start}_linear", "kind": "logistic",
                               "columns": odd, "params": {"C": c},
                               "half_life": half, "train_start": start})
        for spec in spline:
            result.append({**spec, "name": f"history{start}_{spec['name']}",
                           "family": f"history{start}_spline", "train_start": start})
    return result


def candidate_specs(columns):
    """32 declared settings: matched history breadth, learner and format controls."""
    columns = list(columns)
    result = []
    for include_format in (False, True):
        selected = columns if include_format else [c for c in columns if not c.startswith("format_")]
        for spec in _history_specs(selected):
            flag = int(include_format)
            result.append({**spec, "name": f"{spec['name']}_format{flag}",
                           "family": f"{spec['family']}_format{flag}",
                           "include_format": include_format})
    return result


def make_model(spec):
    return SymmetricClassifier(spec) if spec["kind"] == "logistic" else make_spline(spec)


def training_masks(rows, year, start):
    train, validation = temporal_masks(rows, year)
    train &= rows.season.ge(start)
    if not train.any():
        raise ValueError(f"No training rows from {start} before fold {year}")
    return train, validation


def select_configurations(specs, predictions, outcomes, mask):
    """Return the global best and one finalist per matched history/learner cell."""
    def criterion(name):
        return mean_loss(outcomes[mask], predictions[name][mask]), name
    best = min((s["name"] for s in specs), key=criterion)
    families = sorted({s["family"] for s in specs})
    finalists = {family: min((s["name"] for s in specs if s["family"] == family), key=criterion)
                 for family in families}
    return best, finalists


def choose_procedure(outcomes, predictions, specs, seasons, reference):
    """Choose a configuration on 2020--21; use only 2022 for the upgrade gate."""
    seasons = np.asarray(seasons)
    early, gate = np.isin(seasons, (2020, 2021)), seasons == 2022
    if not early.any() or not gate.any():
        raise ValueError("Procedure selection requires early folds and the 2022 gate")
    selected, _ = select_configurations(specs, predictions, outcomes, early)
    candidate_loss = mean_loss(outcomes[gate], predictions[selected][gate])
    reference_loss = mean_loss(outcomes[gate], reference[gate])
    improvement = reference_loss - candidate_loss
    return {"procedure": "adopt_history_model" if improvement >= GATE_IMPROVEMENT else "retain_round1",
            "early_selected": selected, "candidate_2022_loss": candidate_loss,
            "reference_2022_loss": reference_loss, "absolute_improvement": improvement,
            "required_improvement": GATE_IMPROVEMENT, "selection_seasons": [2020, 2021],
            "gate_season": 2022}


def predict_round3(bundle, features):
    """Reloaded inference only; labels and any other extra columns are ignored."""
    required = bundle["feature_names"]
    if features.empty or features.columns.duplicated().any():
        raise ValueError("Prediction input must have rows and unique columns")
    missing = [c for c in required if c not in features]
    if missing:
        raise ValueError(f"Missing frozen features: {missing}")
    try:
        x = features.loc[:, required].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen features must be numeric and finite") from error
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Frozen features must be numeric and finite")
    predictions = {name: model.predict_proba(x) for name, model in bundle["models"].items()}
    first = predict_bundle(bundle["round1_bundle"], x)
    for alias, name in bundle["round1_aliases"].items():
        predictions[alias] = first[name].to_numpy()
    if bundle.get("round2_bundle") is not None:
        second = predict_round2(bundle["round2_bundle"], x)
        for alias, name in bundle["round2_aliases"].items():
            predictions[alias] = second[name].to_numpy()
    for name, values in predictions.items():
        values = np.asarray(values)
        if values.shape != (len(x),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid frozen probabilities: {name}")
    return pd.DataFrame(predictions, index=features.index)


def verify_unchanged_cohort(original, expanded, features, round2_dir):
    """Preserve identities, labels, all original fields, and existing features."""
    original_ids = original.match_id.tolist()
    if expanded.match_id.duplicated().any():
        raise ValueError("Expanded target IDs must remain unique")
    retained = expanded.set_index("match_id").loc[original_ids].reset_index()
    retained = retained.loc[:, original.columns]
    pd.testing.assert_frame_equal(retained, original.reset_index(drop=True), check_dtype=False,
                                  check_exact=True)
    previous = pd.read_csv(round2_dir / "development_features.csv")
    locations = pd.Series(np.arange(len(expanded)), index=expanded.match_id).loc[original_ids].to_numpy()
    rebuilt = features.iloc[locations].reset_index(drop=True)
    extra = [c for c in rebuilt if c not in previous]
    if (list(rebuilt)[:len(previous.columns)] != list(previous) or len(rebuilt) != len(previous)
            or any(not c.startswith("format_") for c in extra)):
        raise ValueError("Round-three schema must preserve round two before appended format features")
    rebuilt = rebuilt.loc[:, previous.columns]
    if not np.allclose(rebuilt.to_numpy(), previous.to_numpy(), atol=1e-10, rtol=0):
        raise ValueError("Existing cohort feature values changed during training expansion")
    return float(np.max(np.abs(rebuilt.to_numpy() - previous.to_numpy())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/round3")
    parser.add_argument("--round1", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--round2", type=Path, default=ROOT / "results/round2")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / "data/processed/model_features.csv"
    inventory_path = ROOT / "data/processed/mens_singles_2006_2025.csv"
    original = pd.read_csv(original_path)
    _, old_oof, first_frozen = load_round1(original, args.round1)
    second_frozen = json.loads((args.round2 / "frozen_selection.json").read_text())
    if second_frozen["recommendation"] != "round1_ensemble":
        raise ValueError("This registered round-three gate assumes round two retained round one")
    inventory = pd.read_csv(inventory_path, dtype={"a_won": "string"})
    inventory = inventory.loc[inventory.season.le(2023)].copy()
    from scripts.expanded_training import build_expanded_targets, build_expanded_features
    print("Building expanded historical fitting cohort", flush=True)
    rows = build_expanded_targets(original, inventory)
    x = build_expanded_features(rows, inventory)
    if not rows.season.between(2009, 2023).all() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Only finite 2009--2023 development inputs are allowed")
    feature_error = verify_unchanged_cohort(original, rows, x, args.round2)
    oof_mask = rows.season.isin(FOLDS)
    oof_rows = rows.loc[oof_mask]
    expected = original.loc[original.season.isin(FOLDS), ["match_id", "season", "a_won"]].reset_index(drop=True)
    if not oof_rows.loc[:, expected.columns].reset_index(drop=True).equals(expected):
        raise ValueError("Historical expansion changed the scored forward-fold cohort")
    specs = candidate_specs(x.columns)
    registry = {"round": 3, "seed": SEED, "folds": list(FOLDS), "train_starts": list(START_SEASONS),
                "format_conditions": [False, True],
                "candidates": specs, "feature_names": list(x), "feature_metadata": x.attrs,
                "target_sha256": sha256(original_path), "inventory_sha256": sha256(inventory_path),
                "round2_feature_sha256": sha256(args.round2 / "development_features.csv"),
                "previous_feature_max_absolute_error": feature_error,
                "training_row_counts": {str(s): int(rows.season.between(s, 2022).sum()) for s in START_SEASONS},
                "exploratory_after_2023": True, "final_test_evaluated": False,
                "selection": "Global best on2020--21; at least0.001 improvement vs round-one early ensemble on2022; refit pooled2020--22 choice."}
    (out / "search_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    rows.to_csv(out / "development_targets.csv", index=False)
    x.to_csv(out / "development_features.csv", index=False)
    y = oof_rows.a_won.to_numpy()
    oof, scores = {}, []
    for number, spec in enumerate(specs, 1):
        p = pd.Series(index=oof_rows.index, dtype=float)
        fold_scores = {}
        for year in FOLDS:
            train, valid = training_masks(rows, year, spec["train_start"])
            model = make_model(spec).fit(x.loc[train], rows.a_won[train], rows.season[train])
            probability = model.predict_proba(x.loc[valid])
            p.loc[rows.index[valid]] = probability
            fold_scores[str(year)] = mean_loss(rows.a_won[valid].to_numpy(), probability)
            fold_scores[f"train_n_{year}"] = int(train.sum())
        values = p.to_numpy()
        oof[spec["name"]] = values
        scores.append({"name": spec["name"], "family": spec["family"],
                       "train_start": spec["train_start"], "mean_log_loss": mean_loss(y, values), **fold_scores})
        pd.DataFrame(scores).to_csv(out / "temporal_cv.csv", index=False)
        print(f"[{number}/{len(specs)}] {spec['name']}: {scores[-1]['mean_log_loss']:.6f}", flush=True)
    reference_gate = np.column_stack([old_oof["r1__" + n] for n in first_frozen["gate_components"]]) @ first_frozen["gate_weights"]
    gate = choose_procedure(y, oof, specs, oof_rows.season.to_numpy(), reference_gate)
    best, finalists = select_configurations(specs, oof, y, np.ones(len(y), dtype=bool))
    choice = best if gate["procedure"] == "adopt_history_model" else "round1_ensemble"
    frozen = {"round": 3, "recommendation": choice, "best_single": best, "selected_by_family": finalists,
              "procedure_gate": gate, "history_lag_days": 21, "fit_end_season": 2022,
              "exploratory_after_2023": True, "final_test_evaluated": False}
    (out / "frozen_selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"Frozen round-three recommendation: {choice}; now scoring exploratory2023", flush=True)
    by_name = {s["name"]: s for s in specs}
    models = {}
    for name in sorted(set(finalists.values()) | {best}):
        spec = by_name[name]
        train, valid = training_masks(rows, 2023, spec["train_start"])
        models[name] = make_model(spec).fit(x.loc[train], rows.a_won[train], rows.season[train])
    first_bundle = joblib.load(args.round1 / "frozen_models.joblib")
    second_bundle = joblib.load(args.round2 / "frozen_models.joblib")
    bundle = {"models": models, "feature_names": list(x), "frozen": frozen,
              "round1_bundle": first_bundle, "round2_bundle": second_bundle,
              "round1_aliases": {"round1_ensemble": "oof_convex_ensemble", "round1_best_single": first_frozen["best_single"],
                                 "rank_reference": first_frozen["selected_by_family"]["lr_000"]},
              "round2_aliases": {"round2_selected_single": second_frozen["best_single"]}}
    valid = rows.season.eq(2023)
    predictions = predict_round3(bundle, x.loc[valid])
    vr = rows.loc[valid]
    table = []
    for name in predictions:
        probability = predictions[name].to_numpy()
        score = score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, probability))
        table.append({"model": name, "recommended": name == choice, **score,
                      "brier": float(np.mean((vr.a_won.to_numpy() - probability) ** 2))})
    result = pd.DataFrame(table).sort_values("mean_log_loss")
    result.to_csv(out / "validation_comparison.csv", index=False)
    pd.concat([vr.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1).to_csv(out / "validation_predictions.csv", index=False)
    pd.concat([oof_rows[["match_id", "season", "a_won"]].reset_index(drop=True), pd.DataFrame(oof)], axis=1).to_csv(out / "oof_predictions.csv", index=False)
    intervals = {name: cluster_interval(vr, predictions.round1_ensemble.to_numpy(), predictions[name].to_numpy()) for name in predictions}
    summary = {"round": 3, "recommendation": choice, "new_candidates": len(specs),
               "validation_matches": len(vr), "comparisons_vs_round1": intervals,
               "training_row_counts": registry["training_row_counts"], "exploratory_after_2023": True,
               "final_test_evaluated": False, "previous_feature_max_absolute_error": feature_error}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    joblib.dump(bundle, out / "frozen_models.joblib")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    from scripts.train_round3 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()

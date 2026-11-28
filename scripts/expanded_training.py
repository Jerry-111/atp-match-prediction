"""Round 3 historical-target expansion, preserving the existing evaluation rows.

The fixed additional cohort is eligible Tour main draws in 2009--2015. History
still starts in 2006 and every feature uses results through T-21 inclusive.
No season is relabeled and the 2024--2025 final holdout remains inaccessible.
"""

from datetime import date, timedelta
from math import isclose, log

import numpy as np
import pandas as pd

from scripts.advanced_features import HistoricalState, _date, build_advanced_features
from scripts.adjusted_point_features import build_adjusted_point_features
from scripts.baseline import DEFAULT_CACHE
from scripts.dynamic_features import build_dynamic_features
from scripts.serve_features import _annual_rows, build_serve_features, development_entries


FIRST_TARGET_SEASON = 2009
BASE = ["log_rank_diff", "overall_rate_diff", "surface_rate_diff", "h2h_rate_diff"]


def _target_row(match, state):
    """Reconstruct the published target schema from one eligible inventory row."""
    day = _date(match["tourney_date"])
    cutoff = day - timedelta(days=21)
    a, b = str(match["player_a_id"]), str(match["player_b_id"])
    rank_a, rank_b = int(match["rank_a"]), int(match["rank_b"])
    name_a, name_b = match.get("player_a_name", a), match.get("player_b_name", b)
    won = float(match["a_won"])
    if a > b:
        a, b, rank_a, rank_b, name_a, name_b, won = b, a, rank_b, rank_a, name_b, name_a, 1-won
    if a == b or min(rank_a, rank_b) <= 0 or won not in (0, 1):
        raise ValueError(f"Invalid inventory target {match['match_id']}")
    surface = str(match["surface"])
    row = {
        "match_id": str(match["match_id"]).removeprefix("tour:"),
        "season": int(match["season"]), "split": "train" if int(match["season"]) < 2023 else "validation",
        "tourney_date": day.isoformat(), "tourney_id": match["tourney_id"],
        "match_num": int(match["match_num"]), "round": match["round"], "surface": surface,
        "player_a_id": a, "player_b_id": b, "player_a_name": name_a, "player_b_name": name_b,
        "rank_a": rank_a, "rank_b": rank_b, "a_won": int(won),
        "log_rank_diff": log(rank_b) - log(rank_a),
    }
    surface_rates, overall_rates = [], []
    for side, player in (("a", a), ("b", b)):
        history = state.surface[player, surface]
        count, wins, _ = history.summary(day - timedelta(days=364), cutoff)
        rate = (wins + 1) / (count + 2)
        latest = history.last_date(cutoff) if count else None
        row.update({f"surface_matches_{side}": count, f"surface_wins_{side}": int(wins),
                    f"surface_rate_{side}": rate,
                    f"latest_history_date_{side}": latest.isoformat() if latest else np.nan})
        surface_rates.append(rate)
        count, wins, _ = state.overall[player].summary(day - timedelta(days=90), cutoff)
        row.update({f"overall_matches_{side}": count, f"overall_wins_{side}": int(wins)})
        overall_rates.append((wins + 1) / (count + 2))
    count, wins, _ = state.pair[a, b].summary(date.min, cutoff)
    row.update({"h2h_matches": count, "h2h_wins_a": int(wins),
                "h2h_rate_diff": 2 * (wins + 1) / (count + 2) - 1,
                "surface_rate_diff": surface_rates[0] - surface_rates[1],
                "overall_rate_diff": overall_rates[0] - overall_rates[1]})
    return row


def build_expanded_targets(original, inventory):
    """Prepend canonical 2009--2015 targets; preserve original rows and dtypes.

    Original values/order are checked against the inventory and four predictors
    are independently recomputed before accepting expansion. Output has a fresh
    RangeIndex, and its final ``len(original)`` rows equal the original values.
    """
    if original.empty or not pd.to_numeric(original.season).between(2016, 2023).all():
        raise ValueError("Original development rows must be nonempty 2016..2023; final holdout is locked")
    if original.match_id.duplicated().any():
        raise ValueError("Original match IDs must be unique")
    state = HistoricalState(inventory)
    eligible = inventory.loc[inventory.source.eq("tour") &
                             pd.to_numeric(inventory.season).between(2009, 2023)]
    inventory_by_id = {str(r["match_id"]).removeprefix("tour:"): r for r in eligible.to_dict("records")}
    for target in original.to_dict("records"):
        identifier = str(target["match_id"])
        if identifier not in inventory_by_id:
            raise ValueError(f"Original target missing from inventory: {identifier}")
        reconstructed = _target_row(inventory_by_id[identifier], state)
        for key in ("season", "split", "player_a_id", "player_b_id", "rank_a", "rank_b", "surface", "a_won"):
            if reconstructed[key] != target[key]:
                raise ValueError(f"Original target identity differs at {identifier}: {key}")
        if _date(target["tourney_date"]) != _date(reconstructed["tourney_date"]):
            raise ValueError(f"Original target date differs: {identifier}")
        for key in BASE:
            if not isclose(float(target[key]), reconstructed[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Recomputed original feature mismatch at {identifier}: {key}")
    older = eligible.loc[pd.to_numeric(eligible.season).between(2009, 2015)]
    older = older.sort_values(["season", "tourney_date", "match_id"])
    additions = pd.DataFrame([_target_row(r, state) for r in older.to_dict("records")])
    if additions.empty:
        return original.reset_index(drop=True).copy()
    missing_columns = set(original.columns) - set(additions.columns)
    if missing_columns:
        raise ValueError(f"Cannot reconstruct original target columns: {sorted(missing_columns)}")
    additions = additions.loc[:, original.columns].astype(original.dtypes.to_dict())
    expanded = pd.concat([additions, original], ignore_index=True)
    if expanded.match_id.duplicated().any():
        raise ValueError("Expanded target IDs must remain unique")
    pd.testing.assert_frame_equal(expanded.iloc[len(additions):].reset_index(drop=True),
                                  original.reset_index(drop=True), check_exact=True)
    expanded.attrs.update({"first_target_season": FIRST_TARGET_SEASON,
                           "added_training_rows": len(additions), "original_rows": len(original),
                           "history_first_season": 2006, "history_lag_days": 21,
                           "verified_original_features": BASE})
    return expanded


def _match_format(targets, cache_dir, raw_rows):
    """Read actual pre-match best_of from the pinned raw metadata, not results."""
    required = {str(r["match_id"]): int(r["season"]) for r in targets.to_dict("records")}
    rows = _annual_rows(cache_dir, development_entries()) if raw_rows is None else raw_rows
    formats = {}
    for season, raw in rows:
        if not 2009 <= int(season) <= 2023:
            continue
        key = f"{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in required:
            continue
        if key in formats or int(season) != required[key]:
            raise ValueError(f"Duplicate or mismatched match-format source: {key}")
        try:
            value = int(raw["best_of"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Missing or invalid best_of for {key}") from exc
        if value not in (3, 5):
            raise ValueError(f"Unsupported best_of for {key}: {value}")
        formats[key] = float(value == 5)
    if required.keys() - formats.keys():
        raise ValueError("Match-format source omitted target matches")
    return pd.Series([formats[str(identifier)] for identifier in targets.match_id], index=targets.index)


def build_expanded_features(targets, inventory, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Build 198 unchanged Round 2 features followed by four match-format columns."""
    if not pd.to_numeric(targets.season).between(FIRST_TARGET_SEASON, 2023).all():
        raise ValueError("Expanded targets must be 2009..2023; final holdout is locked")
    if raw_rows is not None:
        raw_rows = list(raw_rows)  # Two feature builders consume the same fixture.
    options = {"first_target_season": FIRST_TARGET_SEASON}
    advanced = build_advanced_features(targets, inventory, **options)
    serve = build_serve_features(targets, inventory, cache_dir=cache_dir, raw_rows=raw_rows, **options)
    dynamic = build_dynamic_features(targets, inventory, **options)
    adjusted = build_adjusted_point_features(targets, inventory, cache_dir=cache_dir, raw_rows=raw_rows, **options)
    frame = pd.concat([targets[BASE], advanced, serve, dynamic, adjusted], axis=1)
    if frame.columns.duplicated().any() or frame.shape[1] != 198:
        raise ValueError("Expanded feature schema differs from the 198-column Round 2 matrix")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite expanded training features")
    flag = _match_format(targets, cache_dir, raw_rows)
    frame["format_best_of_five"] = flag
    frame["format_log_rank_diff"] = flag * frame["log_rank_diff"]
    frame["format_elo_diff"] = flag * frame["elo_k32_diff"]
    frame["format_points_diff"] = flag * (frame["points_overall_364_serve_rate_diff"] +
                                          frame["points_overall_364_return_rate_diff"])
    frame.attrs.update({"first_target_season": FIRST_TARGET_SEASON,
                        "groups": {"round1": list(BASE) + list(advanced) + list(serve),
                                   "dynamic": list(dynamic), "adjusted_points": list(adjusted),
                                   "format": list(frame.columns[-4:])},
                        "best_of_source": "actual pinned annual CSV best_of, joined by match ID and season",
                        "advanced_metadata": advanced.attrs, "serve_metadata": serve.attrs,
                        "dynamic_metadata": dynamic.attrs, "adjusted_metadata": adjusted.attrs})
    return frame

"""Bounded accuracy-oriented learners with exact player-exchange symmetry.

This module never chooses a model or reads validation outcomes. The caller owns
temporal outer folds and accuracy-first selection. Squared-hinge margins get a
positive, zero-intercept calibration slope from a genuine earlier-fit/later-score
split inside each outer training set; its final learner then uses all outer rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from scripts.round2_learning import candidate_specs as spline_specs
from scripts.round2_learning import make_model as make_spline

SEED = 158
INTERACTIONS = (
    ("log_rank_diff", "profile_age_mean"),
    ("elo_k32_diff", "profile_age_mean"),
    ("adjpoints_h365_overall_serve_prob_diff", "profile_age_mean"),
    ("profile_age_diff", "log_rank_mean"),
    ("profile_age30_diff", "log_rank_mean"),
    ("profile_age_interval_diff", "log_rank_mean"),
    ("overall_inactivity_log_diff", "profile_age_mean"),
    ("profile_age_diff", "overall_inactivity_log_sum"),
)
KERNEL_COLUMNS = (
    "log_rank_diff", "elo_k32_diff", "shrunk_surface_elo_k32_diff",
    "form_90_rate_diff", "h2h_rate_diff",
    "points_overall_364_serve_rate_diff", "points_overall_364_return_rate_diff",
    "points_surface_364_serve_rate_diff", "points_surface_364_return_rate_diff",
    "adjpoints_h365_overall_serve_prob_diff", "adjpoints_h365_surface_serve_prob_diff",
    "profile_age_diff", "profile_age30_diff", "profile_age_interval_diff",
    "log_rank_mean", "profile_age_mean", "surface_is_clay", "surface_is_grass",
    "format_best_of_five", "overall_inactivity_log_sum",
)


def candidate_specs(columns):
    """26 fixed candidates; all settings are independent of outcomes."""
    columns = list(columns)
    if len(columns) != len(set(columns)):
        raise ValueError("Candidate feature names must be unique")
    odd = [c for c in columns if c.endswith("_diff")]
    if not odd:
        raise ValueError("At least one *_diff feature is required")
    interactions = [[a, b] for a, b in INTERACTIONS if a in columns and b in columns]
    kernel = [c for c in KERNEL_COLUMNS if c in columns]
    if not kernel:
        kernel = odd[:1]
    designs = [("linear", 0, 0.), ("context", 0, 0.)]
    designs += [("rff", width, gamma) for width in (256, 512) for gamma in (.5, 2.)]
    result = []
    for design, width, gamma in designs:
        used_interactions = [] if design == "linear" else interactions
        required = list(dict.fromkeys(odd + [b for _, b in used_interactions] +
                                     (kernel if design == "rff" else [])))
        label = design if design != "rff" else f"rff{width}_g{gamma:g}"
        for head in ("logistic", "squared_hinge"):
            for c in (.003, .03):
                result.append({
                    "name": f"r5_{label}_{head}_C{c:g}", "kind": f"r5_{head}",
                    "family": f"r5_{design}_{head}", "design": design,
                    "columns": required, "odd_columns": odd,
                    "kernel_columns": kernel if design == "rff" else [],
                    "interactions": used_interactions,
                    "params": {"C": c, "half_life": 3., "components": width,
                               "gamma_multiplier": gamma, "seed": SEED,
                               "interaction_weight": .3, "kernel_weight": .5},
                })
    for c in (.003, .01):
        spec = next(s for s in spline_specs(columns)
                    if s["params"]["C"] == c and s["params"]["half_life"] == 3.)
        result.append({**spec, "name": f"r5_existing_{spec['name']}", "family": "r5_existing_spline"})
    return result


def _rms(values):
    scale = np.sqrt(np.mean(values * values, axis=0))
    return np.where(scale > 1e-10, scale, 1.)


class OddFeatureMap:
    """Train-only scaled linear terms, limited contexts and odd RFF residuals."""

    def __init__(self, spec):
        self.spec = spec

    def _input(self, x):
        required = self.spec["columns"]
        if x.columns.duplicated().any() or any(c not in x for c in required):
            raise ValueError("Missing or duplicate learner feature columns")
        try:
            selected = x.loc[:, required].astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError("Learner features must be finite numeric values") from error
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError("Learner features must be finite numeric values")
        return selected

    def fit(self, x):
        x = self._input(x)
        odd = self.spec["odd_columns"]
        if not len(x) or not odd or any(not c.endswith("_diff") for c in odd):
            raise ValueError("Feature map needs training rows and odd difference columns")
        self.odd_indices_ = {name: i for i, name in enumerate(odd)}
        self.odd_scale_ = _rms(x[odd].to_numpy())
        self.context_mean_, self.context_scale_ = {}, {}
        for a, b in self.spec["interactions"]:
            if a not in odd or b.endswith("_diff"):
                raise ValueError("Context interactions must be odd times even")
            self.context_mean_[b] = float(x[b].mean())
            self.context_scale_[b] = max(float(x[b].std(ddof=0)), 1e-10)
        interaction = self._interactions(x)
        self.interaction_scale_ = _rms(interaction) if interaction.shape[1] else np.array([])
        kernel = self.spec["kernel_columns"]
        self.kernel_mean_ = np.array([0. if c.endswith("_diff") else x[c].mean() for c in kernel])
        self.kernel_scale_ = _rms(x[kernel].to_numpy()-self.kernel_mean_) if kernel else np.array([])
        self.kernel_signs_ = np.array([-1. if c.endswith("_diff") else 1. for c in kernel])
        width = self.spec["params"]["components"]
        if width:
            rng = np.random.default_rng(self.spec["params"]["seed"])
            # Gaussian k(u,v)=exp(-gamma*||u-v||^2) has N(0,2gamma I)
            # spectral frequencies; phases are uniform on [0,2pi].
            gamma = self.spec["params"]["gamma_multiplier"] / len(kernel)
            self.frequencies_ = rng.normal(0., np.sqrt(2*gamma), size=(len(kernel), width))
            self.phases_ = rng.uniform(0., 2*np.pi, width)
        self.fitted_rows_ = len(x)
        return self

    def _odd(self, x):
        return np.clip(x[self.spec["odd_columns"]].to_numpy()/self.odd_scale_, -8., 8.)

    def _interactions(self, x):
        odd = self._odd(x)
        values = []
        for a, b in self.spec["interactions"]:
            context = np.clip((x[b].to_numpy()-self.context_mean_[b])/self.context_scale_[b], -3., 3.)
            values.append(odd[:, self.odd_indices_[a]]*context)
        return np.column_stack(values) if values else np.empty((len(x), 0))

    def transform(self, x):
        x = self._input(x)
        parts = [self._odd(x)]
        if self.spec["interactions"]:
            parts.append(self._interactions(x)/self.interaction_scale_ *
                         self.spec["params"]["interaction_weight"])
        width = self.spec["params"]["components"]
        if width:
            kernel = self.spec["kernel_columns"]
            scaled = np.clip((x[kernel].to_numpy()-self.kernel_mean_)/self.kernel_scale_, -5., 5.)
            original = scaled @ self.frequencies_ + self.phases_
            exchanged = (scaled*self.kernel_signs_) @ self.frequencies_ + self.phases_
            # Odd projection is the project's symmetry adaptation. sqrt(d)
            # balances the block without changing its penalty as D increases.
            residual = .5*(np.cos(original)-np.cos(exchanged))*np.sqrt(2*len(kernel)/width)
            parts.append(self.spec["params"]["kernel_weight"]*residual)
        return np.column_stack(parts)


def calibration_masks(seasons, dates=None):
    """Last training year is scored by an earlier-only inner fit.

    Exact dates permit the normal inclusive 21-day purge. Without dates, omit
    the whole immediately preceding year to avoid guessing its late dates.
    """
    seasons = np.asarray(seasons, dtype=int)
    year = int(seasons.max())
    valid = seasons == year
    if dates is None:
        train = seasons <= year-2
        metadata = {"policy": "whole-year embargo without dates", "calibration_year": year}
    else:
        parsed = pd.to_datetime(np.asarray(dates), errors="raise")
        if len(parsed) != len(seasons) or pd.isna(parsed).any():
            raise ValueError("Calibration dates must be finite and aligned")
        cutoff = parsed[valid].min()-pd.Timedelta(days=21)
        train = (seasons < year) & (parsed <= cutoff)
        metadata = {"policy": "last-year calibration with 21-day purge",
                    "calibration_year": year, "inner_training_cutoff": cutoff.isoformat(),
                    "first_calibration_date": parsed[valid].min().isoformat()}
    if not train.any() or not valid.any():
        raise ValueError("Need earlier fitting rows and a later calibration year")
    metadata.update(inner_training_rows=int(train.sum()), calibration_rows=int(valid.sum()),
                    inner_training_seasons=sorted(set(seasons[train].tolist())))
    return np.asarray(train), np.asarray(valid), metadata


def positive_calibration_slope(margins, y):
    """Positive zero-intercept sigmoid fit; classifications cannot change."""
    margins, y = np.asarray(margins, dtype=float), np.asarray(y)
    if (margins.shape != y.shape or not len(y) or not np.isfinite(margins).all()
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Calibration needs finite aligned scores and binary outcomes")
    def loss(slope):
        scores = slope*margins
        return float(np.mean(np.logaddexp(0., scores)-y*scores))
    fitted = minimize_scalar(loss, bounds=(.05, 20.), method="bounded",
                             options={"xatol": 1e-8})
    if not fitted.success or not np.isfinite(fitted.x):
        raise ValueError("Positive calibration slope did not converge")
    return float(fitted.x)


class AccuracyMarginClassifier:
    def __init__(self, spec):
        self.spec = spec

    def _fit_margin(self, x, y, seasons):
        feature_map = OddFeatureMap(self.spec).fit(x)
        values = feature_map.transform(x)
        weights = 2.**(-(seasons.max()-seasons)/self.spec["params"]["half_life"])
        if self.spec["kind"] == "r5_logistic":
            estimator = LogisticRegression(C=self.spec["params"]["C"], fit_intercept=False,
                                           max_iter=2000, random_state=SEED)
        else:
            estimator = LinearSVC(C=self.spec["params"]["C"], loss="squared_hinge", dual=False,
                                  fit_intercept=False, max_iter=5000, tol=1e-5, random_state=SEED)
        estimator.fit(values, y, sample_weight=weights)
        return feature_map, estimator

    def fit(self, x, y, seasons, dates=None):
        y, seasons = np.asarray(y), np.asarray(seasons, dtype=float)
        if (not len(x) or len(x) != len(y) or seasons.shape != y.shape
                or not np.isfinite(seasons).all() or not np.equal(seasons, np.floor(seasons)).all()
                or np.any(seasons > 2022) or not np.isin(y, [0, 1]).all()
                or len(np.unique(y)) != 2):
            raise ValueError("Fit requires aligned binary training rows through 2022 only")
        if dates is not None:
            parsed = pd.to_datetime(np.asarray(dates), errors="raise")
            if len(parsed) != len(y) or pd.isna(parsed).any() or (parsed >= pd.Timestamp("2023-01-01")).any():
                raise ValueError("Fit dates must align and precede 2023; final holdout remains locked")
        self.calibration_slope_ = 1.
        self.calibration_metadata_ = {"policy": "native logistic probability; no calibration fit"}
        if self.spec["kind"] == "r5_squared_hinge":
            earlier, later, metadata = calibration_masks(seasons, dates)
            if len(np.unique(y[earlier])) != 2:
                raise ValueError("Earlier calibration training requires both outcome classes")
            inner_map, inner_estimator = self._fit_margin(x.iloc[earlier], y[earlier], seasons[earlier])
            margins = inner_estimator.decision_function(inner_map.transform(x.iloc[later]))
            self.calibration_slope_ = positive_calibration_slope(margins, y[later])
            self.calibration_metadata_ = {**metadata, "slope": self.calibration_slope_,
                "source": "earlier-trained margins on last training year",
                "final_refit": "All outer training rows; assume calibration transports to refit margins"}
        self.feature_map_, self.estimator_ = self._fit_margin(x, y, seasons)
        self.fit_end_season_ = int(seasons.max())
        return self

    def decision_function(self, x):
        return self.estimator_.decision_function(self.feature_map_.transform(x))

    def predict_proba(self, x):
        return np.clip(expit(self.calibration_slope_*self.decision_function(x)), 1e-8, 1-1e-8)


def make_model(spec):
    if spec["kind"] == "odd_spline":
        return make_spline(spec)
    if spec["kind"] not in ("r5_logistic", "r5_squared_hinge"):
        raise ValueError(f"Unsupported round-five learner: {spec['kind']}")
    return AccuracyMarginClassifier(spec)

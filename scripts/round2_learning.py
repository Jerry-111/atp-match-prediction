"""Round-two antisymmetric additive models with smooth, shrunk departures.

All transformations fit on the rows passed to fit; the caller owns chronological
splits. Every design column reverses sign on player exchange, so a no-intercept
logistic head gives complementary probabilities exactly. No validation scoring
or automatic tuning happens here.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer


CORE = (
    "log_rank_diff", "elo_k32_diff", "shrunk_surface_elo_k32_diff",
    "form_90_rate_diff", "h2h_rate_diff",
    "points_overall_364_serve_rate_diff", "points_overall_364_return_rate_diff",
    "points_surface_364_serve_rate_diff", "points_surface_364_return_rate_diff",
)

# Only an odd strength signal times an even context can remain antisymmetric.
INTERACTIONS = (
    ("log_rank_diff", "log_rank_mean"),
    ("elo_k32_diff", "log_rank_mean"),
    ("elo_k32_diff", "surface_is_clay"),
    ("elo_k32_diff", "surface_is_grass"),
    ("shrunk_surface_elo_k32_diff", "surface_is_clay"),
    ("shrunk_surface_elo_k32_diff", "surface_is_grass"),
    ("shrunk_surface_elo_k32_diff", "surface_career_log_count_sum"),
    ("surface_rate_diff", "surface_form_364_log_count_sum"),
    ("form_90_rate_diff", "form_90_log_count_sum"),
    ("h2h_rate_diff", "h2h_career_log_count"),
    ("points_overall_364_serve_rate_diff", "points_overall_364_serve_points_log_sum"),
    ("points_overall_364_return_rate_diff", "points_overall_364_return_points_log_sum"),
    ("points_surface_364_serve_rate_diff", "points_surface_364_serve_points_log_sum"),
    ("points_surface_364_return_rate_diff", "points_surface_364_return_points_log_sum"),
)


def candidate_specs(columns):
    """Six fixed JSON-serializable candidates, independent of any outcomes."""
    columns = list(columns)
    odd = [c for c in columns if c.endswith("_diff")]
    if not odd:
        raise ValueError("At least one antisymmetric *_diff feature is required")
    core = [c for c in CORE if c in columns]
    interactions = [[a, b] for a, b in INTERACTIONS if a in columns and b in columns]
    specs = []
    for shape, interact, half in [(0.3, False, None), (0.3, True, None), (0.3, True, 3.)]:
        for strength in (.003, .01):
            name = f"odd_spline_C{strength:g}_context{int(interact)}_half{half}"
            used_interactions = interactions if interact else []
            required = list(dict.fromkeys(odd + [b for _, b in used_interactions]))
            specs.append({"name": name, "kind": "odd_spline", "family": "odd_spline",
                          "columns": required, "odd_columns": odd, "spline_columns": core,
                          "interactions": used_interactions,
                          "params": {"C": strength, "shape_weight": shape,
                                     "interaction_weight": .3, "half_life": half}})
    return specs


def make_model(spec):
    if spec.get("kind") != "odd_spline":
        raise ValueError(f"Unsupported round-two learner: {spec.get('kind')}")
    return AntisymmetricSplineClassifier(spec)


def _rms(values):
    scale = np.sqrt(np.mean(values * values, axis=0))
    return np.where(scale > 1e-10, scale, 1.)


class AntisymmetricSplineClassifier:
    """Ridge logistic + odd quadratic splines + limited reliability interactions.

    Spline basis B is converted to [B(z)-B(-z)]/2. Inputs z are scaled using
    training RMS, with knots fixed at -3,-1.5,0,1.5,3 RMS units and constant
    extrapolation. Thus spline contributions saturate in sparse tails while
    the regularized linear component remains. Nonlinear blocks receive only
    0.3 times unit RMS scale, imposing a stronger effective ridge penalty.
    """

    def __init__(self, spec):
        self.spec = spec

    def _input(self, x):
        required = self.spec["columns"]
        if x.columns.duplicated().any() or any(c not in x for c in required):
            raise ValueError("Missing or duplicate model feature columns")
        try:
            selected = x.loc[:, required].astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("Model features must be finite numeric values") from error
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError("Model features must be finite numeric values")
        return selected

    def _design(self, x):
        odd = x.loc[:, self.spec["odd_columns"]].to_numpy() / self.odd_scale_
        # Symmetric clipping prevents a single extreme input from dominating.
        odd = np.clip(odd, -8., 8.)
        parts = [odd]
        shape = []
        for column, transformer in self.splines_.items():
            values = odd[:, [self.odd_indices_[column]]]
            shape.append(.5 * (transformer.transform(values) - transformer.transform(-values)))
        if shape:
            block = np.column_stack(shape)
            parts.append(block / self.spline_scale_ * self.spec["params"]["shape_weight"])
        if self.spec["interactions"]:
            interactions = []
            for a, b in self.spec["interactions"]:
                context = np.clip((x[b].to_numpy() - self.context_mean_[b]) / self.context_scale_[b], -3., 3.)
                interactions.append(odd[:, self.odd_indices_[a]] * context)
            block = np.column_stack(interactions)
            parts.append(block / self.interaction_scale_ * self.spec["params"]["interaction_weight"])
        return np.column_stack(parts)

    def fit(self, x, y, seasons=None):
        x = self._input(x)
        y = np.asarray(y)
        if len(x) != len(y) or len(x) == 0 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Training requires aligned rows and both binary classes")
        if seasons is not None:
            seasons = np.asarray(seasons, dtype=float)
            if seasons.shape != y.shape or not np.isfinite(seasons).all():
                raise ValueError("Seasons must be finite and aligned with training rows")
        names = self.spec["odd_columns"]
        if any(not c.endswith("_diff") for c in names):
            raise ValueError("Linear columns must be antisymmetric *_diff features")
        self.odd_indices_ = {name: i for i, name in enumerate(names)}
        self.odd_scale_ = _rms(x[names].to_numpy())
        odd = np.clip(x[names].to_numpy() / self.odd_scale_, -8., 8.)
        self.splines_ = {}
        shape = []
        for column in self.spec["spline_columns"]:
            values = odd[:, [self.odd_indices_[column]]]
            transformer = SplineTransformer(degree=2, knots=np.linspace(-3, 3, 5)[:, None],
                                             include_bias=True, extrapolation="constant")
            transformer.fit(np.concatenate([values, -values]))
            self.splines_[column] = transformer
            shape.append(.5 * (transformer.transform(values) - transformer.transform(-values)))
        self.spline_scale_ = _rms(np.column_stack(shape)) if shape else np.array([])
        self.context_mean_, self.context_scale_ = {}, {}
        interactions = []
        for a, b in self.spec["interactions"]:
            if b.endswith("_diff"):
                raise ValueError("Interaction context must be invariant under player exchange")
            values = x[b].to_numpy()
            self.context_mean_[b] = float(np.mean(values))
            self.context_scale_[b] = max(float(np.std(values)), 1e-10)
            context = np.clip((values - self.context_mean_[b]) / self.context_scale_[b], -3., 3.)
            interactions.append(odd[:, self.odd_indices_[a]] * context)
        self.interaction_scale_ = _rms(np.column_stack(interactions)) if interactions else np.array([])
        design = self._design(x)
        params = self.spec["params"]
        weights = None
        if params["half_life"] is not None:
            if seasons is None:
                raise ValueError("Recency weighting requires training seasons")
            weights = 2. ** (-(seasons.max() - seasons) / params["half_life"])
        self.estimator_ = LogisticRegression(C=params["C"], fit_intercept=False,
                                             max_iter=2000, random_state=158)
        self.estimator_.fit(design, y, sample_weight=weights)
        self.design_width_ = design.shape[1]
        return self

    def predict_proba(self, x):
        selected = self._input(x)
        p = self.estimator_.predict_proba(self._design(selected))[:, 1]
        return np.clip(p, 1e-8, 1 - 1e-8)

"""Frozen-expert Bernoulli mixtures with a small symmetric reliability gate.

This module contains generic numerical routines and temporal-mask guards. The
Round10 preparation uses synthetic data only; no empirical runner is provided.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from scripts.advanced_features import _date
from scripts.multilevel_features import _season_date

EXPERTS=('add_age_spline_C0.01','r5_linear_logistic_C0.003','realmlp_td_s','sequence_mean')
CONTEXTS=('rank_gap','career_coverage','surface_coverage','point_coverage','best_of_five')
FEATURE_FIELDS=('log_rank_abs_gap','career_min_count','surface_career_min_count',
                'points_overall_364_serve_points_min','format_best_of_five')
PROBABILITY_EPS=1e-6
WEIGHT_FLOOR=.02
PENALTY=.01
CONTEXT_CLIP=3.


def _probabilities(probabilities):
    if isinstance(probabilities,pd.DataFrame) and tuple(probabilities.columns)!=EXPERTS:
        raise ValueError('Exact ordered four-expert schema required')
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=4 or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Four finite probability columns in [0,1] required')
    return np.clip(p,PROBABILITY_EPS,1-PROBABILITY_EPS)


def _labels(y,n):
    y=np.asarray(y)
    if y.shape!=(n,) or not np.isin(y,[0,1]).all():
        raise ValueError('Aligned finite binary labels required')
    return y.astype(float)


def _alignment(probabilities,contexts=None,labels=None):
    if isinstance(probabilities,pd.DataFrame):
        if isinstance(contexts,pd.DataFrame) and not probabilities.index.equals(contexts.index):
            raise ValueError('Probability/context row indices must match')
        if isinstance(labels,pd.Series) and not probabilities.index.equals(labels.index):
            raise ValueError('Probability/label row indices must match')


def context_frame(features):
    """Five fixed even predictors; ignore all other columns, including outcomes."""
    if not isinstance(features,pd.DataFrame) or features.columns.duplicated().any():
        raise ValueError('Unique named feature columns required')
    x=features[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(x).all() or np.any(x[:,:4]<0) or not np.isin(x[:,4],[0,1]).all():
        raise ValueError('Finite nonnegative coverage/rank fields and binary format required')
    x=x.copy();x[:,1:4]=np.log1p(x[:,1:4])
    return pd.DataFrame(x,index=features.index,columns=CONTEXTS)


def _contexts(contexts):
    if not isinstance(contexts,pd.DataFrame) or tuple(contexts.columns)!=CONTEXTS:
        raise ValueError('Exact five-context schema required')
    x=contexts.to_numpy(dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty contexts required')
    return x


@dataclass
class ContextScaler:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    @classmethod
    def fit(cls,contexts):
        x=_contexts(contexts)
        mean=x.mean(axis=0);sd=x.std(axis=0)
        active=sd>1e-12
        return cls(mean,np.where(active,np.maximum(sd,1e-6),1.),active)

    def transform(self,contexts):
        x=_contexts(contexts)
        z=np.clip((x-self.mean)/self.scale,-CONTEXT_CLIP,CONTEXT_CLIP)
        z[:,~self.active]=0.
        return z


def contrast_matrix():
    """Fixed orthonormal coordinates for four zero-sum gate logits."""
    h=np.zeros((4,3))
    for j in range(3):
        m=j+1;normalizer=np.sqrt(m*(m+1))
        h[:m,j]=1/normalizer;h[m,j]=-m/normalizer
    return h


def bernoulli_loss(y,p):
    p=np.asarray(p,dtype=float)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def global_objective(weights,p,y):
    """Loss and analytic derivative with respect to simplex weights."""
    q=p@weights
    if np.any((q<=0)|(q>=1)):raise FloatingPointError('Noninterior mixture probability')
    derivative=(q-y)/(q*(1-q))
    return bernoulli_loss(y,q),p.T@derivative/len(y)


@dataclass
class GlobalMixture:
    weights: np.ndarray
    diagnostics: dict

    def predict_proba(self,probabilities):
        return _probabilities(probabilities)@self.weights


def fit_global(probabilities,y):
    _alignment(probabilities,labels=y)
    p=_probabilities(probabilities);y=_labels(y,len(p));k=p.shape[1]
    result=minimize(lambda w:global_objective(w,p,y),np.full(k,1/k),jac=True,
        method='SLSQP',bounds=[(WEIGHT_FLOOR,1-(k-1)*WEIGHT_FLOOR)]*k,
        constraints={'type':'eq','fun':lambda w:float(w.sum()-1),'jac':lambda w:np.ones(k)},
        options=dict(maxiter=1000,ftol=1e-12))
    if not result.success or abs(result.x.sum()-1)>1e-8 or result.x.min()<WEIGHT_FLOOR-1e-8:
        raise RuntimeError(f'Global simplex optimizer failed: {result.message}')
    # Repair only roundoff in feasibility, using the same floored simplex in
    # the global control and the conditional baseline. This is not a new fit.
    v=np.maximum(result.x-WEIGHT_FLOOR,0);v/=v.sum()
    weights=WEIGHT_FLOOR+(1-k*WEIGHT_FLOOR)*v
    loss,gradient=global_objective(weights,p,y)
    free=weights>WEIGHT_FLOOR+1e-7
    multiplier=float(np.mean(gradient[free]))
    kkt=max(float(np.max(np.abs(gradient[free]-multiplier))),
            float(np.max(np.maximum(multiplier-gradient[~free],0))) if (~free).any() else 0.)
    if kkt>1e-5 or abs(loss-float(result.fun))>1e-8:
        raise RuntimeError('Global simplex numerical optimality check failed')
    return GlobalMixture(weights,dict(rows=len(p),iterations=int(result.nit),objective=loss,
        max_kkt_error=kkt,optimizer='SLSQP',weight_floor=WEIGHT_FLOOR))


def conditional_objective(theta_flat,p,y,z,global_weights):
    """Mean Bernoulli loss plus 0.5*lambda*||theta||², analytic gradient."""
    theta=np.asarray(theta_flat).reshape(5,3);h=contrast_matrix()
    gates=softmax(np.log(global_weights)[None,:]+z@theta@h.T,axis=1)
    q=np.sum(gates*p,axis=1)
    loss=bernoulli_loss(y,q)+.5*PENALTY*float(np.sum(theta*theta))
    d=(q-y)/(q*(1-q))
    logits_gradient=d[:,None]*gates*(p-q[:,None])
    gradient=z.T@logits_gradient@h/len(y)+PENALTY*theta
    return loss,gradient.ravel()


@dataclass
class ConditionalMixture:
    global_weights: np.ndarray
    scaler: ContextScaler
    theta: np.ndarray
    diagnostics: dict

    def weights_for(self,contexts):
        z=self.scaler.transform(contexts)
        weights=softmax(np.log(self.global_weights)[None,:]+z@self.theta@contrast_matrix().T,axis=1)
        # Exact finite-arithmetic fallback for the mathematical z=0 identity.
        weights[np.all(z==0,axis=1)]=self.global_weights
        return weights

    def predict_proba(self,probabilities,contexts):
        _alignment(probabilities,contexts)
        p=_probabilities(probabilities)
        if len(p)!=len(contexts):raise ValueError('Probabilities and contexts must align')
        return np.sum(self.weights_for(contexts)*p,axis=1)


def fit_conditional(probabilities,y,contexts,global_model):
    _alignment(probabilities,contexts,y)
    p=_probabilities(probabilities);y=_labels(y,len(p))
    if len(contexts)!=len(p):raise ValueError('Probabilities and contexts must align')
    w=np.asarray(global_model.weights,dtype=float)
    if w.shape!=(4,) or not np.isfinite(w).all() or w.min()<WEIGHT_FLOOR-1e-12 or abs(w.sum()-1)>1e-12:
        raise ValueError('The same valid floored global-mixture control is required')
    scaler=ContextScaler.fit(contexts);z=scaler.transform(contexts)
    result=minimize(lambda b:conditional_objective(b,p,y,z,w),np.zeros(15),jac=True,
        method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-13,gtol=1e-9,maxls=50))
    objective,gradient=conditional_objective(result.x,p,y,z,w)
    if not result.success or not np.isfinite(result.x).all() or np.max(np.abs(gradient))>1e-6:
        raise RuntimeError(f'Conditional optimizer failed numerical check: {result.message}')
    model=ConditionalMixture(w.copy(),scaler,result.x.reshape(5,3),dict(rows=len(p),
        iterations=int(result.nit),objective=objective,penalty=PENALTY,
        max_abs_gradient=float(np.max(np.abs(gradient))),optimizer='L-BFGS-B',
        active_contexts=int(scaler.active.sum()),global_diagnostics=dict(global_model.diagnostics)))
    model.diagnostics['unpenalized_training_log_loss']=bernoulli_loss(y,model.predict_proba(p,contexts))
    return model


def equal_mixture(probabilities):return _probabilities(probabilities).mean(axis=1)


def meta_training_mask(oof_rows,target_rows):
    """2020 is a warmup; no learned-meta 2020 test is manufactured."""
    if oof_rows.empty or target_rows.empty or oof_rows.match_id.duplicated().any() or target_rows.match_id.duplicated().any():
        raise ValueError('Unique nonempty meta/query IDs required')
    if not pd.to_numeric(oof_rows.season,errors='coerce').isin((2020,2021,2022)).all():
        raise ValueError('Meta training source is limited to 2020–2022 OOF')
    years=pd.to_numeric(target_rows.season,errors='coerce').unique()
    if len(years)!=1 or years[0] not in (2021,2022,2023):
        raise ValueError('First honest meta test is 2021; final years are sealed')
    for frame in (oof_rows,target_rows):
        for day,season in zip(frame.tourney_date,frame.season):_season_date(_date(day),int(season))
    cutoff=pd.Timestamp(min(target_rows.tourney_date.map(_date)))-pd.Timedelta(days=21)
    dates=pd.to_datetime(oof_rows.tourney_date)
    mask=oof_rows.season.lt(int(years[0]))&dates.le(cutoff)
    if not mask.any():raise ValueError('No earlier purged meta-training observations')
    if set(oof_rows.loc[mask,'match_id'])&set(target_rows.match_id):
        raise ValueError('Meta train/query identities overlap')
    return mask.to_numpy(),dict(target_year=int(years[0]),cutoff=cutoff.date().isoformat(),
        training_ids=oof_rows.loc[mask,'match_id'].tolist(),query_ids=target_rows.match_id.tolist(),
        training_rows=int(mask.sum()),query_rows=len(target_rows),
        purged_rows=int((oof_rows.season.lt(int(years[0]))&~mask).sum()))

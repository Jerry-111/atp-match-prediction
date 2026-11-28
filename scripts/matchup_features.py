"""Train-only antisymmetric style interactions and strongly regularized heads."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.train_distillation_round7 import validate_rows

STYLE_NAMES=('ace','double_fault','first_in','first_won','second_won',
             'return_first_won','return_second_won')
STYLE_COLUMNS=[f'style_{s}_{suffix}' for s in STYLE_NAMES for suffix in ('diff','sum')]
SURFACES=('hard','clay','grass')
SURFACE_COLUMNS=[f'surface_is_{s}' for s in SURFACES]
PAIRS=tuple(combinations(range(len(STYLE_NAMES)),2))
SHARED_NAMES=[f'matchup_{STYLE_NAMES[i]}_versus_{STYLE_NAMES[j]}_diff' for i,j in PAIRS]
SURFACE_NAMES=[f'matchup_{surface}_{STYLE_NAMES[i]}_versus_{STYLE_NAMES[j]}_diff'
               for surface in SURFACES for i,j in PAIRS]
SHARED_FACTOR=.2
SURFACE_FACTOR=.1


def reconstruct_styles(x):
    if x.empty or x.columns.duplicated().any() or not set(STYLE_COLUMNS).issubset(x):
        raise ValueError('Complete unique nonempty style-rate pairs required')
    diff=x[[f'style_{s}_diff' for s in STYLE_NAMES]].to_numpy(dtype=float)
    total=x[[f'style_{s}_sum' for s in STYLE_NAMES]].to_numpy(dtype=float)
    a,b=(total+diff)/2,(total-diff)/2
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < -1e-12) or np.any(a > 1+1e-12)
            or np.any(b < -1e-12) or np.any(b > 1+1e-12)):
        raise ValueError('Reconstructed style rates must be finite and in [0,1]')
    return np.clip(a,0,1),np.clip(b,0,1)


def skew_products(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=7 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite aligned seven-dimensional player styles required')
    return np.column_stack([a[:,i]*b[:,j]-a[:,j]*b[:,i] for i,j in PAIRS])


class MatchupTransformer:
    """An observed-feature skew basis; no player identity embeddings."""
    def fit(self,x):
        a,b=reconstruct_styles(x)
        players=np.concatenate([a,b])
        self.center_=players.mean(axis=0)
        scale=players.std(axis=0)
        self.scale_=np.where(scale>1e-10,scale,1.)
        shared=self._shared(x)
        rms=np.sqrt(np.mean(shared**2,axis=0))
        self.interaction_rms_=np.where(rms>1e-10,rms,1.)
        self.training_rows_=len(x)
        return self

    def _shared(self,x):
        a,b=reconstruct_styles(x)
        a=np.clip((a-self.center_)/self.scale_,-4,4)
        b=np.clip((b-self.center_)/self.scale_,-4,4)
        return skew_products(a,b)

    def transform(self,x):
        if not set(SURFACE_COLUMNS).issubset(x):
            raise ValueError('Known surface indicators required')
        indicators=x[SURFACE_COLUMNS].to_numpy(dtype=float)
        if not np.isin(indicators,(0,1)).all() or np.any(indicators.sum(axis=1)>1):
            raise ValueError('Surface indicators must be zero or one-hot')
        z=self._shared(x)/self.interaction_rms_
        # Scale AFTER RMS, and never pass these blocks through another scaler.
        surface=[SURFACE_FACTOR*z*indicators[:,[j]] for j in range(3)]
        return pd.DataFrame(np.column_stack([SHARED_FACTOR*z,*surface]),
                            columns=SHARED_NAMES+SURFACE_NAMES,index=x.index)


def candidate_specs(columns):
    columns=list(columns)
    odd=[c for c in columns if c.endswith('_diff')]
    if len(columns)!=len(set(columns)) or not odd or not set(STYLE_COLUMNS+SURFACE_COLUMNS).issubset(columns):
        raise ValueError('Unique baseline columns and existing styles/surfaces required')
    return [dict(name=f'matchup_{structure}',structure=structure,main_columns=odd,
        columns=list(dict.fromkeys(odd+STYLE_COLUMNS+SURFACE_COLUMNS)),
        C=.01,half_life=3.,shared_factor=SHARED_FACTOR,surface_factor=SURFACE_FACTOR)
        for structure in ('no_new_interaction','shared','surface')]


class MatchupClassifier:
    def __init__(self,spec):
        self.spec=spec

    def _input(self,x):
        required=self.spec['columns']
        if x.empty or x.columns.duplicated().any() or not set(required).issubset(x):
            raise ValueError('Complete finite frozen input columns required')
        selected=x[required].astype(float)
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError('Complete finite frozen input columns required')
        return selected

    def design(self,x):
        x=self._input(x)
        main=x[self.spec['main_columns']].to_numpy()/self.main_scale_
        if self.spec['structure']=='no_new_interaction':return main
        interactions=self.transformer_.transform(x)
        names=SHARED_NAMES if self.spec['structure']=='shared' else SHARED_NAMES+SURFACE_NAMES
        return np.column_stack([main,interactions[names].to_numpy()])

    def fit(self,x,rows):
        if not x.index.equals(rows.index):raise ValueError('Aligned feature and training rows required')
        validate_rows(rows,fitting=True)
        y=rows.a_won.to_numpy()
        if not np.isin(y,(0,1)).all() or len(np.unique(y))!=2:
            raise ValueError('Both original binary training labels required')
        if self.spec['structure'] not in ('no_new_interaction','shared','surface'):
            raise ValueError('Unknown fixed matchup structure')
        if any(not n.endswith('_diff') for n in self.spec['main_columns']):
            raise ValueError('Main-effect columns must be odd')
        if self.spec['shared_factor']!=SHARED_FACTOR or self.spec['surface_factor']!=SURFACE_FACTOR:
            raise ValueError('Fixed interaction penalties changed')
        x=self._input(x)
        main=x[self.spec['main_columns']].to_numpy()
        # Same mirrored population and variance calculation as the old linear
        # baseline. No centering: odd inputs must remain exactly odd.
        self.main_scale_=StandardScaler(with_mean=False).fit(np.concatenate([main,-main])).scale_
        self.transformer_=MatchupTransformer().fit(x) if self.spec['structure']!='no_new_interaction' else None
        design=self.design(x)
        weights=np.exp2(-(rows.season.max()-rows.season.to_numpy())/self.spec['half_life'])
        self.estimator_=LogisticRegression(C=self.spec['C'],fit_intercept=False,max_iter=2000,random_state=158)
        self.estimator_.fit(np.concatenate([design,-design]),np.r_[y,1-y],sample_weight=np.tile(weights/2,2))
        self.training_rows_=len(rows)
        self.fit_max_season_=int(rows.season.max())
        self.feature_names_=list(self.spec['columns'])
        self.design_names_=list(self.spec['main_columns'])+(SHARED_NAMES if self.spec['structure']=='shared'
            else SHARED_NAMES+SURFACE_NAMES if self.spec['structure']=='surface' else [])
        return self

    def predict_proba(self,x):
        return np.clip(self.estimator_.predict_proba(self.design(x))[:,1],1e-8,1-1e-8)

    def contributions(self,x):
        values=self.design(x)*self.estimator_.coef_[0]
        width=len(self.spec['main_columns'])
        result={'main':values[:,:width].sum(axis=1)}
        result['shared']=values[:,width:width+21].sum(axis=1) if values.shape[1]>width else np.zeros(len(x))
        result['surface']=values[:,width+21:].sum(axis=1) if values.shape[1]>width+21 else np.zeros(len(x))
        return pd.DataFrame(result,index=x.index)

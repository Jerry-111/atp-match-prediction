"""Chronological, symmetric adaptation of the official RealMLP-TD-S recipe.

Upstream: dholzmueller/realmlp-td-s_standalone at
a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e (mlp.py, preprocessing.py).
The scaling, NTP layers, TD-S network and optimizer schedule are adapted from:

Copyright (c) 2024- David Holzmüller

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

from datetime import timedelta
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.advanced_features import _date
from scripts.multilevel_features import _season_date

UPSTREAM_REVISION = 'a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e'
MAX_EPOCHS = 256
SEEDS = (1741, 1742, 1743)
BATCH_SIZE = 256


def candidate_specs():
    return [dict(name=name, kind=kind, seeds=list(SEEDS), max_epochs=MAX_EPOCHS,
                 hidden_sizes=[256,256,256], batch_size=BATCH_SIZE)
            for name,kind in [('realmlp_td_s','realmlp'),('plain_mlp','plain')]]


def configure_device(device='cpu'):
    torch.set_num_threads(4)
    if device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable; explicitly choose CPU')
        torch.mps.set_per_process_memory_fraction(.55)
    elif device != 'cpu':
        raise ValueError('Only audited CPU/MPS devices are supported')
    return device


def mirror(x, columns):
    a = np.asarray(x,dtype=np.float64).copy()
    if a.ndim != 2 or a.shape[1] != len(columns):
        raise ValueError('Feature schema mismatch')
    a[:,[str(c).endswith('_diff') for c in columns]] *= -1
    return a


class RobustScaleSmoothClip:
    """Author's numeric transform; fit only on the mirrored training cohort."""
    def fit(self,x):
        x = np.asarray(x,dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Finite nonempty training features required')
        self.median_ = np.median(x,axis=0)
        spread = np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
        fallback = .5*(x.max(axis=0)-x.min(axis=0))
        spread = np.where(spread==0,fallback,spread)
        self.factors_ = np.divide(1,spread+1e-30,where=spread!=0,
                                  out=np.zeros_like(spread))
        return self

    def transform(self,x):
        x = np.asarray(x,dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.median_) or not np.isfinite(x).all():
            raise ValueError('Finite features matching fitted schema required')
        scaled = (x-self.median_)*self.factors_
        return (scaled/np.sqrt(1+(scaled/3)**2)).astype(np.float32)


class ScalingLayer(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))
    def forward(self,x):
        return x*self.scale[None,:]


class NTPLinear(nn.Module):
    def __init__(self,in_features,out_features,zero_init=False):
        super().__init__()
        self.in_features = in_features
        factor = 0. if zero_init else 1.
        self.weight = nn.Parameter(factor*torch.randn(in_features,out_features))
        self.bias = nn.Parameter(factor*torch.randn(1,out_features))
    def forward(self,x):
        return (x@self.weight)/math.sqrt(self.in_features)+self.bias


def make_network(kind,input_dim):
    if kind == 'realmlp':
        return nn.Sequential(ScalingLayer(input_dim),NTPLinear(input_dim,256),nn.SELU(),
            NTPLinear(256,256),nn.SELU(),NTPLinear(256,256),nn.SELU(),
            NTPLinear(256,2,zero_init=True))
    if kind == 'plain':
        return nn.Sequential(nn.Linear(input_dim,256),nn.ReLU(),nn.Linear(256,256),
            nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,2))
    raise ValueError('Unknown frozen network recipe')


def schedule_value(progress):
    return .5-.5*np.cos(2*np.pi*np.log2(1+15*progress))


def _optimizer(model,kind):
    if kind == 'realmlp':
        params = list(model.parameters())
        return torch.optim.Adam([dict(params=params[:1]),dict(params=params[1::2]),
                                 dict(params=params[2::2])],betas=(.9,.95))
    return torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999))


def _raw_predict(model,x,device,batch_size=1024):
    model.eval()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            logits = model(torch.as_tensor(x[start:start+batch_size],device=device))
            values.append(torch.softmax(logits,dim=-1)[:,1].cpu().numpy().astype(float))
    return np.concatenate(values) if values else np.empty(0)


def _symmetric_predict(model,prep,x,columns,device,batch_size=1024):
    a = _raw_predict(model,prep.transform(x),device,batch_size)
    b = _raw_predict(model,prep.transform(mirror(x,columns)),device,batch_size)
    return .5*(a+1-b)


def train_network(x,y,columns,kind,seed,epochs,*,device='cpu',validation=None,
                  schedule_epochs=MAX_EPOCHS, progress=None):
    """One seed; validation selects epochs only, never changes the schedule horizon.

    Refits intentionally stop the original 256-epoch schedule at the inner-selected
    epoch. Compressing it to the selected duration would fit a different schedule.
    """
    if not 1 <= epochs <= schedule_epochs or kind not in {'realmlp','plain'}:
        raise ValueError('Invalid fixed training recipe')
    x,y=np.asarray(x,dtype=float),np.asarray(y)
    if y.ndim!=1 or len(x)!=len(y) or not np.isin(y,[0,1]).all() or len(x)<2:
        raise ValueError('Aligned binary training labels required')
    y=y.astype(int)
    if validation is not None:
        vx,vy=np.asarray(validation[0],dtype=float),np.asarray(validation[1])
        if vy.ndim!=1 or len(vx)!=len(vy) or not len(vy) or not np.isin(vy,[0,1]).all():
            raise ValueError('Aligned nonempty binary inner validation labels required')
        validation=(vx,vy.astype(int))
    torch.manual_seed(int(seed))
    generator=torch.Generator(device='cpu').manual_seed(int(seed)+10000)
    aug=np.concatenate([x,mirror(x,columns)])
    prep=RobustScaleSmoothClip().fit(aug)
    xx=torch.as_tensor(prep.transform(aug),device=device)
    yy=torch.as_tensor(np.concatenate([y,1-y]),dtype=torch.long,device=device)
    model=make_network(kind,x.shape[1]).to(device)
    optimizer=_optimizer(model,kind)
    criterion=nn.CrossEntropyLoss(label_smoothing=.1 if kind=='realmlp' else 0.)
    batch_size=min(BATCH_SIZE,len(xx))
    n_batches=len(xx)//batch_size
    best_loss=float('inf'); best_epoch=epochs; losses=[]
    started=time.monotonic()
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(xx),generator=generator).to(device)
        for batch in range(n_batches):
            if kind=='realmlp':
                t=(epoch*n_batches+batch)/(schedule_epochs*n_batches)
                lr=.04*schedule_value(t)
                for group,mult in zip(optimizer.param_groups,[6.,1.,.1]):
                    group['lr']=float(mult*lr)
            indices=order[batch*batch_size:(batch+1)*batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(xx[indices]),yy[indices])
            loss.backward(); optimizer.step()
        if validation is not None:
            vx,vy=validation
            p=np.clip(_symmetric_predict(model,prep,vx,columns,device),1e-7,1-1e-7)
            vy=np.asarray(vy)
            value=float(np.mean(-vy*np.log(p)-(1-vy)*np.log1p(-p)))
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite inner loss')
            losses.append(value)
            if value<=best_loss:
                best_loss=value; best_epoch=epoch+1
        if progress is not None and ((epoch+1)%32==0 or epoch+1==epochs):
            progress(dict(epoch=epoch+1,seconds=time.monotonic()-started,
                          inner_loss=None if not losses else losses[-1]))
    state={name:tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise FloatingPointError('Nonfinite trained parameter')
    result=dict(kind=kind,seed=int(seed),prep=prep,state_dict=state,epochs=int(epochs),
        selected_epoch=int(best_epoch),inner_losses=losses,
        elapsed_seconds=time.monotonic()-started,schedule_epochs=int(schedule_epochs),
        parameter_count=sum(p.numel() for p in model.parameters()))
    del model,optimizer,xx,yy
    if device=='mps': torch.mps.empty_cache()
    return result


def inner_masks(rows):
    """The last eligible training season is exclusively an inner stopping tail."""
    if rows.empty or rows.match_id.duplicated().any():
        raise ValueError('Unique nonempty training rows required')
    for day,season in zip(rows.tourney_date,rows.season):
        if int(season) not in range(2009,2023):
            raise ValueError('Network fitting is limited to seasons 2009–2022')
        _season_date(_date(day),int(season))
    last=int(rows.season.max())
    tail=rows.season.eq(last).to_numpy()
    cutoff=min(rows.loc[tail,'tourney_date'].map(_date))-timedelta(days=21)
    earlier=(rows.season.lt(last)&rows.tourney_date.map(_date).le(cutoff)).to_numpy()
    if not earlier.any() or not tail.any():
        raise ValueError('An earlier inner-training period and last-year tail are required')
    return earlier,tail,dict(tail_season=last,purge_cutoff=cutoff.isoformat(),
        train_ids=rows.loc[earlier,'match_id'].tolist(),tail_ids=rows.loc[tail,'match_id'].tolist(),
        train_rows=int(earlier.sum()),tail_rows=int(tail.sum()),
        purged_rows=int((~earlier&~tail).sum()))


def fit_ensemble(x,rows,spec,*,device='cpu',progress=None):
    if len(x)!=len(rows) or not x.index.equals(rows.index):
        raise ValueError('Features and target rows must be aligned')
    earlier,tail,metadata=inner_masks(rows)
    columns=list(x.columns); values=x.to_numpy(dtype=float); y=rows.a_won.to_numpy()
    members=[]; tuning=[]
    for seed in spec['seeds']:
        if progress: progress(dict(seed=seed,stage='inner_start'))
        probe=train_network(values[earlier],y[earlier],columns,spec['kind'],seed,
            spec['max_epochs'],device=device,validation=(values[tail],y[tail]),progress=progress)
        epoch=probe['selected_epoch']
        tuning.append({k:v for k,v in probe.items() if k not in {'state_dict','prep'}})
        del probe
        if progress: progress(dict(seed=seed,stage='refit_start',epochs=epoch))
        members.append(train_network(values,y,columns,spec['kind'],seed,epoch,
                                     device=device,progress=progress))
    return dict(spec=spec,columns=columns,members=members,inner_metadata=metadata,
                inner_tuning=tuning,training_ids=rows.match_id.tolist())


def predict_ensemble(bundle,x,*,device='cpu',batch_size=1024):
    columns=bundle['columns']
    if not isinstance(x,pd.DataFrame) or list(x.columns)!=columns:
        raise ValueError('Exact ordered fitted feature schema required')
    predictions=[]
    for member in bundle['members']:
        model=make_network(member['kind'],len(columns)).to(device)
        model.load_state_dict(member['state_dict'])
        predictions.append(_symmetric_predict(model,member['prep'],x.to_numpy(dtype=float),
                                              columns,device,batch_size))
        del model
    result=np.mean(predictions,axis=0)
    if not np.isfinite(result).all() or np.any((result<0)|(result>1)):
        raise FloatingPointError('Invalid neural probability')
    return result

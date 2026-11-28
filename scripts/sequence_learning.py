"""Small exchangeable history encoders with chronological epoch selection.

Optional Torch dependency: import/run in .venv-foundation only.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from scripts.sequence_features import COLUMNS,TOKENS,SequenceFeatures
from scripts.train_distillation_round7 import validate_rows
from scripts.train_models import mean_loss

ARCHITECTURES=('sequence_mean','sequence_gru','sequence_shuffled_gru')
SEEDS=(158,2718)
HIDDEN=16
BATCH=512
MAX_EPOCHS=30


def check_sequence(sequence):
    n=len(sequence.values)
    if (sequence.values.shape!=(n,2,TOKENS,len(COLUMNS))
            or sequence.mask.shape!=(n,2,TOKENS) or sequence.dates.shape!=(n,2,TOKENS)
            or tuple(sequence.columns)!=COLUMNS or sequence.mask.dtype!=bool
            or not np.isfinite(sequence.values[sequence.mask]).all()):
        raise ValueError('Finite aligned 16x22 masked player histories required')
    # Histories are left-padded, not arbitrarily gapped.
    if np.any(np.diff(sequence.mask.astype(int),axis=-1)<0):
        raise ValueError('Valid sequence tokens must be a contiguous suffix')


@dataclass
class SequenceNormalizer:
    static_columns: tuple
    static_center: np.ndarray
    static_scale: np.ndarray
    token_center: np.ndarray
    token_scale: np.ndarray
    training_rows: int
    training_valid_tokens: int

    @classmethod
    def fit(cls,x,sequence):
        check_sequence(sequence)
        if len(x)!=len(sequence.values) or not np.array_equal(x.index.to_numpy(),sequence.indices):
            raise ValueError('Training features and sequence rows must align')
        values=x.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or x.columns.duplicated().any():
            raise ValueError('Finite unique static features required')
        odd=np.array([c.endswith('_diff') for c in x])
        center=values.mean(0);center[odd]=0
        scale=np.sqrt(np.mean((values-center)**2,axis=0)).clip(1e-6)
        tokens=sequence.values[sequence.mask].astype(np.float64)
        tc=tokens.mean(0) if len(tokens) else np.zeros(len(COLUMNS))
        ts=tokens.std(0).clip(1e-6) if len(tokens) else np.ones(len(COLUMNS))
        return cls(tuple(x.columns),center,scale,tc,ts,len(x),len(tokens))

    def transform(self,x,sequence):
        check_sequence(sequence)
        if (len(x)!=len(sequence.values) or not np.array_equal(x.index.to_numpy(),sequence.indices)
                or x.columns.duplicated().any() or not set(self.static_columns).issubset(x)):
            raise ValueError('Frozen input schema or row alignment mismatch')
        static=x[list(self.static_columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(static).all():raise ValueError('Nonfinite static input')
        static=((static-self.static_center)/self.static_scale).astype(np.float32)
        tokens=((sequence.values-self.token_center)/self.token_scale).astype(np.float32)
        tokens[~sequence.mask]=0
        if not np.isfinite(static).all() or not np.isfinite(tokens).all():
            raise ValueError('Nonfinite normalized input')
        return static,tokens,sequence.mask.copy()


def shuffle_table():
    table=np.tile(np.arange(TOKENS),(TOKENS+1,1))
    for n in range(2,TOKENS+1):
        table[n,-n:]=TOKENS-n+np.random.default_rng(901+n).permutation(n)
    return table


class SequenceNet(nn.Module):
    def __init__(self,architecture,static_columns):
        super().__init__()
        if architecture not in ARCHITECTURES:raise ValueError('Unknown sequence architecture')
        self.architecture=architecture
        self.projection=nn.Linear(len(COLUMNS),HIDDEN)
        self.cell=nn.GRUCell(HIDDEN,HIDDEN) if architecture!='sequence_mean' else None
        self.head=nn.Sequential(nn.Linear(len(static_columns)+2*HIDDEN,32),nn.Tanh(),nn.Linear(32,1))
        self.register_buffer('parity',torch.tensor([-1. if c.endswith('_diff') else 1. for c in static_columns]))
        self.register_buffer('shuffle_indices',torch.tensor(shuffle_table(),dtype=torch.long))

    def encode(self,tokens,mask):
        n=tokens.shape[0]
        if self.architecture=='sequence_shuffled_gru':
            order=self.shuffle_indices[mask.sum(-1)]
            performance=torch.gather(tokens[:,:,:-2],1,order[:,:,None].expand(-1,-1,len(COLUMNS)-2))
            tokens=torch.cat([performance,tokens[:,:,-2:]],dim=-1)
        projected=torch.tanh(self.projection(tokens))
        if self.cell is None:
            state=(projected*mask[:,:,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None]
        else:
            state=torch.zeros((n,HIDDEN),dtype=tokens.dtype,device=tokens.device)
            for step in range(TOKENS):
                proposed=self.cell(projected[:,step],state)
                state=torch.where(mask[:,step,None],proposed,state)
        return state*mask.any(-1)[:,None]

    def forward(self,static,tokens,mask):
        n=len(static)
        encoded=self.encode(tokens.reshape(n*2,TOKENS,len(COLUMNS)),mask.reshape(n*2,TOKENS)).reshape(n,2,HIDDEN)
        original=self.head(torch.cat([static,encoded[:,0],encoded[:,1]],dim=1))
        reverse=self.head(torch.cat([static*self.parity,encoded[:,1],encoded[:,0]],dim=1))
        return ((original-reverse)/2).squeeze(-1)


def inner_masks(rows):
    dates=validate_rows(rows,fitting=True)
    latest=int(rows.season.max())
    valid=rows.season.eq(latest)
    cutoff=dates[valid].min()-pd.Timedelta(days=21)
    train=rows.season.lt(latest)&dates.le(cutoff)
    if not train.any() or not valid.any() or (train&valid).any():
        raise ValueError('Need nonempty purged chronological inner train/tail')
    return train.to_numpy(),valid.to_numpy(),cutoff


def recency_weights(rows):
    seasons=rows.season.to_numpy(dtype=float)
    weights=2.**(-(seasons.max()-seasons)/3.)
    return (weights/weights.mean()).astype(np.float32)


def prepare_arrays(normalizer,x,sequence,device):
    arrays=normalizer.transform(x,sequence)
    return tuple(torch.as_tensor(np.ascontiguousarray(a),device=device) for a in arrays)


def network_logits(model,arrays,batch_size=1024):
    model.eval();chunks=[]
    with torch.no_grad():
        for start in range(0,len(arrays[0]),batch_size):
            chunks.append(model(*(a[start:start+batch_size] for a in arrays)).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks)


def optimize(model,arrays,labels,weights,*,seed,epochs,validation=None):
    if epochs<1:raise ValueError('Positive epoch count required')
    device=arrays[0].device
    y=torch.as_tensor(np.asarray(labels,dtype=np.float32),device=device)
    w=torch.as_tensor(np.asarray(weights,dtype=np.float32),device=device)
    if len(y)!=len(arrays[0]) or not torch.isfinite(y).all() or not torch.isin(y,torch.tensor([0.,1.],device=device)).all():
        raise ValueError('Aligned binary training labels required')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(seed);history=[];best_epoch=1;best=float('inf');patience_best=float('inf');stale=0
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;mass=0
        for start in range(0,len(y),BATCH):
            # One common fresh permutation per epoch, not one per batch.
            if start==0:order=rng.permutation(len(y))
            idx=torch.as_tensor(order[start:start+BATCH],device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(*(a[idx] for a in arrays))
            loss=(F.binary_cross_entropy_with_logits(logits,y[idx],reduction='none')*w[idx]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite sequence training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach().cpu())*len(idx);mass+=len(idx)
        item=dict(epoch=epoch,training_loss=loss_sum/mass)
        if validation is not None:
            valid_arrays,valid_labels=validation
            loss=mean_loss(valid_labels,expit(network_logits(model,valid_arrays)))
            item['inner_log_loss']=loss
            if loss<best:best,best_epoch=loss,epoch
            if loss<patience_best-1e-5:patience_best,stale=loss,0
            else:stale+=1
        history.append(item)
        if validation is not None and epoch>=8 and stale>=5:break
    return history,best_epoch if validation is not None else epochs


def fresh_network(architecture,columns,seed,device):
    torch.manual_seed(seed)
    return SequenceNet(architecture,columns).to(device)


@dataclass
class FittedSequence:
    architecture: str
    seed: int
    normalizer: SequenceNormalizer
    state: dict
    metadata: dict

    def predict_proba(self,x,sequence,*,device='cpu',batch_size=1024):
        network=SequenceNet(self.architecture,self.normalizer.static_columns)
        network.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in self.state.items()})
        network.to(device)
        return expit(network_logits(network,prepare_arrays(self.normalizer,x,sequence,device),batch_size))


def fit_sequence(architecture,seed,x,rows,sequence,*,device='cpu'):
    dates=validate_rows(rows,fitting=True)
    if (not rows.index.equals(x.index) or not np.array_equal(rows.index.to_numpy(),sequence.indices)
            or not np.isin(rows.a_won,[0,1]).all()):
        raise ValueError('Features, sequences and original labels must align')
    started=time.perf_counter();train,valid,cutoff=inner_masks(rows)
    inner_normalizer=SequenceNormalizer.fit(x[train],sequence.take(train))
    training_arrays=prepare_arrays(inner_normalizer,x[train],sequence.take(train),device)
    validation_arrays=prepare_arrays(inner_normalizer,x[valid],sequence.take(valid),device)
    network=fresh_network(architecture,x.columns,seed,device)
    history,epochs=optimize(network,training_arrays,rows.a_won.to_numpy()[train],recency_weights(rows[train]),
        seed=seed,epochs=MAX_EPOCHS,validation=(validation_arrays,rows.a_won.to_numpy()[valid]))
    del network,training_arrays,validation_arrays
    normalizer=SequenceNormalizer.fit(x,sequence)
    network=fresh_network(architecture,x.columns,seed,device)
    arrays=prepare_arrays(normalizer,x,sequence,device)
    refit_history,_=optimize(network,arrays,rows.a_won.to_numpy(),recency_weights(rows),seed=seed,epochs=epochs)
    state={k:v.detach().cpu().numpy().copy() for k,v in network.state_dict().items()}
    metadata=dict(architecture=architecture,seed=seed,selected_epoch=epochs,
        inner_training_rows=int(train.sum()),inner_validation_rows=int(valid.sum()),
        inner_cutoff=str(cutoff.date()),inner_validation_season=int(rows.season.max()),
        inner_training_last_date=str(dates[train].max().date()),refit_rows=len(rows),
        refit_last_date=str(dates.max().date()),inner_training_valid_tokens=inner_normalizer.training_valid_tokens,
        refit_valid_tokens=normalizer.training_valid_tokens,inner_history=history,refit_history=refit_history,
        parameter_count=sum(p.numel() for p in network.parameters()),device=str(device),
        inner_weight_mean=float(recency_weights(rows[train]).mean()),refit_weight_mean=float(recency_weights(rows).mean()),
        elapsed_seconds=time.perf_counter()-started,
        inner_static_center=inner_normalizer.static_center.tolist(),inner_static_scale=inner_normalizer.static_scale.tolist(),
        inner_token_center=inner_normalizer.token_center.tolist(),inner_token_scale=inner_normalizer.token_scale.tolist())
    return FittedSequence(architecture,seed,normalizer,state,metadata)


def predict_average(models,x,sequence,*,device='cpu'):
    if len(models)!=2 or tuple(m.seed for m in models)!=SEEDS or len({m.architecture for m in models})!=1:
        raise ValueError('Exactly the two fixed same-architecture seed models required')
    return np.mean([m.predict_proba(x,sequence,device=device) for m in models],axis=0)

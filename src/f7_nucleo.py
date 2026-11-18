"""Núcleo F7: reponderación, estimador de razón y calibración originales."""
import hashlib
import json
import math

import numpy as np

from rutas import PUBLICO

CONFIG = json.loads((PUBLICO/'config/F7_v1.json').read_text())
METRICS = ['RNX','delta_aj','S_aj']
PAIRS = [('PCA','Isomap'),('PCA','UMAP'),('Isomap','UMAP')]
NAMES = [f'{m}:{a}-{b}' for m in METRICS for a,b in PAIRS]


def evaluate(weights, counts, sums):
    denominator=weights@counts
    if (denominator<=0).any():
        raise ValueError('Réplica sin cobertura')
    flat=weights@sums.reshape(len(counts),-1)
    ratio=flat.reshape(len(weights),counts.shape[1],sums.shape[2])/denominator[:,:,None]
    return ratio.mean(axis=1)


def draw_weights(counts, groups, size, seed, mechanism):
    rng=np.random.default_rng(seed)
    d=len(counts)
    rejected=[]
    attempted=0
    output=[]
    probabilities=np.full(d,1/d)
    while len(output)<size:
        remaining=size-len(output)
        if mechanism=='multinomial':
            batch=rng.multinomial(d,probabilities,size=remaining)
        elif mechanism=='exponencial':
            batch=rng.exponential(size=(remaining,d))
            assert np.all(batch>0)
        else:
            raise ValueError(mechanism)
        absent=batch@counts==0
        for i in np.flatnonzero(absent.any(axis=1)):
            rejected.append(dict(intento=attempted+int(i)+1,grupos_ausentes='|'.join(groups[absent[i]])))
        output.extend(batch[~absent.any(axis=1)])
        attempted+=len(batch)
    assert attempted==size+len(rejected)
    return np.stack(output),attempted,rejected


def order_stat(values,p):
    r=math.ceil(p*len(values))
    return float(np.sort(values)[r-1]),r


def calibrate(theta, reps):
    se=np.std(reps,axis=0,ddof=CONFIG['ddof_SE'])
    invalid=(~np.isfinite(se)) | (se <= CONFIG['tolerancia_SE_relativa']*np.maximum(1,abs(theta)))
    if invalid.any():
        raise ValueError(f'SE degenerado: {np.asarray(NAMES)[invalid].tolist()} {se[invalid]}')
    t=np.max(abs((reps-theta)/se),axis=1)
    critical,r=order_stat(t,CONFIG['confianza'])
    halves=[order_stat(h,CONFIG['confianza'])[0] for h in np.split(t,2)]
    return se,critical,r,halves


def hash_weights(weights):
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


"""F4 y F5. Evaluación posterior a la selección y eventos por ocurrencia."""
import math

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

from f0 import digest, verify_hashes
import json
from geometria import vecinos
from rutas import EXP, REVISION

CFG = REVISION / 'config'
OUT = REVISION / 'resultados'


def read(path):
    return json.loads(path.read_text())


def events(original, projected, labels, k):
    a, b = original[:, :k], projected[:, :k]
    shared = a[:, :, None] == b[:, None, :]
    retained_a, retained_b = shared.any(axis=2), shared.any(axis=1)
    match_a = labels[a] == labels[:, None]
    match_b = labels[b] == labels[:, None]
    retained = retained_a.sum(axis=1)
    same_retained = (match_a & retained_a).sum(axis=1)
    a_u = match_a.sum(axis=1)
    a_z = match_b.sum(axis=1)
    dropped = (match_a & ~retained_a).sum(axis=1)
    added = (match_b & ~retained_b).sum(axis=1)
    m = k - retained
    assert np.array_equal(a_u, same_retained+dropped)
    assert np.array_equal(a_z, same_retained+added)
    assert np.array_equal(m, (~retained_b).sum(axis=1))
    return retained, same_retained, a_u, a_z, m, dropped, added, match_a


def group_metrics(x, z, original, projected, labels, k, trust):
    n = len(x)
    retained, same_retained, a_u, a_z, m, dropped, added, match_a = events(
        original, projected, labels, k)
    freq = (labels[:, None] == labels[None, :]).sum(axis=1)
    pi = float((freq-1).mean()/(n-1))
    pmax = float(np.minimum(k, freq-1).mean()/k)
    denom = pmax-pi
    assert denom > 0 and (k >= freq.min() or pmax == 1)
    c = k/(n-1)
    pu, pz = a_u/k, a_z/k
    q = (freq-1-a_u)/(n-1-k)
    delta = (added-dropped)/k
    blind = m*(q-pu)/k
    farthest = (match_a & (np.arange(k)[None, :] >= k-m[:, None])).sum(axis=1)
    blind_tail = (m*q-farthest)/k
    assert np.allclose(delta, pz-pu, atol=1e-12, rtol=0)
    per_occurrence = dict(ret=retained, ret_mismo=same_retained, a_u=a_u, a_z=a_z, m=m,
                          out_mismo=dropped, in_mismo=added, q=q, R=retained/k,
                          RNX=(retained/k-c)/(1-c), P_original=pu, P=pz,
                          P_original_aj=(pu-pi)/denom, P_aj=(pz-pi)/denom, delta=delta,
                          delta_aj=delta/denom, nula=blind, S=delta-blind,
                          S_aj=(delta-blind)/denom, S_rango=delta-blind_tail,
                          S_rango_aj=(delta-blind_tail)/denom)
    metrics = {name: float(values.mean()) for name, values in per_occurrence.items()}
    metrics.update(pi=pi, P_max=pmax,
                   T=trustworthiness(x, z, n_neighbors=k) if trust else np.nan)
    return metrics, per_occurrence


def pooled(data):
    """Agrupa eventos con peso idéntico por ocurrencia y semilla."""
    m = data.m.sum()
    ret_total = data.ret.sum()
    pin = data.in_mismo.sum()/m if m else np.nan
    pout = data.out_mismo.sum()/m if m else np.nan
    q = (data.m*data.q).sum()/m if m else np.nan
    pu = (data.m*data.P_original).sum()/m if m else np.nan
    pret = data.ret_mismo.sum()/ret_total if ret_total else np.nan
    share = 1-data.R.mean()
    delta = share*(pin-pout) if m else 0
    select = share*((pin-q)-(pout-pu)) if m else 0
    retained = data.R.mean()*pret if ret_total else 0
    assert abs(data.delta.mean()-delta) < 1e-12
    assert abs(data.S.mean()-select) < 1e-12
    assert abs(data.P.mean()-(retained+(share*pin if m else 0))) < 1e-12
    assert abs(data.P_original.mean()-(retained+(share*pout if m else 0))) < 1e-12
    return dict(eventos_sustitucion=int(m), fraccion_sustituida=share, P_in=pin, P_out=pout,
                P_ret=pret, q_ponderada=q, P_original_ponderada=pu, delta=delta, S=select,
                delta_observado=data.delta.mean(), S_observada=data.S.mean(),
                S_aj=data.S_aj.mean(), S_rango=data.S_rango.mean())


def main():
    from rutas import exigir_trabajo_local
    exigir_trabajo_local()
    check = read(REVISION/'checkpoints'/'F3.json')
    cfg = read(CFG/'F3_definitiva_v1.json')
    assert check['validado'] and digest(OUT/'F3_proyecciones.npz') == cfg['sha256_proyecciones']
    assert digest(CFG/'CP0_v1.json') == cfg['CP0_sha256']
    info = pd.read_csv(EXP/'common.csv', usecols=['id','group','document','synset'])
    group_results, details, decomposition = [], [], []
    with np.load(OUT/'F3_proyecciones.npz', allow_pickle=False) as fitted, np.load(
            OUT/'F1_proyecciones.npz', allow_pickle=False) as historical:
        for g, ids in info.groupby('group').indices.items():
            p = f'{g}::'
            rows = info.iloc[ids].reset_index(drop=True)
            np.testing.assert_array_equal(rows.id, fitted[p+'ids'])
            np.testing.assert_array_equal(ids, fitted[p+'indices'])
            labels = rows.synset.to_numpy()
            assert rows.synset.value_counts().min() >= 10
            x = historical[p+'X_unit']
            original = vecinos(x)
            n = len(x)
            proportional = {math.ceil(phi*(n-1)) for phi in (.10,.25,.40)}
            scales = sorted(set(range(1,11)) | proportional)
            selected_k = cfg['umap_vecinos'][g]
            for condition, method, seeds, projections in (
                ('principal','PCA',[None],[fitted[p+'PCA']]),
                ('principal','Isomap',[None],[fitted[p+'Isomap']]),
                ('principal','UMAP',cfg['semillas'],fitted[f'{p}UMAP::{selected_k}']),
                ('C0','PCA',[None],[historical[p+'PCA']]),
                ('C0','Isomap',[None],[historical[p+'Isomap']]),
                ('C0','UMAP',[42],[historical[p+'UMAP']]),
                ('m12_20_semillas','UMAP',cfg['semillas'],fitted[p+'UMAP::12']),
            ):
                batches = {}
                for seed, z in zip(seeds, projections):
                    rank = vecinos(z)
                    for k in scales:
                        trust = k < n/2 and (k!=10 or k in proportional)
                        metrics, occurrence = group_metrics(x,z,original,rank,labels,k,trust)
                        keys = dict(condicion=condition, grupo=g, metodo=method, semilla=seed, k=k,
                                    primaria=k<=9, continuidad=k in (3,5,10), proporcional=k in proportional)
                        group_results.append({**keys, **metrics})
                        frame = rows[['id','document']].assign(**keys,**occurrence)
                        details.append(frame)
                        batches.setdefault(k,[]).append(frame)
                for k, frames in batches.items():
                    decomposition.append(dict(condicion=condition,grupo=g,metodo=method,k=k,
                                              **pooled(pd.concat(frames,ignore_index=True))))
            print(g, flush=True)
    detail = pd.concat(details,ignore_index=True)
    detail.to_csv(OUT/'F45_ocurrencias.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    additive = ['R','RNX','P_original','P','P_original_aj','P_aj','delta','delta_aj','S','S_aj','S_rango','S_rango_aj']
    mean_items = detail.groupby(['condicion','metodo','grupo','id','document','k'])[additive].mean().reset_index()
    mean_items.to_csv(OUT/'F45_ocurrencias_promedio.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    scores = ['RNX','delta_aj','S_aj']
    auc_input = mean_items[mean_items.k <= 9].copy()
    auc_input[scores] = auc_input[scores].div(auc_input.k,axis=0) / sum(1/k for k in range(1,10))
    auc = auc_input.groupby(['condicion','metodo','grupo','id','document'])[scores].sum().reset_index()
    auc.to_csv(OUT/'F45_AUC_ocurrencias.csv',index=False)
    auc.groupby(['condicion','metodo','grupo'])[scores].mean().to_csv(OUT/'F45_AUC_grupos.csv')
    auc.groupby(['condicion','metodo','grupo'])[scores].mean().groupby(level=[0,1]).mean().to_csv(
        OUT/'F45_AUC_macro.csv')
    uniform = mean_items[mean_items.k <= 9].groupby(
        ['condicion','metodo','grupo'])[scores].mean().groupby(level=[0,1]).mean()
    uniform.to_csv(OUT/'F45_AUC_macro_uniforme.csv')
    metrics = pd.DataFrame(group_results)
    metrics.to_csv(OUT/'F4_grupo_semilla.csv',index=False)
    keys = ['condicion','metodo','k']
    f4_cols = ['R','RNX','P_original','P','P_original_aj','P_aj','delta','delta_aj','T','pi','P_max','S','S_aj']
    group_means = metrics.groupby(keys+['grupo'])[f4_cols].mean()
    group_means.to_csv(OUT/'F4_grupos.csv')
    macro = group_means.groupby(level=keys).mean()
    macro['n_grupos'] = group_means.groupby(level=keys).size()
    macro['n_grupos_T'] = group_means.groupby(level=keys)['T'].count()
    macro.to_csv(OUT/'F4_macro.csv')
    ns = info.groupby('group').size()
    proportional_frames = []
    for phi in (.10,.25,.40):
        table = group_means.reset_index()
        include = table.k == np.ceil(phi*(table.grupo.map(ns)-1))
        proportional_frames.append(table[include].assign(phi=phi))
    proportional_table = pd.concat(proportional_frames, ignore_index=True)
    proportional_table.to_csv(OUT/'F4_proporcional_grupos.csv',index=False)
    proportional_table.groupby(['condicion','metodo','phi'])[f4_cols].mean().to_csv(
        OUT/'F4_proporcional_macro.csv')
    variances = []
    for key, frame in metrics.query('condicion == "principal" and metodo == "UMAP"').groupby('k'):
        for col in f4_cols:
            valid=frame.dropna(subset=[col])
            if valid.empty:
                continue
            group_var=valid.groupby('grupo')[col]
            between=group_var.mean().var(ddof=0)
            within=group_var.var(ddof=0).mean()
            weights=valid['grupo'].map(1/group_var.size())
            center=np.average(valid[col],weights=weights)
            total=np.average((valid[col]-center)**2,weights=weights)
            assert abs(total-between-within)<1e-12
            variances.append(dict(k=key,metrica=col,n_grupos=group_var.ngroups,
                                  var_grupo=between,var_semilla_dentro_grupo=within,var_total=total))
    pd.DataFrame(variances).to_csv(OUT/'F4_varianzas_umap.csv',index=False)
    parts = pd.DataFrame(decomposition)
    parts.to_csv(OUT/'F5_descomposicion_grupo.csv',index=False)
    macros = []
    for key, frame in parts.groupby(keys):
        active=frame[frame.eventos_sustitucion>0]
        weights=len(active)/len(frame)
        volume=active.fraccion_sustituida.to_numpy()
        quality=(active.P_in-active.P_out).to_numpy()
        product=volume.mean()*quality.mean() if len(active) else 0
        cov=np.mean((volume-volume.mean())*(quality-quality.mean())) if len(active) else 0
        delta=frame.delta.mean()
        assert abs(delta-weights*(product+cov))<1e-12
        macros.append(dict(zip(keys,key),n_grupos=len(frame),n_activos=len(active),delta=delta,
                           producto_activos=product,cov_activos=cov,peso_activos=weights,
                           S=frame.S.mean(),S_aj=frame.S_aj.mean(),S_rango=frame.S_rango.mean()))
    pd.DataFrame(macros).to_csv(OUT/'F5_descomposicion_macro.csv',index=False)
    verify_hashes()


if __name__ == '__main__':
    main()

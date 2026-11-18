"""F3, ajuste y selección geométrica sobre entradas sin etiquetas."""
from collections import Counter
import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from umap import UMAP

from f0 import digest, verify_hashes
from geometria import concordancia, normalizar, vecinos
from rutas import EXP, HISTORICAL, REVISION

OUT = REVISION / 'resultados'
CFG = REVISION / 'config'


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def geometry_input():
    common = pd.read_csv(EXP / 'common.csv', usecols=['id', 'group'])
    rows = pd.read_csv(EXP / 'rows_bert.csv', usecols=['row', 'id'])
    assert common.id.is_unique and common.id.equals(rows.id)
    assert np.array_equal(rows.row, np.arange(601))
    return common, np.load(EXP / 'X_common.npy', mmap_mode='r', allow_pickle=False)


def scales(n, grid):
    return {**{f'abs{k}': k for k in grid['absolutos']},
            **{f'prop{p:.2f}': math.ceil(p*(n-1)) for p in grid['proporcionales']}}


def fit_umap(x, params, k, seed):
    model = UMAP(**{**params, 'n_neighbors': k, 'random_state': seed})
    assert model.n_neighbors == k and model.random_state == seed
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        z = model.fit_transform(x)
    assert model._n_neighbors == k and z.shape == (len(x), 2) and np.isfinite(z).all()
    return z, Counter(str(w.message) for w in notices)


def control():
    assert all(read(REVISION / 'checkpoints' / 'F1.json').values())
    with np.load(OUT / 'F1_proyecciones.npz', allow_pickle=False) as c0, np.load(
            HISTORICAL / 'proyecciones_20_grupos.npz', allow_pickle=False) as history:
        assert len(c0.files) == 100 and set(c0.files) == set(history.files)
        for key in c0.files:
            np.testing.assert_array_equal(c0[key], history[key])
    print('C0: reproducción F1 independiente verificada contra el archivo histórico, 100 matrices', flush=True)


def main():
    from rutas import exigir_trabajo_local
    exigir_trabajo_local()
    verify_hashes()
    control()
    cp = read(CFG / 'CP0_v1.json')
    grid = read(CFG / 'F3_rejilla_v1.json')
    assert cp['estado'] == 'aprobada_y_congelada'
    assert digest(OUT / 'F2_reglas_grupo.csv') == cp['fuente_sha256']
    c0_cfg = read(CFG / 'C0.json')
    params = {**c0_cfg['UMAP'], 'compatibility_layout': True}
    params['precomputed_knn'] = tuple(params['precomputed_knn'])
    common, raw = geometry_input()
    assert set(common.group) == set(cp['grupos'])
    seeds = grid['semillas']
    assert seeds == list(range(42, 62))
    cache = REVISION / 'checkpoints' / 'F3_grupos'
    cache.mkdir(exist_ok=True)
    signature = tuple(digest(p) for p in (CFG / 'CP0_v1.json', CFG / 'F3_rejilla_v1.json',
                                          CFG / 'C0.json', EXP / 'X_common.npy'))
    stable_groups, pairs, assignments, changes, validation = [], [], [], [], []
    all_z = {}
    with np.load(OUT / 'F1_proyecciones.npz', allow_pickle=False) as c0, np.load(
            OUT / 'F2_matrices.npz', allow_pickle=False) as diag:
        for group, idx in common.groupby('group').indices.items():
            prefix = f'{group}::'
            assert np.array_equal(idx, c0[prefix+'indices'])
            assert np.array_equal(idx, diag[prefix+'indices'])
            x = normalizar(raw[idx])
            np.testing.assert_array_equal(x, c0[prefix+'X_unit'])
            n = len(idx)
            settings = cp['grupos'][group]
            assert settings['n'] == n
            m = settings['n_neighbors']
            iso = diag[f'{prefix}{m}::ISOMAP']
            pca = c0[prefix+'PCA']
            ids = common.iloc[idx].id.to_numpy(dtype=str)
            all_z.update({prefix+'indices': idx, prefix+'ids': ids, prefix+'PCA': pca,
                          prefix+'Isomap': iso})
            grid_g = scales(n, grid)
            unique = sorted(set(grid_g.values()))
            path = cache / f'{group}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    assert tuple(saved['signature']) == signature
                    np.testing.assert_array_equal(saved['ids'], ids)
                    fitted = {k: saved[str(k)] for k in unique}
                    notices = json.loads(saved['warnings'].item())
            else:
                fitted, notices = {}, {}
                for k in unique:
                    batch = [fit_umap(x, params, k, seed) for seed in seeds]
                    fitted[k] = np.stack([z for z, _ in batch])
                    notices[str(k)] = dict(sum((w for _, w in batch), Counter()))
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, signature=signature, ids=ids, warnings=json.dumps(notices),
                                    **{str(k): z for k, z in fitted.items()})
                temp.replace(path)
            np.testing.assert_array_equal(fitted[12][0], c0[prefix+'UMAP'])
            for k, batch in fitted.items():
                assert batch.shape == (20, n, 2) and np.isfinite(batch).all()
                repeat, repeat_notices = fit_umap(x, params, k, seeds[-1])
                np.testing.assert_array_equal(repeat, batch[-1])
                validation.append((group, k, seeds[-1], True, json.dumps(notices[str(k)]),
                                   json.dumps(dict(repeat_notices))))
                ranked = [vecinos(z) for z in batch]
                values = []
                for i, j in itertools.combinations(range(len(seeds)), 2):
                    value = concordancia(ranked[i], ranked[j], grid['k_concordancia'])
                    pairs.append((group, k, seeds[i], seeds[j], value))
                    values.append(value)
                stable_groups.append((group, n, k, np.mean(values), np.std(values), min(values), max(values)))
                all_z[f'{prefix}UMAP::{k}'] = batch
            for rule, k in grid_g.items():
                assignments.append((group, rule, k))
            for name, z in [('PCA', pca), ('Isomap', iso)]:
                old = c0[prefix+name]
                changes.append((group, name, m if name=='Isomap' else None, None,
                                concordancia(vecinos(z), vecinos(old), 5), procrustes(old,z)[2]))
            print(f'{group}: ISOMAP m={m}, UMAP {unique}, {len(unique)*20} ajustes certificados', flush=True)
    groups = pd.DataFrame(stable_groups, columns=['grupo','n','n_neighbors','rnx_media','rnx_sd','rnx_min','rnx_max'])
    rules = pd.DataFrame(assignments, columns=['grupo','regla','n_neighbors']).merge(groups)
    summary = rules.groupby('regla', sort=False).agg(
        vecinos_mediana=('n_neighbors','median'), vecinos_media=('n_neighbors','mean'),
        estabilidad=('rnx_media','median')).reset_index()
    summary = summary.sort_values(['vecinos_mediana','vecinos_media'], kind='stable')
    for threshold in grid['sensibilidad']:
        summary[f'pasa_{threshold:.2f}'] = summary.estabilidad >= threshold
    group_config = None
    eligible = summary[summary.estabilidad >= grid['umbral']]
    if len(eligible):
        selected = eligible.iloc[0].regla
        selected_rows = rules[rules.regla == selected]
        group_config = selected_rows.set_index('grupo').n_neighbors.to_dict()
        for group, k in group_config.items():
            prefix = f'{group}::'
            ref = vecinos(all_z[prefix+'UMAP::12'][0])
            for seed, z in zip(seeds, all_z[f'{prefix}UMAP::{k}']):
                old = all_z[prefix+'UMAP::12'][0]
                changes.append((group,'UMAP',k,seed,concordancia(vecinos(z),ref,5),procrustes(old,z)[2]))
    for filename, frame in {
        'estabilidad_grupo': groups, 'reglas': rules, 'seleccion': summary,
        'pares_semillas': pd.DataFrame(pairs,columns=['grupo','n_neighbors','semilla_a','semilla_b','rnx5']),
        'validacion_semillas': pd.DataFrame(validation,columns=['grupo','n_neighbors','semilla_repetida','exacta','avisos','avisos_repeticion']),
        'comparacion_C0': pd.DataFrame(changes,columns=['grupo','metodo','vecinos','semilla','rnx5_vs_C0','procrustes_vs_C0'])
    }.items():
        frame.to_csv(OUT / f'F3_{filename}.csv', index=False)
    np.savez_compressed(OUT / 'F3_proyecciones.npz', **all_z)
    verify_hashes()
    print(summary.to_string(index=False), flush=True)
    if group_config is None:
        raise SystemExit('Ninguna escala UMAP supera el umbral; F4 y F5 quedan bloqueadas.')
    dump(CFG / 'F3_definitiva_v1.json', dict(version=1, CP0_sha256=signature[0],
         umap_regla=selected, umap_vecinos=group_config, umap_parametros=params, semillas=seeds,
         isomap_fuente='F2_matrices.npz, escalamiento clásico verificado contra sklearn',
         umap_version=read(CFG/'entorno.json')['packages']['umap-learn'],
         sha256_proyecciones=digest(OUT/'F3_proyecciones.npz')))
    dump(REVISION/'checkpoints'/'F3.json',dict(validado=True,control_C0=True,regla_umap=selected,
         ajustes=len(validation)*20,repeticiones_exactas=len(validation),entradas=['id','group','embeddings']))


if __name__ == '__main__':
    main()

"""Reajuste independiente y comparación con las referencias congeladas."""
import inspect
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, trustworthiness
from umap import UMAP

from geometria import coincidencia, normalizar, ranking_distancias, vecinos
from rutas import EXP, HISTORICAL, REVISION

KS = (3, 5, 10)
METHODS = ('PCA', 'Isomap', 'UMAP')
OUT = REVISION / 'resultados'
UMAP_PARAMS = dict(n_neighbors=12, n_components=2, metric='euclidean', min_dist=0.1,
                   init='spectral', random_state=42, n_jobs=1)


def compare(table, historical, keys, values, name, tolerance):
    table = table.merge(historical[keys + values], on=keys, suffixes=('', '_historico'),
                        how='outer', validate='one_to_one', indicator=True)
    assert (table.pop('_merge') == 'both').all(), name
    errors = {}
    for value in values:
        table[f'dif_{value}'] = table[value] - table[f'{value}_historico']
        errors[value] = float(table[f'dif_{value}'].abs().max())
    table.to_csv(OUT / f'F1_{name}.csv', index=False)
    print(name, len(table), errors, flush=True)
    return all(error <= tolerance for error in errors.values())


def main():
    from rutas import exigir_trabajo_local
    exigir_trabajo_local()
    raw = np.load(EXP / 'X_common.npy', mmap_mode='r', allow_pickle=False)
    common = pd.read_csv(EXP / 'common.csv')
    supports_layout = 'compatibility_layout' in inspect.signature(UMAP).parameters
    estimators = dict(PCA=PCA(n_components=2, svd_solver='full'),
                      Isomap=Isomap(n_components=2, n_neighbors=12, metric='euclidean',
                                    path_method='auto'),
                      UMAP=UMAP(**UMAP_PARAMS))
    config = {name: estimator.get_params() for name, estimator in estimators.items()}
    config['compatibility_layout_disponible'] = supports_layout
    (REVISION / 'config' / 'C0.json').write_text(json.dumps(config, indent=2) + '\n')
    r, t, p, d, coords = [], [], [], [], []
    arrays = {}
    null = np.zeros((10_000, 4, 3))
    rng = np.random.default_rng(42)
    with np.load(HISTORICAL / 'proyecciones_20_grupos.npz', allow_pickle=False) as history:
        for group, indices in common.groupby('group', sort=True).indices.items():
            x = normalizar(raw[indices])
            n = len(x)
            prefix = f'{group}::'
            assert np.array_equal(indices, history[prefix + 'indices'])
            assert np.array_equal(x, history[prefix + 'X_unit'])
            original = vecinos(x)
            cosine = ranking_distancias(1 - np.clip(x @ x.T, -1, 1))
            assert np.array_equal(original, cosine)
            labels = common.iloc[indices].synset.to_numpy()
            group_data = common.iloc[indices].reset_index(drop=True)
            strata = {}
            for positions in group_data.groupby(['document', 'sentence'], sort=False).indices.values():
                assert len(set(labels[positions])) == 1
                strata.setdefault(len(positions), []).append(positions)
            lab, _ = pd.factorize(labels, sort=True)
            permuted = np.broadcast_to(lab, (len(null), n)).copy()
            for blocks in strata.values():
                block_labels = lab[[pos[0] for pos in blocks]]
                if len(blocks) == 1:
                    continue
                assignments = block_labels[np.argsort(rng.random((len(null), len(blocks))), axis=1)]
                for j, pos in enumerate(blocks):
                    permuted[:, pos] = assignments[:, j, None]
            arrays[prefix + 'indices'] = indices
            arrays[prefix + 'X_unit'] = x
            fits = {'Original': x}
            for method, estimator in estimators.items():
                fits[method] = estimator.fit_transform(x)
            if supports_layout and group in ('guide#v', 'table#n'):
                for mode in (True, False):
                    alternative = UMAP(**UMAP_PARAMS, compatibility_layout=mode).fit_transform(x)
                    print(f'{group} layout={mode}: diferencia máxima contra implícito '
                          f'{np.max(np.abs(alternative - fits["UMAP"]))}', flush=True)
            baseline = {}
            for position, (method, z) in enumerate(fits.items()):
                rank = original if method == 'Original' else vecinos(z)
                if method != 'Original':
                    arrays[prefix + method] = z
                    reference = history[prefix + method]
                    signs = np.where((reference * z).sum(axis=0) < 0, -1, 1)
                    error = float(np.max(np.abs(z - reference)))
                    aligned_error = float(np.max(np.abs(z * signs - reference)))
                    coords.append((group, method, error, aligned_error, *signs,
                                   np.array_equal(rank[:, :10], vecinos(reference)[:, :10])))
                for j, k in enumerate(KS):
                    top = rank[:, :k]
                    purity = (labels[top] == labels[:, None]).mean(axis=1).mean()
                    p.append((group, n, method, k, purity))
                    null[:, position, j] += (permuted[:, top] == permuted[:, :, None]).mean(axis=(1, 2)) / 20
                    if method == 'Original':
                        baseline[k] = purity
                        continue
                    r.append((group, n, method, k, coincidencia(original, rank, k)))
                    d.append((group, n, method, k, purity, baseline[k], purity - baseline[k]))
                    if k in (3, 5):
                        t.append((group, n, method, k, trustworthiness(x, z, n_neighbors=k)))
    np.savez_compressed(OUT / 'F1_proyecciones.npz', **arrays)
    coordinates = pd.DataFrame(coords, columns=['grupo', 'metodo', 'max_abs', 'max_abs_signos',
                                                'signo_1', 'signo_2', 'top10_igual'])
    coordinates.to_csv(OUT / 'F1_coordenadas.csv', index=False)
    common_keys = ['grupo', 'n', 'metodo', 'k']
    tables = {
        'R': (r, common_keys, ['R_k'], 'conservacion_Rk_20_grupos.csv'),
        'T': (t, common_keys, ['trustworthiness'], 'trustworthiness_20_grupos.csv'),
        'P': (p, ['grupo', 'n', 'representacion', 'k'], ['pureza'], 'pureza_vecindarios_20_grupos.csv'),
        'delta': (d, common_keys, ['pureza', 'pureza_original', 'delta_pureza'],
                  'delta_pureza_20_grupos.csv'),
    }
    gates, macro = {}, []
    for name, (records, keys, values, filename) in tables.items():
        table = pd.DataFrame(records, columns=keys + values)
        historical = pd.read_csv(HISTORICAL / filename).rename(
            columns={'Grupo': 'grupo', 'Método': 'metodo', 't': 'k'})
        gates[name] = compare(table, historical, keys, values, name, 1e-9)
        group_keys = keys[2:]
        macro.append(table.groupby(group_keys)[values].mean().reset_index().assign(metrica=name))
    pd.concat(macro, ignore_index=True).to_csv(OUT / 'F1_macro.csv', index=False)
    with np.load(HISTORICAL / 'distribuciones_permutacion.npz', allow_pickle=False) as hist_null:
        key = next(key for key in hist_null.files if hist_null[key].shape == null.shape)
        old_null = hist_null[key]
        null_error = float(np.max(np.abs(null - old_null)))
    np.savez_compressed(OUT / 'F1_nulos.npz', nulos=null, referencia=old_null)
    print('Error máximo de distribuciones nulas:', null_error)
    gates['nulos'] = null_error <= 1e-9
    linear = coordinates.metodo != 'UMAP'
    gates['coordenadas_lineales'] = bool((coordinates.loc[linear, 'max_abs_signos'] <= 1e-9).all())
    gates['coordenadas_umap'] = bool((coordinates.loc[~linear, 'max_abs'] <= 1e-6).all())
    print(coordinates.groupby('metodo')[['max_abs', 'max_abs_signos']].max())
    (REVISION / 'checkpoints' / 'F1.json').write_text(json.dumps(gates, indent=2) + '\n')
    if not all(gates.values()):
        raise SystemExit('F1 no superó todos los criterios')


if __name__ == '__main__':
    main()

"""Diagnóstico ISOMAP con grafos explícitos y selección sin etiquetas."""
from collections import Counter
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from geometria import concordancia, normalizar, ranking_distancias, vecinos
from rutas import EXP, REVISION

OUT = REVISION / 'resultados'
CONFIG = json.loads((REVISION / 'config' / 'F2.json').read_text())


def graph(rank, m):
    edge = np.zeros((len(rank), len(rank)), dtype=bool)
    np.put_along_axis(edge, rank[:, :m], True, axis=1)
    return edge | edge.T


def paths(distances, edge):
    rows, cols = np.nonzero(edge)
    sparse = csr_matrix((distances[rows, cols], (rows, cols)), shape=edge.shape)
    return shortest_path(sparse, directed=False, return_predecessors=True)


def hops_from_predecessors(predecessors):
    n = len(predecessors)
    hops = np.zeros_like(predecessors)
    for start in range(n):
        for end in range(n):
            if end == start or predecessors[start, end] < 0:
                continue
            cursor = end
            while cursor != start:
                hops[start, end] += 1
                assert hops[start, end] < n
                cursor = predecessors[start, cursor]
    return hops


def classical(distances):
    squared = distances ** 2
    means = squared.mean(axis=0)
    kernel = -0.5 * (squared - means[:, None] - means[None, :] + means.mean())
    values, vectors = eigh(kernel)
    return values, vectors[:, -2:] * np.sqrt(np.maximum(values[-2:], 0))


def grid(n, minimum):
    proposed = {f'min+{delta}': minimum + delta for delta in CONFIG['incrementos_minimo']}
    proposed.update({f'abs{m}': m for m in CONFIG['absolutos']})
    proposed.update({f'prop{p:.2f}': math.ceil(p * (n - 1)) for p in CONFIG['proporcionales']})
    return proposed, {rule: min(n - 1, max(minimum, m)) for rule, m in proposed.items()}


def probe(x, m, group, n_components):
    model = Isomap(n_neighbors=m, n_components=2, metric='euclidean', path_method='auto')
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        try:
            model.fit(x)
        except (ValueError, RuntimeError) as exc:
            return dict(grupo=group, m=m, componentes=n_components, error=str(exc))
    return dict(grupo=group, m=m, componentes=n_components,
                finita=bool(np.isfinite(model.dist_matrix_).all()),
                avisos=sorted({str(w.message) for w in seen}))


def group_run(group, x, indices, output, records, rule_rows, stability, table_members):
    n = len(x)
    pairs = np.triu_indices(n, 1)
    euclidean = cdist(x, x)
    distances = euclidean[pairs]
    rank = ranking_distancias(euclidean)
    pca = PCA(n_components=2, svd_solver='full').fit_transform(x)
    pca_rank = vecinos(pca)
    prefix = f'{group}::'
    output.update({prefix + 'indices': indices, prefix + 'euclidiana': euclidean, prefix + 'PCA': pca})
    tol = CONFIG['tolerancia_distancias']
    k = CONFIG['k_concordancia']
    previous, previous_edge = np.full_like(euclidean, np.inf), np.zeros_like(euclidean, dtype=bool)
    rows, projected = {}, {}
    minimum = None
    for m in range(1, n):
        edge = graph(rank, m)
        degrees = edge.sum(axis=1)
        components, labels = connected_components(csr_matrix(edge), directed=False)
        assert np.all(~previous_edge | edge)
        previous_edge = edge
        geo, pred = paths(euclidean, edge)
        assert np.allclose(geo, geo.T, atol=tol, rtol=0)
        assert np.all(geo >= euclidean - tol) and np.all(geo <= previous + tol)
        assert np.allclose(geo[edge], euclidean[edge], atol=tol, rtol=0)
        previous = geo
        sizes = np.bincount(labels)
        row = dict(grupo=group, n=n, m=m, componentes=components,
                   tamanos_componentes=json.dumps(sorted(sizes.tolist(), reverse=True)),
                   aristas=int(degrees.sum() // 2), densidad=degrees.mean() / (n - 1),
                   rho_m=m / (n - 1), grado_min=degrees.min(), grado_max=degrees.max(),
                   grado_medio=degrees.mean(), distribucion_grados=json.dumps(dict(sorted(Counter(map(int, degrees)).items()))),
                   cv_distancias=distances.std() / distances.mean(),
                   contraste_relativo=(distances.max() - distances.min()) / distances.min())
        assert row['rho_m'] - tol <= row['densidad'] <= min(1, 2 * row['rho_m']) + tol
        if group == 'table#n' and m <= 12:
            for i, label in enumerate(labels):
                table_members.append((m, int(indices[i]), int(label), int(sizes[label])))
        if components == 1:
            if minimum is None:
                minimum = m
            values, y = classical(geo)
            hops = hops_from_predecessors(pred)[pairs]
            adjacent = edge[pairs]
            assert np.array_equal(hops == 1, adjacent)
            g = geo[pairs]
            stretch = (g / distances)[~adjacent]
            disparity = procrustes(pca, y)[2]
            projected[m] = vecinos(y)
            row.update(
                saltos_media=hops.mean(), saltos_max=hops.max(),
                distribucion_saltos=json.dumps(dict(sorted(Counter(map(int, hops)).items()))),
                estiramiento_media=stretch.mean() if stretch.size else np.nan,
                estiramiento_mediana=np.median(stretch) if stretch.size else np.nan,
                estiramiento_p90=np.quantile(stretch, .9) if stretch.size else np.nan,
                iguales_euclidiana=np.isclose(g, distances, rtol=0, atol=tol).mean(),
                pearson=pearsonr(g, distances).statistic, spearman=spearmanr(g, distances).statistic,
                masa_negativa=float(-values[values < -1e-12].sum() / np.abs(values).sum()),
                fraccion_dos_positivas=values[-2:].sum() / values[values > 0].sum(),
                varianza_residual=1 - pearsonr(g, cdist(y, y)[pairs]).statistic ** 2,
                reconstruction_error=np.linalg.norm(values[:-2]) / n,
                procrustes_pca=disparity, rnx_pca=concordancia(pca_rank, projected[m], k))
            output.update({f'{prefix}{m}::geodesica': geo, f'{prefix}{m}::espectro': values,
                           f'{prefix}{m}::ISOMAP': y, f'{prefix}{m}::saltos': hops,
                           f'{prefix}{m}::estiramiento': stretch})
            if m in (minimum, 12, n - 1):
                sk = Isomap(n_neighbors=m, n_components=2, metric='euclidean', path_method='auto')
                sk_y = sk.fit_transform(x)
                assert np.allclose(sk.dist_matrix_, geo, atol=tol, rtol=0)
                assert abs(sk.reconstruction_error() - row['reconstruction_error']) < tol
                assert procrustes(sk_y, y)[2] < CONFIG['tolerancia_procrustes_completo']
            if m == n - 1:
                assert np.max(np.abs(geo - euclidean)) < tol
                assert disparity < CONFIG['tolerancia_procrustes_completo']
        rows[m] = row
        records.append(row)
    proposed, actual = grid(n, minimum)
    used = sorted(set(actual.values()))
    between = {}
    for a, b in zip(used, used[1:]):
        between[a, b] = concordancia(projected[a], projected[b], k)
        stability.append((group, a, b, between[a, b]))
    for rule, m in actual.items():
        pos = used.index(m)
        below = between[used[pos - 1], m] if pos else np.nan
        above = between[m, used[pos + 1]] if pos < len(used) - 1 else np.nan
        rule_rows.append(dict(rows[m], regla=rule, minimo=minimum, m_propuesto=proposed[rule],
                              equivalentes=';'.join(key for key, val in actual.items() if val == m),
                              rnx_inferior=below, rnx_superior=above,
                              estabilidad=min(v for v in (below, above) if np.isfinite(v))))


def summarize(rules):
    summaries = []
    for name, data in rules.groupby('regla', sort=False):
        lower = data.rnx_inferior.fillna(data.rnx_superior).median()
        upper = data.rnx_superior.fillna(data.rnx_inferior).median()
        score = data.estabilidad.median()
        for density in CONFIG['densidades']:
            count = int((data.densidad <= density).sum())
            for threshold in CONFIG['estabilidades']:
                summaries.append(dict(regla=name, umbral_densidad=density, umbral_estabilidad=threshold,
                                      grupos_en_limite=count, densidad_mediana=data.densidad.median(),
                                      estabilidad=score, estabilidad_inferior=lower, estabilidad_superior=upper,
                                      cumple_densidad=count >= CONFIG['grupos_minimos'],
                                      cumple_estabilidad=score >= threshold,
                                      admisible=count >= CONFIG['grupos_minimos'] and score >= threshold,
                                      admisible_inferior=count >= CONFIG['grupos_minimos'] and lower >= threshold,
                                      admisible_superior=count >= CONFIG['grupos_minimos'] and upper >= threshold,
                                      procrustes_mediana=data.procrustes_pca.median(),
                                      rnx_pca_mediana=data.rnx_pca.median()))
    return pd.DataFrame(summaries)


def main():
    from rutas import exigir_trabajo_local
    exigir_trabajo_local()
    gates = json.loads((REVISION / 'checkpoints' / 'F1.json').read_text())
    if not all(gates.values()):
        raise SystemExit('F1 pendiente')
    raw = np.load(EXP / 'X_common.npy', mmap_mode='r', allow_pickle=False)
    metadata = pd.read_csv(EXP / 'common.csv', usecols=['group'])
    records, rules, stability, members, array_data = [], [], [], [], {}
    for group, indices in metadata.groupby('group', sort=True).indices.items():
        group_run(group, normalizar(raw[indices]), indices, array_data, records, rules, stability, members)
        print('Grupo verificado:', group, flush=True)
    diagnostics = pd.DataFrame(records)
    diagnostics.to_csv(OUT / 'F2_diagnostico.csv', index=False)
    rules = pd.DataFrame(rules)
    rules.to_csv(OUT / 'F2_reglas_grupo.csv', index=False)
    summary = summarize(rules)
    summary.to_csv(OUT / 'F2_admisibilidad.csv', index=False)
    pd.DataFrame(stability, columns=['grupo', 'm_a', 'm_b', 'rnx5']).to_csv(OUT / 'F2_estabilidad.csv', index=False)
    pd.DataFrame(members, columns=['m', 'fila_original', 'componente', 'tamano']).to_csv(
        OUT / 'F2_table_componentes.csv', index=False)
    np.savez_compressed(OUT / 'F2_matrices.npz', **array_data)
    synthetic = np.array([[0., 0.], [0.1, 0.], [0., 0.2], [5., 1.], [5.1, 1.], [5., 1.2]])
    table = normalizar(raw[array_data['table#n::indices']])
    probes = []
    for name, x, m in [('sintetico', synthetic, 2), ('table#n', table, 11)]:
        edges = graph(vecinos(x), m)
        probes.append(probe(x, m, name, connected_components(csr_matrix(edges), directed=False)[0]))
    (OUT / 'F2_reparacion_sklearn.json').write_text(json.dumps(probes, indent=2) + '\n')
    reference = summary.query('umbral_densidad == 0.30 and umbral_estabilidad == 0.50')
    print(reference.to_string(index=False))


if __name__ == '__main__':
    main()

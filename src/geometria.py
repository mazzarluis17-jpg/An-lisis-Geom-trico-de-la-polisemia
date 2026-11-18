"""Transformaciones y vecindades geométricas sin etiquetas."""
import numpy as np
from scipy.spatial.distance import cdist


def normalizar(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ranking_distancias(distances):
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind='stable')[:, :-1]


def vecinos(points):
    return ranking_distancias(cdist(points, points))


def coincidencia(a, b, k):
    hits = (a[:, :k, None] == b[:, None, :k]).any(axis=2).sum()
    return hits / (len(a) * k)


def concordancia(a, b, k):
    chance = k / (len(a) - 1)
    return (coincidencia(a, b, k) - chance) / (1 - chance)

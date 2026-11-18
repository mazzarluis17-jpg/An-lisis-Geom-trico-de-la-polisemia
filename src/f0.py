"""Controles reutilizados por el flujo local sin el inventario histórico privado."""
import hashlib

from rutas import REVISION


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_hashes():
    """Verifica el manifiesto local que suministra el investigador."""
    manifest = REVISION/'logs'/'SHA256SUMS_originales.txt'
    if not manifest.is_file():
        raise FileNotFoundError('Falta el manifiesto de insumos locales: logs/SHA256SUMS_originales.txt')
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert digest(REVISION/relative) == expected, relative

"""Rutas del paquete público y de la ejecución local opcional."""
import os
from pathlib import Path

PUBLICO = Path(__file__).resolve().parents[1]
REVISION = Path(os.environ.get('POLISEMIA_TRABAJO', PUBLICO)).resolve()
EXP = Path(os.environ.get('POLISEMIA_EXP', PUBLICO / 'datos' / 'local')).resolve()
HISTORICAL = EXP / 'resultados_fase2'
DATOS_C0 = PUBLICO / 'datos' / 'c0'


def exigir_trabajo_local():
    if ('POLISEMIA_TRABAJO' not in os.environ or 'POLISEMIA_EXP' not in os.environ
            or REVISION == PUBLICO or REVISION.is_relative_to(PUBLICO)):
        raise SystemExit('Flujo local: indica POLISEMIA_EXP y una carpeta de trabajo externa con POLISEMIA_TRABAJO. Consulta datos/README.md.')

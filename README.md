# Análisis geométrico de la polisemia mediante embeddings contextuales

**Autor:** Luis Alfredo Ramírez Maza

[Artículo científico definitivo (PDF)](articulo/articulo_polisemia_definitivo.pdf) · [Protocolo metodológico](docs/Protocolo_metodologico_v1.1.md) · [Trazabilidad de las cifras](resultados/F9_trazabilidad.csv)

## Pregunta y diseño

¿Las relaciones de vecindad que conservan o introducen los métodos de reducción dimensional corresponden a ocurrencias que comparten el mismo synset de WordNet?

El piloto estudia 601 ocurrencias de SemCor, en 20 grupos lema–categoría gramatical y 222 documentos. Compara proyecciones bidimensionales de embeddings BERT normalizados con PCA, ISOMAP de mínimo conectante individual y UMAP prop0.15 con veinte semillas. Las escalas se eligieron mediante criterios geométricos. Tras el ajuste, las etiquetas de WordNet evaluaron la correspondencia semántica. Las métricas distinguen conservación ajustada por azar, cambio de pureza ajustada y selectividad de las sustituciones.

## Resultados principales

UMAP obtuvo una AUC macro de conservación de **0.535582**, frente a **0.400374** de PCA y **0.397632** de ISOMAP. Las bandas simultáneas de conservación para las comparaciones frente a UMAP excluyen cero en la configuración principal. La ventaja descriptiva depende de la escala y se invierte en k=9 para prop0.15. Las bandas principales de los contrastes semánticos incluyen cero. El artículo demuestra que el cambio de pureza es el producto de la fracción de vecinos sustituidos y la diferencia de pureza entre quienes entran y salen.

Estas bandas son aproximaciones condicionales a la geometría fija y a la reponderación de los documentos observados. La muestra es determinista, informada por sentidos, y los veinte grupos son fijos. El estudio se limita al piloto.

## Qué se puede reproducir

- **Verificación del artículo.** Las macros se regeneran desde 233 recetas trazadas y el PDF se recompila desde las fuentes y figuras incluidas.
- **Reproducción estadística parcial.** Los agregados sin localizadores permiten recalcular los nueve contrastes y sus bandas simultáneas con pesos multinomiales y exponenciales en la configuración principal. Se conservan el orden documental, las semillas y la calibración. Coinciden los hashes de los pesos y la comprobación de valores usa tolerancia absoluta de 1e-12. El error máximo observado fue menor que 1e-16.
- **Sensibilidades y ajustes.** Las bandas de abs5 y C0 y los diagnósticos geométricos se distribuyen como resultados verificables. Reconstruir reductores y puntuaciones por ocurrencia requiere insumos originales externos. [Datos, agregación y flujo local](datos/README.md)

## Instalación y ejecución

Desde la raíz, con Python 3.11 y pip:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH=src python -B -m unittest discover -s tests -v
python -B src/reproducir_bandas.py
python -B src/f9_trazabilidad.py
cmp build/trazabilidad/F8_macros.tex articulo/tables/F8_macros.tex
cmp build/trazabilidad/F8_grupos.tex articulo/tables/F8_grupos.tex
cmp build/trazabilidad/F9_trazabilidad.csv resultados/F9_trazabilidad.csv
sha256sum -c --quiet SHA256SUMS.txt
```

El archivo **build/bandas_reproducidas.csv** y la carpeta **build/trazabilidad/** contienen las salidas nuevas.

Para recompilar el artículo con XeLaTeX y Biber, instala biblatex-apa, Babel en español, TeX Gyre Pagella, TeX Gyre Pagella Math, Lato y Latin Modern Mono. Ejecuta desde la raíz:

```bash
bash articulo/compilar.sh
```

La recompilación se guarda en **articulo/build/**. [Validación pública](docs/Validacion_publica.md)

## Estructura

| Directorio | Contenido |
| --- | --- |
| articulo/ | PDF definitivo, fuentes XeLaTeX y figuras vectoriales |
| config/ | Parámetros congelados y entorno |
| datos/ | Agregados públicos y las dos tablas históricas citadas |
| resultados/ | Tablas de respaldo, sensibilidades y trazabilidad |
| src/ y tests/ | Código de análisis, comandos públicos y pruebas |
| docs/ | Protocolo, informes metodológicos y validación |

Las atribuciones de SemCor, WordNet y BERT y los derechos sobre código y artículo se documentan por separado. [Avisos de terceros](THIRD_PARTY_NOTICES.md)

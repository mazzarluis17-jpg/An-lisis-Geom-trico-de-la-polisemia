# Análisis geométrico de la polisemia mediante embeddings contextuales

**Investigación en Matemáticas Aplicadas, Ciencia de Datos y Procesamiento del Lenguaje Natural**

**Autor:** Luis Alfredo Ramírez Maza  
**Institución:** Universidad Nacional Autónoma de México — FES Acatlán

## Descripción

Esta investigación analiza la relación entre la geometría de los embeddings contextuales y los sentidos léxicos anotados en WordNet.

Se estudia cómo tres métodos de reducción dimensional —PCA, ISOMAP y UMAP— modifican las relaciones de vecindad de representaciones BERT al proyectarlas de 768 a 2 dimensiones.

La pregunta central es:

> ¿Las relaciones geométricas que conservan o introducen PCA, ISOMAP y UMAP corresponden a ocurrencias que comparten el mismo sentido de WordNet?

## Datos y metodología

El estudio utiliza una muestra fija del corpus SemCor:

| Característica | Valor |
|---|---|
| Ocurrencias | 601 |
| Grupos lema–POS | 20 |
| Lemas distintos | 19 |
| Synsets WordNet | 50 |
| Documentos | 222 |
| Modelo | BERT Base Uncased |
| Dimensión original | 768 |
| Dimensión reducida | 2 |

Los embeddings se normalizan mediante la norma L2 y cada grupo lema–POS se analiza independientemente.

Los métodos evaluados son:

- **PCA:** reducción lineal mediante componentes principales.
- **ISOMAP:** reducción basada en distancias de caminos mínimos sobre grafos de proximidad.
- **UMAP:** reducción mediante relaciones locales ponderadas.

La selección de parámetros utiliza criterios geométricos y de estabilidad, sin emplear las etiquetas semánticas de WordNet.

## Métricas de evaluación

Se consideran tres dimensiones principales del análisis:

**Conservación geométrica:** proporción de vecinos originales retenidos después de la reducción, corregida por coincidencia esperada bajo azar mediante \(R_{NX}\).

**Correspondencia semántica:** pureza de vecindarios según synsets de WordNet y cambio de pureza ajustada respecto al espacio original.

**Selectividad semántica:** diferencia entre los cambios observados y una referencia de sustitución uniforme condicionada al número de vecinos reemplazados.

Una identidad central del estudio permite descomponer el cambio de pureza:

\[
\Delta P=(1-R)(P_{\mathrm{in}}-P_{\mathrm{out}})
\]

El análisis considera escalas de vecindad \(k=1,\ldots,9\) y utiliza agregación macro y bandas simultáneas bootstrap condicionadas a los documentos observados y a las representaciones ajustadas.

## Resultados principales

En la configuración principal, UMAP presenta la mayor conservación geométrica para la AUC que concede más peso a las vecindades pequeñas.

| Método | AUC \(R_{NX}\) | AUC \(\Delta P_{\mathrm{aj}}\) |
|---|---:|---:|
| PCA | 0.400374 | -0.043308 |
| ISOMAP | 0.397632 | -0.042994 |
| UMAP | **0.535582** | **-0.006783** |

Las bandas simultáneas respaldan los dos contrastes de conservación frente a UMAP. Las diferencias de correspondencia semántica entre los métodos no quedan respaldadas por las bandas principales.

Los resultados dependen del tamaño de vecindad y de las configuraciones seleccionadas. La investigación es exploratoria y no establece superioridad universal de ningún método.

## Artículo científico

El documento completo, con fundamentos matemáticos, metodología, resultados, discusión y referencias, está disponible en:

**[Consultar artículo científico (PDF)](articulo/articulo_polisemia_definitivo.pdf)**

## Código y reproducibilidad

El repositorio contiene el código seleccionado para documentar los procedimientos computacionales y los resultados del artículo.

Las instrucciones de instalación, ejecución y los requisitos de datos se especifican en los archivos correspondientes del repositorio.

Los resultados estadísticos son condicionales a las representaciones y vecindarios observados. La selección determinista de los grupos limita la generalización a otros lemas y corpus.

## Referencias

Las referencias científicas completas se encuentran en el artículo, incluyendo los trabajos fundamentales sobre BERT, PCA, ISOMAP, UMAP, SemCor y WordNet.

---

**Alcance:** investigación exploratoria sobre una muestra fija de 601 ocurrencias. No pretende demostrar la existencia de una variedad semántica subyacente ni establecer superioridad universal entre métodos de reducción dimensional.

# F3 · Configuración y estabilidad de las proyecciones

CP0 quedó aprobado y congelado en `config/CP0_v1.json`, con todos los parámetros efectivos de ISOMAP, PCA y la referencia a C0. SHA-256 de ese archivo: `a032529abf9a23c33ae0fe8dcf83c0410cf80910b259be95ca3701576e058ac9`. La regla `min+0` tiene densidad mediana 0.150094, 19 grupos dentro de δ≤0.30 y estabilidad mediana 0.567600. Cumple τ=0.50 y queda por debajo de τ=0.60. Se conservaron los 20 grupos, con m=12 en `table#n`. La comparación principal usa PCA, ISOMAP y UMAP.

## Control C0 y ejecución

Antes de evaluar la rejilla se verificó la reproducción independiente C0 de F1 contra las 100 matrices históricas. Esto incluye las 60 proyecciones, los 20 índices y las 20 matrices de entrada L2. Como control de F3 se volvió a ajustar UMAP con 12 vecinos y semilla 42 en cada grupo, ahora con `compatibility_layout=True` explícito. Las 20 matrices reprodujeron exactamente las coordenadas de C0. Se conservaron PCA de C0 e ISOMAP de F2, donde el escalamiento clásico del mínimo conectante ya había sido verificado contra scikit-learn.

F3 cargó únicamente la matriz cruda y las columnas `id`, `row` y `group`. Comprobó orden, IDs, normalización contra C0 y coincidencia de índices con F2. UMAP recibió únicamente X, con métrica euclidiana, dos componentes, min_dist 0.1, inicialización espectral, n_jobs=1 y umap-learn 0.6.0. Se ejecutaron 2 200 ajustes distintos y se repitió semilla 61 para las 110 combinaciones grupo–escala efectiva, con coincidencia exacta en las 110. Los lotes usaron las semillas 42–61. La captura de avisos durante el ajuste devolvió cero avisos en esos lotes. Los mensajes de inicialización de TensorFlow en el log corresponden a la importación de UMAP.

## Regla UMAP seleccionada

Cada valor efectivo se ajustó una vez por semilla dentro de cada grupo. La estabilidad por grupo y escala es la media de las 190 parejas de semillas en R_NX a k=5. La estabilidad de una regla es la mediana de esas 20 medias. Antes de ajustar se fijó el orden por mediana del número efectivo de vecinos entre grupos, luego por media y luego por orden de declaración para desempates. El protocolo no ordena explícitamente reglas mixtas absolutas y proporcionales. En este piloto, ordenar por media primero habría elegido `abs5` (media 5.00 frente a 5.05). La decisión de la comparación principal usa el orden registrado.

| Regla | Mediana de vecinos | Media de vecinos | Estabilidad mediana | Pasa τ=0.40/0.50/0.60 |
| --- | ---: | ---: | ---: | --- |
| prop0.15 | 4 | 5.05 | 0.756689 | Sí / Sí / Sí |
| abs5 | 5 | 5.00 | 0.766031 | Sí / Sí / Sí |
| prop0.25 | 6 | 7.60 | 0.725943 | Sí / Sí / Sí |
| abs8 | 8 | 8.00 | 0.719561 | Sí / Sí / Sí |
| prop0.40 | 10 | 12.10 | 0.665925 | Sí / Sí / Sí |
| abs12 | 12 | 12.00 | 0.631279 | Sí / Sí / Sí |

La regla seleccionada es **prop0.15**, `ceil(0.15*(n−1))`. Los valores 3, 4, 6 y 8 se aplican a 1, 12, 3 y 4 grupos, respectivamente. Las configuraciones quedan registradas en `config/F3_definitiva_v1.json` y `resultados/F3_reglas.csv`. El criterio escogió la menor escala admisible. En estos datos la estabilidad entre semillas no aumenta sistemáticamente con la escala.

## Diferencias geométricas respecto de C0

| Método | Mediana R_NX a k=5 | Mediana de disparidad de Procrustes |
| --- | ---: | ---: |
| PCA | 1.000000 | 0 a precisión numérica |
| ISOMAP, mínimo frente a m=12 | 0.425000 | 0.379571 |
| UMAP, prop0.15 frente a C0 con la misma semilla 42 | 0.646462 | 0.336106 |
| UMAP, prop0.15 promediado por grupo sobre semillas, frente a C0 semilla 42 | 0.652740 | 0.314850 |

La comparación UMAP a igual semilla controla esa fuente de variación. El promedio sobre las 20 semillas describe a la vez el cambio de escala y la variación entre semillas respecto de una referencia fija. Los 19 grupos con cambio de m contribuyen a la comparación ISOMAP, junto con `table#n`. Los CSV de F3 contienen recuentos, vecinos efectivos, las 20 900 parejas de semillas, controles de repetición y comparaciones C0 por grupo. Los 433 SHA-256 originales coincidieron antes y después del ajuste.

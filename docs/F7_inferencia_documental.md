# F7 · Inferencia documental y robustez

**Resultado principal.** Las dos bandas de AUC de R_NX que comparan UMAP con PCA e ISOMAP excluyen cero a favor de UMAP. Mantienen dirección y exclusión de cero en las seis combinaciones de configuración y mecanismo. Ninguna otra banda principal excluye cero. Algunas comparaciones de pureza ajustada sí excluyen cero en sensibilidades de configuración. La selectividad ajustada presenta cambios de signo al variar la escala.

Las bandas son aproximaciones condicionales a la geometría fija y a la reponderación de los documentos observados. Se calibraron por separado para cada familia configuración–mecanismo. La cobertura aproximada de cada familia corresponde a sus nueve contrastes.

## Fuentes y controles

Se leyeron el protocolo, los informes F3 y F4–F5 y la auditoría previa. La configuración principal usa PCA, ISOMAP mínimo conectante por grupo y UMAP prop0.15 con semillas 42–61. Los veinte grupos, incluido table#n, tienen el mismo peso. Se verificaron IDs y orden contra las 601 filas de metadatos, las proyecciones F3 y las matrices históricas. Los 433 hashes originales y el manifiesto F3–F5 coincidieron.

La fuente principal fue `F45_AUC_ocurrencias.csv`. Las puntuaciones principales se verificaron desde matrices y proyecciones con la función de F4–F5. Se reconstruyeron las seis escalas UMAP y C0 en lectura. Para abs5 se obtuvieron 0.542318105960, +0.013851983689 y 0.199098374541 en AUC macro de R_NX, ΔP ajustada y S ajustada. El cotejo con los valores publicados a seis decimales usó tolerancia absoluta 5×10⁻⁷. La tolerancia de cálculo para AUC, identidad de contrastes y métricas C0 fue 10⁻¹² sin tolerancia relativa. C0 se verificó frente a las proyecciones históricas, las 180 filas de R_k, las 120 de trustworthiness, las 240 de pureza original y reducida y las 180 diferencias de pureza a las escalas históricas pertinentes.

## Estimando, réplicas y calibración

Se restaron los AUC de cada par de métodos dentro de la misma ocurrencia. UMAP se promedió sobre sus 20 semillas por ocurrencia para la configuración principal y abs5. C0 conserva la semilla histórica 42. Para cada réplica se recalculó la media ponderada por grupo y se promediaron las 20 medias con peso 1/20. Las proyecciones, vecindades, puntuaciones y constantes de corrección se mantuvieron fijas.

Se usó un solo vector de pesos por réplica para los nueve contrastes y los veinte grupos. Las matrices de pesos se conservaron en el archivo local original y cada una se reutilizó en las tres configuraciones. El repositorio público regenera los pesos desde las semillas. Los documentos se ordenan lexicográficamente por identificador de origen y se identifican como D001 a D222 en la entrega. La multinomial generó 20 001 extracciones y aceptó 20 000. En el intento 17 932 faltó military#a y se descartó la réplica completa. La exponencial generó 20 000 extracciones válidas. La incidencia incluye 138 documentos multigrupo y entre 11 y 43 documentos por grupo.

El SE se calculó con desviación estándar muestral (ddof=1). El cuantil fue el estadístico de orden 19 000 de los 20 000 máximos absolutos de desviaciones centradas y estandarizadas. Todos los SE fueron finitos y superiores a la tolerancia de degeneración registrada. Se usó el mismo SE completo al partir los máximos en dos mitades disjuntas, para aislar la variación aproximada del cuantil. En cada mitad se tomó el estadístico de orden 9 500. Esta comparación es un diagnóstico Monte Carlo aproximado.

| Configuración | Pesos | Cuantil completo | Mitad 1 | Mitad 2 | Diferencia absoluta |
| --- | --- | ---: | ---: | ---: | ---: |
| Principal | Multinomial | 2.713665 | 2.729788 | 2.698519 | 0.031269 |
| Principal | Exponencial | 2.694873 | 2.698816 | 2.688126 | 0.010690 |
| abs5 | Multinomial | 2.704454 | 2.693500 | 2.712650 | 0.019150 |
| abs5 | Exponencial | 2.690816 | 2.696936 | 2.687575 | 0.009360 |
| C0 | Multinomial | 2.710745 | 2.726568 | 2.692573 | 0.033995 |
| C0 | Exponencial | 2.704888 | 2.709173 | 2.703437 | 0.005736 |

*Calibración de bandas simultáneas aproximadas, condicional a la geometría fija y a la reponderación de los documentos observados.*

## Nueve contrastes principales

*Bandas simultáneas aproximadas al 95%, condicionales a la geometría fija y a la reponderación multinomial de los documentos observados.*

| AUC | Contraste | Estimador | SE | Banda al 95% | Cero |
| --- | --- | ---: | ---: | --- | --- |
| R_NX | PCA−ISOMAP | +0.002742 | 0.014016 | [−0.035291, +0.040776] | Incluye |
| R_NX | PCA−UMAP | −0.135208 | 0.012629 | [−0.169480, −0.100937] | Excluye |
| R_NX | ISOMAP−UMAP | −0.137951 | 0.010835 | [−0.167354, −0.108547] | Excluye |
| ΔP ajustada | PCA−ISOMAP | −0.000313 | 0.020801 | [−0.056761, +0.056135] | Incluye |
| ΔP ajustada | PCA−UMAP | −0.036525 | 0.016642 | [−0.081687, +0.008637] | Incluye |
| ΔP ajustada | ISOMAP−UMAP | −0.036212 | 0.016506 | [−0.081004, +0.008581] | Incluye |
| S ajustada | PCA−ISOMAP | −0.016956 | 0.020005 | [−0.071242, +0.037330] | Incluye |
| S ajustada | PCA−UMAP | +0.014005 | 0.015680 | [−0.028545, +0.056555] | Incluye |
| S ajustada | ISOMAP−UMAP | +0.030960 | 0.015613 | [−0.011408, +0.073329] | Incluye |

## Robustez

- **Mecanismo de pesos.** Para la configuración principal, la reponderación exponencial redujo el SE entre 0.000546 y 0.001764 y la amplitud total entre 0.003348 y 0.010290. Ningún contraste principal cambió su inclusión de cero. En C0, S ajustada PCA−ISOMAP incluyó cero con pesos multinomiales y lo excluyó con pesos exponenciales, con límite superior −0.001253 en la última variante. Los cambios completos de SE y amplitud están en `F7_comparacion_pesos.csv`.
- **Configuración.** Ambas bandas principales de conservación que excluyen cero mantuvieron dirección y exclusión en todas las sensibilidades. No se activó la definición de fragilidad inferencial para ninguna de las dos. En abs5 las dos bandas de ΔP ajustada frente a UMAP excluyeron cero con ambos mecanismos. En C0 la banda de ΔP ajustada de PCA−UMAP excluyó cero con ambos mecanismos y la de R_NX PCA−ISOMAP también lo excluyó. Esas exclusiones corresponden a las familias de sensibilidad. C0 cambia también ISOMAP y el promedio de semillas UMAP, de modo que su comparación con la principal refleja cambios conjuntos.
- **Fragilidad direccional.** El estimador de S ajustada PCA−UMAP cambió de signo con abs5, con C0 y con cuatro escalas adicionales de la rejilla. El estimador de S ajustada ISOMAP−UMAP cambió de signo en cuatro escalas de la tabla descriptiva, manteniendo el signo en abs5 y C0. Las columnas de fragilidad inferencial, direccional y documental se registran por separado. Las seis escalas se muestran en `F7_escalas_umap.csv`, con nueve estimadores por regla y sin calibración de bandas adicional.
- **Omisión documental.** Se recalcularon 222 omisiones de un documento con puntuaciones fijas. Los dos contrastes de R_NX frente a UMAP conservaron dirección y magnitud superior al margen original en las 222 omisiones. AUC R_NX PCA−ISOMAP invirtió el signo en dos omisiones y ΔP ajustada PCA−ISOMAP en 72. En los otros siete contrastes fuera de las dos exclusiones principales, todas las omisiones quedaron por debajo del margen original. Esos siete estimadores completos ya partían por debajo de ese margen. La marca documental aplicada literalmente refleja esa situación. El umbral usa el margen de la banda completa y no constituye una banda jackknife recalibrada.
- **Influencia.** D121 produjo el mayor cambio absoluto en R_NX PCA−UMAP (−0.005135), en ΔP ajustada PCA−ISOMAP (−0.005812) y en S ajustada ISOMAP−UMAP (+0.004262). Tiene 11 ocurrencias en seis grupos. D138 produjo el mayor cambio en S ajustada PCA−ISOMAP (+0.005798). `F7_influencia_documental.csv` contiene las 1 998 filas documento–contraste y la figura muestra el documento de mayor cambio por contraste. Se publican identificadores internos.

El análisis cuantifica cuánto cambian las puntuaciones de la muestra bajo reponderación de sus documentos. El diseño estratificado por sentido y la geometría fijada limitan el alcance a esta comparación. Las bandas no incorporan el reajuste de los reductores ni la reconstrucción de vecindades. No se calcularon ajustes escalonados de p-valores ni pruebas por grupos.


## Reproducción pública

Los comandos del [README](../README.md) reconstruyen las dos familias de la configuración principal. Las cuatro familias de sensibilidad de configuración se consultan en [los resultados guardados](../resultados/F7_bandas_sensibilidad.csv).

# Protocolo metodológico definitivo · v1.1

**Análisis geométrico de la polisemia mediante embeddings contextuales y técnicas de reducción dimensional**

Versión metodológica v1.1 aplicada al artículo definitivo. Consolidación de las decisiones aprobadas y la ejecución verificada.

## 1. Pregunta, alcance y unidades

¿Las relaciones geométricas que conservan o introducen PCA, ISOMAP y UMAP al reducir embeddings de BERT a dos dimensiones corresponden a ocurrencias con el mismo synset de WordNet? El estudio es exploratorio y se refiere a 601 ocurrencias, 20 grupos lema–POS fijos, 19 lemas, 50 synsets y 222 documentos. Los grupos se seleccionaron de forma determinista e informada por sentidos. La elección posterior de escalas y los ajustes utilizan únicamente geometría. Se reserva WordNet para evaluar correspondencia semántica en las proyecciones ya fijadas. La comparación no permite extrapolar a otros lemas, otras extracciones o todo SemCor.

## 2. Datos, control histórico y configuración

Se utiliza `X_common.npy` cruda, 601×768 float32. Por grupo se verifica la correspondencia por ID y se deriva una copia L2 en float64. La distancia de vecindad es euclidiana, con diagonal excluida y orden estable por fila local. En el espacio unitario, el ranking coincide con el del coseno, según la identidad de distancias y los controles de F1. La reducción es independiente por grupo y de dos dimensiones.

- **PCA** usa SVD completa, con parámetros en [CP0 v1](../config/CP0_v1.json).
- **ISOMAP** usa `min+0`, el mínimo conectante individual. Se conservan todos los grupos, incluido table#n con m=12. La matriz geodésica usa peso euclidiano y caminos cortos sobre la unión simétrica. El escalamiento clásico se verificó con scikit-learn en los valores controlados.
- **UMAP** usa `prop0.15` y 20 semillas 42–61, con distancia euclidiana, min_dist=0.1, inicialización espectral, un hilo y `compatibility_layout=True` de umap-learn 0.6.0. La regla efectiva es techo de 0.15(n−1). Los parámetros completos y sus valores por grupo están en [F3 definitiva](../config/F3_definitiva_v1.json).
- **C0** conserva PCA, ISOMAP m=12 y UMAP con 12 vecinos y semilla 42 por la ruta heredada. Las 60 proyecciones y las métricas históricas principales se reprodujeron. La condición de 12 vecinos con 20 semillas permite además separar descriptivamente el cambio de escala de UMAP del uso de una sola semilla. Sus métricas se conservan en F4–F5. [Configuración C0](../config/C0.json)

## 3. Diagnóstico y selección de ISOMAP

El grafo k-NN dirigido con m vecinos se simetriza por unión. La densidad efectiva es δ_E=2E/[n(n−1)]. La fracción m/(n−1) mide el parámetro dirigido y es una cota inferior de δ_E. La selección se evaluó con las 11 reglas de v1.0, piso de conectividad y deduplicación por valor efectivo dentro del grupo. Las geodésicas son distancias de caminos mínimos en el grafo ponderado. En una arista alcanzan la distancia euclidiana. Un par sin arista también puede alcanzar igualdad por colinealidad. En el grafo completo el escalamiento clásico coincide geométricamente con PCA.

La densidad de referencia 0.30 exige 18 de 20 grupos dentro del límite. La estabilidad usa R_NX a k=5 entre proyecciones de valores de m distintos y consecutivos. Para una regla interior se toma la menor concordancia de las dos adyacencias por grupo. En un extremo se usa la adyacencia disponible. La mediana entre grupos debe alcanzar τ_s=0.50. Se evaluaron las atribuciones inferior y superior como sensibilidad, además de densidades 0.25/0.30/0.35 y estabilidad 0.40/0.50/0.60.

Se aprobó `min+0` como la regla admisible de menor densidad mediana (0.150094), con 19 grupos dentro del límite y estabilidad 0.567600. A densidad 0.30 y estabilidad 0.60 ninguna regla pasa. Esta sensibilidad permanece explícita. Las geodésicas y las proyecciones PCA/ISOMAP muestran diferencias. La proximidad de AUC de conservación no establece equivalencia ni identifica una variedad. [Diagnóstico F2](../resultados/F2_diagnostico.csv) · [CP0 aprobado](../config/CP0_v1.json)

## 4. Escala y semillas de UMAP

La rejilla propia combina 5, 8 y 12 vecinos con techo de φ(n−1), para φ=0.15, 0.25, 0.40. Los valores efectivos repetidos por grupo se ajustaron una sola vez por semilla. La estabilidad se obtiene promediando R_NX a k=5 en los 190 pares de semillas de cada grupo y tomando la mediana entre los 20 grupos. Se ordenaron las reglas por mediana de vecinos efectivos, luego media y orden de declaración. La menor escala que supera τ_s=0.50 es prop0.15, con estabilidad 0.756689. El orden por media primero habría elegido abs5. Se documenta esa elección metodológica y se conserva abs5 como sensibilidad. La pureza, las etiquetas, R_k frente al original y trustworthiness no se usaron para seleccionar configuraciones. [Rejilla previa](../config/F3_rejilla_v1.json) · [Informe F3](F3_configuracion_y_estabilidad.md)

## 5. Métricas y escalas

R_k es la fracción media de vecinos originales retenidos entre los k del espacio reducido. R_NX=(R_k−c)/(1−c), con c=k/(n−1), tiene esperanza cero bajo la referencia de vecinos uniformes independientes del conjunto original. Trustworthiness se calcula en el dominio k<n/2, es secundaria y registra los grupos elegibles.

P es la fracción media de vecinos que comparten synset. La referencia analítica sin autoelección es π=Σ_s n_s(n_s−1)/[n(n−1)]. P_max es el promedio por ocurrencia de min(k,n_s−1)/k. P_aj=(P−π)/(P_max−π) y ΔP_aj=ΔP/(P_max−π). La referencia analítica difiere de la permutación histórica por bloques de oración, que conserva restricciones de intercambiabilidad por tamaño de bloque. Se reproducen las medias nulas históricas contra su propio NPZ.

La ventana primaria es k=1…9. La AUC es la media discreta con pesos normalizados 1/k. Se conservan k=3, 5 y 10 para continuidad y escalas proporcionales 0.10, 0.25 y 0.40 como complemento. La AUC uniforme, los puntos k=5 y k=9 y la curva sobre las seis reglas UMAP son descriptivos. [Resultados F4](../resultados/F4_macro.csv) · [Sensibilidad por escala](../resultados/preF8_RNX_sensibilidad.csv)

## 6. Sustituciones y selectividad

Por ocurrencia, $m_i=k-|N_i^U\cap N_i^Z|$ es el número de sustituciones. Para cada grupo y escala se agrupan eventos sobre ocurrencias y semillas antes de calcular P_in, P_out y otras proporciones. ΔP=(1−R)(P_in−P_out). S resta de ΔP la esperanza de perder y ganar coincidencias con retiro uniforme y adición uniforme desde el complemento, condicionado a $m_i$ y a la vecindad original. Para cada ocurrencia se resta $(m_i/k)(q_i-p_i^U)$, donde $p_i^U$ es su pureza original y $q_i$ la proporción del mismo sentido entre los n−1−k candidatos del complemento. S_aj usa el mismo divisor P_max−π. Retirar los vecinos de mayor rango se conserva como sensibilidad descriptiva.

Con $v_g=1-R_g$ y $d_g=P_{in,g}-P_{out,g}$, el macro de ΔP vale $p_{\mathrm{act}}[\bar v_{\mathrm{act}}\bar d_{\mathrm{act}}+\operatorname{Cov}_{\mathrm{act}}(v,d)]$, con covarianza poblacional y $p_{\mathrm{act}}$ igual a la fracción de grupos activos. Si todos están inactivos, el macro vale cero. Si un grupo carece de eventos, sus proporciones condicionadas quedan indefinidas, con ΔP=S=0. La selectividad por sustitución σ=S/(1−R) se comunica descriptivamente. Su media entre grupos activos da el mismo peso a esos grupos. La media agrupada por eventos pondera por conteo de sustituciones. [Descomposición](../resultados/F5_descomposicion_grupo.csv) · [Selectividad por sustitución](../resultados/preF8_selectividad_resumen.csv)

## 7. Reponderación documental

El estimador de cada contraste es el macro de 20 medias de grupo de diferencias pareadas por ocurrencia. UMAP se promedia por ocurrencia sobre las 20 semillas en la comparación principal. Las métricas de la familia son AUC R_NX, AUC ΔP_aj y AUC S_aj. Los pares son PCA−ISOMAP, PCA−UMAP e ISOMAP−UMAP. Las constantes de corrección y las puntuaciones permanecen fijas.

El mecanismo principal usa 20 000 réplicas aceptadas de conteos multinomiales de 222 documentos con semilla 42. Cada extracción comparte un vector de pesos entre métodos, grupos y contrastes. Se descarta la extracción completa si algún grupo tiene denominador cero. Las medias de razón se agregan después dando peso 1/20 a cada grupo. Hay 11–43 documentos por grupo, 138 documentos multigrupo y 161 de 190 pares de grupos con documentos compartidos. El mínimo de 11 corresponde a military#a. Una extracción fue descartada.

Para cada familia se calcula el SE muestral de cada contraste y el máximo de las desviaciones absolutas centradas en el estimador observado y divididas por ese SE. La banda bilateral al 95% usa el estadístico de orden 19 000 de los 20 000 máximos. Se detiene la calibración ante un SE degenerado. Dos mitades disjuntas, con el SE completo, diagnostican variación Monte Carlo del cuantil. Se comunica si la banda simultánea incluye o excluye cero. La inferencia principal no usa Wilcoxon, inversión de signos por grupo ni p-valores escalonados Westfall–Young. Los contrastes históricos quedan como antecedentes de su configuración.

Se repite la calibración con pesos Exp(1), semilla 43, y se cruzan las dos leyes con la condición principal, abs5 y C0. Cada combinación calibra su propio máximo sobre nueve contrastes y comparte los vectores de pesos con las demás configuraciones bajo la misma ley. No se atribuye control simultáneo a la unión de las seis familias. Se distingue fragilidad inferencial, direccional y documental. Las omisiones individuales de documentos evalúan dirección y margen original con geometría fija. Las bandas son aproximaciones condicionales a la reponderación de los documentos observados. Excluyen la variabilidad de reajustar reductores, reconstruir rankings o rediseñar la muestra. [Configuración F7](../config/F7_v1.json) · [Informe F7](F7_inferencia_documental.md)

## 8. Análisis complementarios y límites

Se ejecutaron y validaron trustworthiness y los componentes empíricos de varianza entre grupos y semillas dentro de grupo para UMAP. Se ejecutaron como sensibilidades parciales las rejillas, configuraciones y mecanismos de pesos descritos arriba, una revisión inicial de anisotropía mediante normas/cosenos y distancias, una comprobación de layout UMAP alternativo en dos grupos y la reproducción de permutaciones por bloques para C0. Se realizó una omisión focal y descriptiva de military#a en el cierre pre-F8. Quedaron sin ejecutar el jackknife completo por grupo, centrado global + L2, centrado por grupo + L2, eliminación de siete componentes + L2, nuevas permutaciones de la configuración principal y la malla semántica factorial ISOMAP×UMAP. Los datos y el método ejecutado permiten responder la pregunta del piloto dentro del alcance indicado. Las pruebas faltantes limitan hipótesis adicionales sobre anisotropía o composición del conjunto de grupos. [Inventario detallado F6](Inventario_F6.md)

## 9. Jerarquía y presentación

La configuración congelada, las tres AUC de la ventana primaria, sus nueve bandas y la descomposición fundamentan la comparación principal. C0 da continuidad histórica. Abs5 y pesos exponenciales son sensibilidades de F7 con calibración propia. Las nuevas comprobaciones por escala, σ y la omisión focal son descriptivas y carecen de bandas adicionales. Incluir cero deja abiertas diferencias dentro de los límites de la banda. La equivalencia requeriría un margen científicamente definido y un diseño para evaluarla. Cada cifra debe rastrearse a los CSV, NPZ o configuraciones. El texto debe distinguir retención del original, concordancia entre proyecciones y estructura de las distancias.

## 10. Validación y punto de control

La validación cubre hashes SHA-256, correspondencia de IDs y orden, reproducciones C0, invariantes de grafos, configuración y semillas, identidades de pureza y selectividad, cobertura de réplicas, uso conjunto de pesos y estadístico de orden. La [trazabilidad numérica](../resultados/F9_trazabilidad.csv) enlaza las cifras del artículo definitivo con las fuentes que se distribuyen.

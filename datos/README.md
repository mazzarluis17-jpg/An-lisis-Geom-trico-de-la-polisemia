# Datos públicos y entrada local opcional

## Agregados de F7

"F7_agregados_principal.npz" contiene una matriz de conteos de 222 documentos × 20 grupos y un tensor con la suma de las nueve diferencias AUC pareadas por celda. Se generó a partir de la condición principal de las puntuaciones por ocurrencia congeladas, verificando su emparejamiento completo. Se conservó la acumulación en el orden de las 601 filas del análisis. Los códigos D001–D222 mantienen exactamente el orden lexicográfico de documentos que F7 usó al generar pesos. El archivo incluye esos códigos, los nombres de grupos y de contrastes. No contiene textos, embeddings, etiquetas por ocurrencia, rutas de documentos, índices de pasajes ni tabla de correspondencia.

Para cada réplica con pesos documentales w, el estimador de grupo es la suma ponderada de las diferencias dividida por la suma ponderada de los conteos. Se promedian los veinte grupos por igual. Si un grupo pierde cobertura, se rechaza la extracción conjunta. Los agregados son suficientes para esta operación. La calibración y los generadores de pesos de F7 se conservaron en el núcleo público.

Los CSV de "c0/" son las dos tablas históricas agregadas citadas en el apéndice. Los demás CSV del artículo se encuentran en "resultados/". No se distribuyen los agregados de las sensibilidades de configuración ni las puntuaciones que permitirían reconstruirlas.

## Código para el flujo local

"f1.py", "f2.py", "f3.py", "f45.py", "f7.py" y "revision_pre_f8.py" documentan el flujo con ajuste y evaluación. Sus funciones no necesitan red para el análisis si los artefactos ya están disponibles. Sus puntos de entrada requieren "POLISEMIA_EXP" apuntando a un directorio de insumos autorizados y "POLISEMIA_TRABAJO" apuntando a una carpeta de trabajo externa al repositorio. Esta separación evita escribir sobre los archivos públicos. Estos puntos de entrada no forman parte de los comandos públicos verificados.

- F1 necesita X_common.npy y common.csv, más los NPZ y CSV históricos bajo resultados_fase2/.
- F2 necesita la matriz, los grupos alineados y el control F1 aprobado.
- F3 necesita además rows_bert.csv, C0 y el diagnóstico de F2.
- F4–F5 necesitan etiquetas, documentos, proyecciones F1/F3 y el control de F3.
- El F7 completo reconstruye las sensibilidades desde esas proyecciones y requiere las tablas de F1, F3 y F4–F5. El cierre pre-F8 usa dichas salidas.
- El flujo local conserva controles de integridad y de etapas. Requiere los correspondientes manifiestos y checkpoints en la carpeta de trabajo, con rutas relativas o configurables, y las configuraciones congeladas. El entorno adicional para ajustar UMAP está en requirements-ajustes.txt.

El repositorio no contiene el insumo local ni sus mapeos. No se ha validado en esta entrega el flujo local opcional. Para reconstruir el artículo o la inferencia pública no es necesario descargar SemCor ni BERT.

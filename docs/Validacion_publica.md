# Validación del paquete académico

## Instalación y ejecución

Se instaló requirements.txt desde PyPI en un entorno virtual nuevo de Python 3.11.2. Se ejecutaron los comandos del README desde una copia temporal fuera del árbol de la investigación, sin acceso declarado a los insumos locales. Pasaron las once pruebas públicas. Se regeneraron las 233 entradas de trazabilidad y los dos archivos de macros y parámetros de grupo. Las tres salidas coincidieron byte a byte con las copias publicadas.

XeLaTeX y Biber recompilaron el documento desde esa copia. El último pase concluyó sin referencias, citas o caracteres faltantes ni desbordamientos. El texto extraído de las 17 páginas A4 coincidió exactamente con el del PDF definitivo. La reproducción de ajustes opcionales queda fuera de esta validación.

## Validación numérica de la reponderación

Las comparaciones usan tolerancia absoluta 1e-12 y tolerancia relativa cero. Cada fila resume las diferencias máximas absolutas frente a F7 sobre los nueve contrastes de la configuración principal.

| Pesos | Estimador | SE | Cuantil | Límite inferior | Límite superior |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multinomial | 8.33e-17 | 9.02e-17 | 0 | 8.33e-17 | 9.71e-17 |
| Exponencial | 8.33e-17 | 9.89e-17 | 0 | 9.02e-17 | 9.71e-17 |

Los hashes de ambos vectores de pesos completos coincidieron con F7. La multinomial aceptó 20 000 de 20 001 extracciones y reprodujo la ausencia de military#a en el mismo intento. La exponencial aceptó las 20 000 extracciones. Se verificó la identidad entre las tres comparaciones pareadas dentro de cada métrica. Los cuantiles siguen el estadístico de orden 19 000 y las familias se calibran por separado. Estas bandas son aproximaciones condicionales a la geometría fija y a la reponderación de los documentos observados.

## Privacidad e integridad

Se inspeccionaron rutas, encabezados y localizadores de los CSV, tipos y campos del NPZ, texto del PDF y patrones de credenciales en los archivos de texto. El NPZ público contiene agregados y códigos internos sin tabla de correspondencia. El manifiesto cubre solo los archivos distribuidos. En el proyecto original coincidieron los 433 hashes históricos y el manifiesto del artículo definitivo. No se inicializó Git.

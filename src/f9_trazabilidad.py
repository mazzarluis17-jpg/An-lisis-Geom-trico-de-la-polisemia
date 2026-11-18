"""Trazabilidad de las correcciones editoriales, sin ajustar ni remuestrear."""
import csv
import json

import pandas as pd

from f8_macros import (FORMULAS, HISTORICAL, OUT, REVISION, formatted, recipe_value,
                       records, render)

ARTICLE = REVISION/'articulo'
TRACE = OUT/'F9_trazabilidad.csv'
DENSIDADES = (.25,.30,.35)
ESTABILIDADES = (.40,.50,.60)
REGLAS = ('abs5','abs8','abs12','prop0.15','prop0.25','prop0.40')


def registrar(rows, name, path, selection, cols, location, dp=6, op='identity',
              value=None, formula=None, secondary=None):
    assert not any(row['macro']==name for row in rows)
    if formula is None:
        formula = FORMULAS[op]
    if value is None:
        value = recipe_value(path,selection,cols,op)
    rows.append(dict(macro=name, valor=str(value), tex=formatted(value,dp),
                     origen=str(path.relative_to(REVISION)) if path.is_relative_to(REVISION)
                     else '../'+str(path.relative_to(REVISION.parent)),
                     filas=json.dumps(selection), columnas=json.dumps(cols),
                     operacion=op, formula=formula, decimales='' if dp is None else dp,
                     uso=location, origen_secundario=secondary or ''))


def correcciones():
    rows = []
    def csv_row(name, filename, selection, col, location, dp=6):
        registrar(rows,name,OUT/filename,selection,[col],location,dp)
    for suffix,col in [('Estimate','estimador'),('Lower','inferior'),('Upper','superior')]:
        csv_row('c0Rnx'+suffix,'F7_bandas_sensibilidad.csv',
                {'configuracion':'C0','mecanismo':'multinomial','contraste':'RNX:PCA-Isomap'},
                col,'Resultados, PCA–ISOMAP')
    for method in ('PCA','Isomap'):
        csv_row('c0Auc'+method,'F45_AUC_macro.csv',{'condicion':'C0','metodo':method},
                'RNX','Resultados, PCA–ISOMAP')
    path = OUT/'F3_seleccion.csv'
    rules = pd.read_csv(path)
    registrar(rows,'umapPass060',path,{},['pasa_0.60'],'§3.4, selección UMAP',0,'sum',formula='Conteo de verdaderos en pasa_0.60.')
    csv_row('umapMinStability','F3_seleccion.csv',{'regla':rules.loc[rules.estabilidad.idxmin(),'regla']},
            'estabilidad','§3.4, selección UMAP')
    assert rules.sort_values('vecinos_media').iloc[0].regla=='abs5'
    assert rules.sort_values(['vecinos_mediana','vecinos_media']).iloc[0].regla=='prop0.15'

    scale = OUT/'preF8_RNX_sensibilidad.csv'
    for i,rule in enumerate(REGLAS,1):
        selection={'escala':'k9','regla_umap':rule}
        for key,col in [('Value','UMAP'),('PCA','UMAP_menos_PCA'),('Iso','UMAP_menos_Isomap')]:
            registrar(rows,f'nine{i}{key}',scale,selection,[col],'Tabla UMAP a k=9')
        wins=sum(recipe_value(scale,selection,[col],'identity')>0
                 for col in ('UMAP_menos_PCA','UMAP_menos_Isomap'))
        registrar(rows,f'nine{i}Class',scale,selection,['UMAP_menos_PCA','UMAP_menos_Isomap'],
                  'Tabla UMAP a k=9',None,'clasificacion',
                  value=('A ninguno','A ISOMAP','A ambos')[wins],
                  formula='Ambos si ambas diferencias son positivas. Si solo una lo es, se indica el método. Si ninguna lo es, se indica «A ninguno».')
        if wins==1:
            assert recipe_value(scale,selection,['UMAP_menos_Isomap'],'identity')>0

    cp = OUT/'F2_admisibilidad.csv'
    decisions = pd.read_csv(cp)
    for i,density in enumerate(DENSIDADES,1):
        for j,stability in enumerate(ESTABILIDADES,1):
            selection={'umbral_densidad':density,'umbral_estabilidad':stability}
            part=decisions.query('umbral_densidad==@density and umbral_estabilidad==@stability')
            assert len(part)==11
            eligible=part[part.admisible].sort_values('densidad_mediana')
            best=eligible.iloc[0].regla if not eligible.empty else 'Ninguna'
            registrar(rows,f'cpCell{i}{j}',cp,selection,['regla','admisible','densidad_mediana'],
                      '§3.4, tabla 3×3 CP0',None,'seleccion_cp0',value=best,
                      formula='Filtrar admisible=True y tomar la regla de menor densidad mediana. Si el filtro queda vacío, ninguna.')
    pca={'condicion':'principal','metodo':'PCA'}
    pair={'regla_umap':'abs5','contraste':'delta_aj:PCA-UMAP'}
    auc=recipe_value(OUT/'F45_AUC_macro.csv',pca,['delta_aj'],'identity')
    contrast=recipe_value(OUT/'F7_escalas_umap.csv',pair,['estimador'],'identity')
    registrar(rows,'absDeltaAuc',OUT/'F7_escalas_umap.csv',pair,['estimador'],
              'Resultados, pureza',6,'restar_a_PCA',value=auc-contrast,
              formula='AUC ΔP ajustada de PCA menos el estimador PCA−UMAP en la regla abs5.',
              secondary=json.dumps(dict(archivo='resultados/F45_AUC_macro.csv',
                                        filas=pca,columnas=['delta_aj'])))
    historic=HISTORICAL/'pureza_vecindarios_20_grupos.csv'
    for method in ('Isomap','UMAP'):
        registrar(rows,'historicalP'+method,historic,{'k':10,'representacion':method},['pureza'],
                  'Apéndice histórico de C0',6,'mean',
                  formula='Media aritmética de las 20 purezas por grupo del CSV original.')
    return rows


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=REVISION/"build"/"trazabilidad")
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    rows=records()+correcciones()
    for_tex = [
        {**row, 'tex': rf"\text{{{row['tex']}}}"}
        if row['operacion'] in ('clasificacion','seleccion_cp0') else row
        for row in rows
    ]
    (args.salida/'F8_macros.tex').write_text(render(for_tex))
    with (args.salida/'F9_trazabilidad.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=[*rows[0],'origen_secundario'])
        writer.writeheader()
        writer.writerows(rows)
    from f8_macros import group_table
    (args.salida/'F8_grupos.tex').write_text(group_table())
    print(f'{len(rows)} entradas trazadas')


if __name__=='__main__':
    main()

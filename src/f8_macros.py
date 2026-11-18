"""Extrae cifras para el manuscrito sin ejecutar análisis experimentales."""
import csv
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from rutas import DATOS_C0 as HISTORICAL, PUBLICO as REVISION

FORMULAS = {
    'json': 'Acceso a la ruta de claves JSON indicada.',
    'identity': 'Valor de la columna en la única fila con el selector indicado.',
    'median': 'Mediana de la columna en todas las filas seleccionadas.',
    'min': 'Mínimo de la columna en todas las filas seleccionadas.',
    'max': 'Máximo de la columna en todas las filas seleccionadas.',
    'count': 'Número de filas seleccionadas.',
    'countNonfragile': 'Conteo de filas con fragilidad_documental=False.',
    'countExcludesNegative': 'Conteo de filas con límite superior de banda < 0.',
    'countPositiveBoth': 'Conteo de filas con ambas diferencias UMAP-restante > 0.',
    'countBelowDensity': 'Conteo de filas con densidad efectiva <= 0.30.',
    'weightedContributionPercent': '100 por suma de diferencia/k en k<=3 sobre suma de diferencia/k en k=1..9.',
    'maxAbsDifference': 'Máximo de la diferencia absoluta entre las dos columnas.',
}

ARTICLE = REVISION / 'articulo'
TRACE = REVISION / 'resultados/F8_trazabilidad.csv'
MACROS = ARTICLE / 'tables/F8_macros.tex'
OUT = REVISION / 'resultados'


@lru_cache
def source(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.suffix == '.json' else pd.read_csv(path)


def csv_selection(path, filters):
    d = source(path)
    for key, value in filters.items():
        d = d[d[key] == value]
    if d.empty:
        raise ValueError((path, filters))
    return d


def recipe_value(path, selection, columns, op):
    if path.suffix == '.json':
        v = source(path)
        for key in selection:
            v = v[key]
        return v
    d = csv_selection(path, selection)
    if op == 'count':
        return len(d)
    if op == 'weightedContributionPercent':
        return 100 * (d.loc[d.k <= 3, columns[1]] / d.loc[d.k <= 3, 'k']).sum() / (d[columns[1]] / d.k).sum()
    if op == 'countNonfragile':
        return int((~d[columns[0]].astype(bool)).sum())
    if op == 'countExcludesNegative':
        return int((d[columns[0]] < 0).sum())
    if op == 'countPositiveBoth':
        return int((d[columns] > 0).all(axis=1).sum())
    if op == 'countBelowDensity':
        return int((d[columns[0]] <= .30).sum())
    series = d[columns[0]]
    if op == 'identity':
        return series.item()
    if op == 'maxAbsDifference':
        return (d[columns[0]]-d[columns[1]]).abs().max()
    return getattr(series, op)()


def formatted(value, decimals):
    if decimals is None:
        return str(value)
    decimal = Decimal(str(value))
    assert decimal.is_finite()
    rounded = decimal.quantize(Decimal(1).scaleb(-decimals),rounding=ROUND_HALF_UP)
    return format(abs(rounded) if rounded.is_zero() else rounded,f'.{decimals}f')


def records():
    rows = []

    def add(key, path, selection, columns, usage, decimals=6, op='identity'):
        assert key not in {r['macro'] for r in rows}, key
        value = recipe_value(path,selection,columns,op)
        display = formatted(value,decimals)
        rows.append(dict(macro=key,valor=str(value),tex=display,origen=str(path.relative_to(
            REVISION)) if path.is_relative_to(REVISION) else '../'+str(path.relative_to(REVISION.parent)),
            filas=json.dumps(selection,ensure_ascii=False),columnas=json.dumps(columns),
            operacion=op,formula=FORMULAS[op],decimales='' if decimals is None else decimals,uso=usage))

    def result(key,file,filters,col,where,dp=6,op='identity'):
        add(key,OUT/(file+'.csv'),filters,[col],where,dp,op)

    def config(key,file,keys,where,dp=None):
        add(key,REVISION/'config'/f'{file}.json',keys,[],where,dp,'json')

    for key,keys in [('n',[ 'shape',0]),('d',['shape',1]),('docs',['documents']),
                     ('lemmas',['lemmas']),('synsets',['synsets'])]:
        config(key,'entorno',keys,'Resumen y muestra',0)
    for key,col,op in [('groups','group','count'),('nmin','n','min'),('nmax','n','max')]:
        result(key,'F0_grupos',{},col,'Métodos y limitaciones',0,op)

    for key,field in [('sharedDocs','documentos_multigrupo'),('sharedPairs','pares_con_documentos_compartidos'),
                      ('docMin','documentos_por_grupo_min'),('docMax','documentos_por_grupo_max')]:
        add(key,OUT/'preF7_dependencia.json',[field],[],'Métodos y limitaciones',0,'json')
    for key,package in [('numpy','numpy'),('scipy','scipy'),
                        ('sklearn','scikit-learn'),('umapVersion','umap-learn')]:
        config(key,'entorno',['packages',package],'Apéndice de reproducibilidad')
    add('c0Holm',HISTORICAL/'comparaciones_pareadas.csv',{'k':10,'comparacion':'Isomap vs UMAP'},
        ['p_Holm'],'Apéndice C0',5)
    add('c0Delta',HISTORICAL/'comparaciones_pareadas.csv',{'k':10,'comparacion':'Isomap vs UMAP'},
        ['media_diferencia'],'Apéndice C0',5)
    for name,configfile,path in [('isoC','C0',['Isomap','n_neighbors']),
                                  ('umapC','C0',['UMAP','n_neighbors']),
                                  ('seedC','C0',['UMAP','random_state']),
                                  ('isoTable','CP0_v1',['grupos','table#n','n_neighbors'])]:
        config(name,configfile,path,'Métodos y apéndice C0',0)
    for key,fil,col in [('isoDensity','F2_reglas_grupo','densidad'),
                        ('isoAgreement','F2_reglas_grupo','rnx_pca'),
                        ('isoProc','F2_reglas_grupo','procrustes_pca'),
                        ('isoStretch','F2_reglas_grupo','estiramiento_mediana'),
                        ('isoSpearman','F2_reglas_grupo','spearman')]:
        result(key,fil,{'regla':'min+0'},col,'Métodos y geometría',3,'median')
    result('isoDenseC','F2_diagnostico',{'m':12},'densidad','Métodos',3,'median')
    result('isoFits','F2_reglas_grupo',{'regla':'min+0'},'densidad','Métodos',0,'countBelowDensity')
    result('isoStability','F2_admisibilidad',{'regla':'min+0','umbral_densidad':.30,'umbral_estabilidad':.50},
           'estabilidad','Métodos',6)
    for rule,key in [('prop0.15','prop'),('abs5','abs')]:
        for col,suffix,dp in [('estabilidad','Stability',6),('vecinos_mediana','Median',0),('vecinos_media','Mean',2)]:
            result(key+suffix,'F3_seleccion',{'regla':rule},col,'Métodos',dp)
    group_path=REVISION/'config/CP0_v1.json'
    for i,g in enumerate(source(group_path)['grupos'],1):
        add(f'group{i}n',group_path,['grupos',g,'n'],[],'Tabla de parámetros por grupo',0,'json')
        add(f'group{i}iso',group_path,['grupos',g,'n_neighbors'],[],'Tabla de parámetros por grupo',0,'json')
        config(f'group{i}umap','F3_definitiva_v1',['umap_vecinos',g],'Tabla de parámetros por grupo',0)

    for method in ('PCA','Isomap','UMAP'):
        for col in ('RNX','delta_aj','S_aj'):
            result(f'macro{method}{col}','F45_AUC_macro',{'condicion':'principal','metodo':method},
                   col,'Tabla AUC y resumen',6)
        for col in ('fraccion_macro','S_macro','sigma_macro_activos','sigma_agrupada_eventos','eventos','oportunidades'):
            result(f'sub{method}{col}','preF8_selectividad_resumen',
                   {'condicion':'principal','metodo':method,'k':5},col,'Tabla selectividad',
                   0 if col in ('eventos','oportunidades') else 4)
        for col in ('R','T','P'):
            result(f'c0{method}{col}','F4_macro',{'condicion':'C0','metodo':method,'k':5},col,'Tabla C0',4)
    result('c0Original','F4_macro',{'condicion':'C0','metodo':'PCA','k':5},'P_original','Apéndice C0',4)
    result('pi','F4_macro',{'condicion':'principal','metodo':'PCA','k':5},'pi','Métodos',6)
    result('absDeltaContrast','F7_escalas_umap',{'regla_umap':'abs5','contraste':'delta_aj:PCA-UMAP'},
           'estimador','Apéndice, sensibilidad de pureza')
    for i,col in enumerate(('UMAP_menos_PCA','UMAP_menos_Isomap'),1):
        add(f'local{i}',OUT/'preF8_RNX_curvas.csv',{'regla_umap':'prop0.15'},['k',col],
            'Resumen y dependencia de escala',0,'weightedContributionPercent')
    for scale in ('AUC_uniforme','k5','k9'):
        for method in ('PCA','Isomap','UMAP'):
            result(f'scale{scale}{method}','preF8_RNX_sensibilidad',
                   {'regla_umap':'prop0.15','escala':scale},method,'Tabla de escala',4)
        add(f'positive{scale}',OUT/'preF8_RNX_sensibilidad.csv',{'escala':scale},
            ['UMAP_menos_PCA','UMAP_menos_Isomap'],'Resultados por escala',0,'countPositiveBoth')

    for col,label in [('delta','delta'),('S','select')]:
        add(f'identity{label}',OUT/'F5_descomposicion_grupo.csv',{},[col,'delta_observado' if col=='delta' else 'S_observada'],
            'Resultados de identidades (registro exacto)',None,'maxAbsDifference')
    for col,key in [('delta','umapFiveDelta'),('producto_activos','umapFiveProduct'),
                    ('cov_activos','umapFiveCov')]:
        result(key,'F5_descomposicion_macro',{'condicion':'principal','metodo':'UMAP','k':5},
               col,'Resultados, descomposición macro')
    result('selectRows','preF8_selectividad_grupo',{},'sigma','Resultados selectividad',0,'count')
    result('selectActive','preF8_selectividad_grupo',{'sin_sustituciones':False},'sigma','Resultados selectividad',0,'count')
    for metric in ('RNX','delta_aj','S_aj'):
        for i,pair in enumerate(('PCA-Isomap','PCA-UMAP','Isomap-UMAP'),1):
            c=f'{metric}:{pair}'
            for column in ('estimador','inferior','superior'):
                result(f'band{metric}{i}{column}','F7_bandas_principales',{'contraste':c},
                       column,'Tabla de bandas y resultados')
            if metric=='RNX' and i>1:
                result(f'robust{i}','F7_bandas_sensibilidad',{'contraste':c},'superior',
                       'Resultados y robustez',0,'countExcludesNegative')
                result(f'omit{i}','F7_influencia_documental',{'contraste':c},'fragilidad_documental',
                       'Resultados y robustez',0,'countNonfragile')
    result('critical','F7_monte_carlo',{'configuracion':'principal','mecanismo':'multinomial'},
           'cuantil','Resultados inferenciales')
    result('accepted','F7_pesos_diagnostico',{'mecanismo':'multinomial'},'aceptadas','Inferencia y apéndice',0)
    result('rejected','F7_pesos_diagnostico',{'mecanismo':'multinomial'},'rechazadas','Inferencia y apéndice',0)
    return rows


def numeric_tex(value):
    if 'e' in value and value.replace('-', '', 1)[0].isdigit():
        coefficient, exp = value.split('e')
        return rf'{coefficient}\times 10^{{{int(exp)}}}'
    return value


def render(rows):
    lines=['% Generado desde fuentes validadas por src/f8_macros.py.',
           r'\newcommand{\Num}[1]{\ensuremath{\csname F8@#1\endcsname}}']
    lines.extend(rf"\expandafter\def\csname F8@{row['macro']}\endcsname{{{numeric_tex(row['tex'])}}}"
                 for row in rows)
    return '\n'.join(lines)+'\n'


def group_table():
    cp=source(REVISION/'config/CP0_v1.json')['grupos']
    lines=[]
    for i,group in enumerate(cp,1):
        group=group.replace('#',r'\#')
        lines.append(rf'\grupo{{{group}}} & \Num{{group{i}n}} & \Num{{group{i}iso}} & \Num{{group{i}umap}} \\')
    return '\n'.join(lines+[r'\bottomrule'])+'\n'



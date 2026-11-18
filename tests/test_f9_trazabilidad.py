import csv
from decimal import Decimal, ROUND_HALF_UP
import math
import re
import unittest

import pandas as pd

from f8_macros import HISTORICAL, OUT, group_table
from f9_trazabilidad import correcciones
from f8_macros import records
from f9_trazabilidad import ARTICLE, TRACE, REGLAS, DENSIDADES, ESTABILIDADES
from test_f8_macros import from_source


class CorreccionesF9(unittest.TestCase):
    def test_all_macros_and_sources(self):
        with TRACE.open(newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),len({r['macro'] for r in rows}))
        macros=(ARTICLE/'tables/F8_macros.tex').read_text()
        defs=dict(re.findall(r'\\csname F8@([^\n]+?)\\endcsname\{([^\n]*)\}',macros))
        self.assertEqual(set(defs),{r['macro'] for r in rows})
        sources={r['macro']:r for r in rows}
        for row in rows:
            key=row['macro']
            if row['operacion'] in ('clasificacion','seleccion_cp0'):
                self.assertEqual(defs[key],r'\text{'+row['valor']+'}')
            else:
                self.assertEqual(defs[key],row['tex'] if 'e-' not in row['tex']
                                 else row['tex'].split('e')[0]+r'\times 10^{'+str(int(row['tex'].split('e')[1]))+'}')
            if row['decimales']:
                places=int(row['decimales'])
                rounded=Decimal(row['valor']).quantize(Decimal(10)**(-places),rounding=ROUND_HALF_UP)
                self.assertEqual(format(rounded if rounded else abs(rounded),f'.{places}f'),row['tex'])
            if row['operacion'] in ('sum','clasificacion','seleccion_cp0','restar_a_PCA','mean'):
                continue
            value=from_source(row)
            if isinstance(value,str):
                self.assertEqual(value,row['valor'])
            else:
                self.assertTrue(math.isclose(value,float(row['valor']),abs_tol=1e-12,rel_tol=0))
        auc=pd.read_csv(OUT/'F45_AUC_macro.csv')
        pair=pd.read_csv(OUT/'F7_escalas_umap.csv')
        pca=auc.query('condicion=="principal" and metodo=="PCA"').delta_aj.item()
        delta=pair.query('regla_umap=="abs5" and contraste=="delta_aj:PCA-UMAP"').estimador.item()
        self.assertAlmostEqual(float(sources['absDeltaAuc']['valor']),pca-delta,places=12)
        self.assertEqual(sources['absDeltaAuc']['tex'],'0.013852')
        self.assertEqual(int(sources['umapPass060']['valor']),sum(pd.read_csv(OUT/'F3_seleccion.csv')['pasa_0.60']))
        historic=pd.read_csv(HISTORICAL/'pureza_vecindarios_20_grupos.csv').query('k==10')
        for method in ('Isomap','UMAP'):
            self.assertAlmostEqual(float(sources['historicalP'+method]['valor']),
                                   historic[historic.representacion==method].pureza.mean(),places=12)
        self.assertAlmostEqual(float(sources['historicalPIsomap']['valor'])-
                               float(sources['historicalPUMAP']['valor']),
                               float(sources['c0Delta']['valor']),places=5)
        self.assertEqual(sources['c0Holm']['tex'],'0.04337')
        text='\n'.join(p.read_text() for p in (ARTICLE/'sections').glob('*.tex'))
        self.assertLessEqual(set(re.findall(r'\\Num\{([^}]+)\}',text)),set(sources))
        self.assertIn(r'\eqref{eq:delta} y~\eqref{eq:select}',text)
        self.assertEqual((ARTICLE/'tables/F8_grupos.tex').read_text(),group_table())
        self.assertEqual({r['macro']:r['tex'] for r in rows},
                         {r['macro']:r['tex'] for r in records()+correcciones()})

    def test_sensitivity_tables_and_c0(self):
        with TRACE.open(newline='') as f:
            rows={r['macro']:r for r in csv.DictReader(f)}
        cp=pd.read_csv(OUT/'F2_admisibilidad.csv')
        for i,d in enumerate(DENSIDADES,1):
            for j,t in enumerate(ESTABILIDADES,1):
                eligible=cp[(cp.umbral_densidad==d)&(cp.umbral_estabilidad==t)&cp.admisible]
                rule=eligible.loc[eligible.densidad_mediana.idxmin(),'regla'] if len(eligible) else 'Ninguna'
                self.assertEqual(rows[f'cpCell{i}{j}']['valor'],rule)
        scale=pd.read_csv(OUT/'preF8_RNX_sensibilidad.csv').query('escala=="k9"').set_index('regla_umap')
        for i,rule in enumerate(REGLAS,1):
            u,p,z=scale.loc[rule,['UMAP','PCA','Isomap']]
            label='A ambos' if u>max(p,z) else 'A ninguno' if u<min(p,z) else 'A ISOMAP'
            self.assertEqual(rows[f'nine{i}Class']['valor'],label)
            self.assertAlmostEqual(float(rows[f'nine{i}PCA']['valor']),u-p,places=12)
            self.assertAlmostEqual(float(rows[f'nine{i}Iso']['valor']),u-z,places=12)
        self.assertGreater(float(rows['c0RnxLower']['valor']),0)
        self.assertLess(float(rows['bandRNX1inferior']['valor']),0)
        self.assertGreater(float(rows['bandRNX1superior']['valor']),0)


if __name__=='__main__':
    unittest.main()

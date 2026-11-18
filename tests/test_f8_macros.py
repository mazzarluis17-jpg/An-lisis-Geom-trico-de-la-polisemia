import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import re
import statistics
import unittest

from f9_trazabilidad import ARTICLE, TRACE
from f8_macros import REVISION


def coerce(text, sample):
    if isinstance(sample, bool):
        return {'True': True, 'False': False}[text]
    return type(sample)(text)


def from_source(row):
    path=REVISION/row['origen']
    selector=json.loads(row['filas'])
    columns=json.loads(row['columnas'])
    op=row['operacion']
    if op=='json':
        value=json.loads(path.read_text())
        for field in selector:
            value=value[field]
        return value
    with path.open(newline='') as f:
        raw=[r for r in csv.DictReader(f)
             if all(coerce(r[key],v)==v for key,v in selector.items())]
    if not raw:
        raise AssertionError(row)
    if op=='count':
        return len(raw)
    if op=='countNonfragile':
        return sum(r[columns[0]]=='False' for r in raw)
    if op=='countPositiveBoth':
        return sum(all(float(r[c])>0 for c in columns) for r in raw)
    if op=='weightedContributionPercent':
        terms=[(int(r['k']),float(r[columns[1]])/int(r['k'])) for r in raw]
        return 100*math.fsum(v for k,v in terms if k<=3)/math.fsum(v for _,v in terms)
    if op=='maxAbsDifference':
        return max(abs(float(r[columns[0]])-float(r[columns[1]])) for r in raw)
    values=[float(r[columns[0]]) for r in raw]
    if op=='identity':
        assert len(values)==1
        return values[0]
    if op=='countBelowDensity':
        return sum(v<=.30 for v in values)
    if op=='countExcludesNegative':
        return sum(v<0 for v in values)
    return {'median':statistics.median,'min':min,'max':max}[op](values)


class EditorialTrace(unittest.TestCase):
    def test_links_citations_and_macros(self):
        tex_files=[ARTICLE/'main.tex',*sorted((ARTICLE/'sections').glob('*.tex')),
                   ARTICLE/'tables/F8_grupos.tex']
        content='\n'.join(f.read_text() for f in tex_files)
        with TRACE.open(newline='') as source:
            keys={row['macro'] for row in csv.DictReader(source)}
        self.assertLessEqual(set(re.findall(r'\\Num\{([^}]+)\}',content)),keys)
        labels=re.findall(r'\\label\{([^}]+)\}',content)+re.findall(
            r'^\s*\\figura\{[^\n]+\}\{(fig:[^}]+)\}',content,re.M)
        self.assertEqual(len(labels),len(set(labels)))
        refs={label for arg in re.findall(r'\\(?:c?ref)\{([^}]+)\}',content)
              for label in arg.split(',')}
        self.assertLessEqual(refs,set(labels))
        bib=(ARTICLE/'bibliography/referencias.bib').read_text()
        cites={key for arg in re.findall(r'\\(?:parencite|textcite)\{([^}]+)\}',content)
               for key in arg.split(',')}
        self.assertEqual(cites,set(re.findall(r'@\w+\{([^,]+),',bib)))
        for name in ('01_densidad.pdf','F7_bandas_A4.pdf'):
            self.assertTrue((ARTICLE/'figures'/name).read_bytes().startswith(b'%PDF-'))


if __name__=='__main__':
    unittest.main()

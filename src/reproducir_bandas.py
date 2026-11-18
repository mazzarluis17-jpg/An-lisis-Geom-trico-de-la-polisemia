"""Reproduce los nueve contrastes principales desde los agregados públicos."""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from f7_nucleo import CONFIG, NAMES, calibrate, draw_weights, evaluate, hash_weights
from rutas import PUBLICO

TOL = 1e-12


def reproducir():
    with np.load(PUBLICO/'datos/F7_agregados_principal.npz',allow_pickle=False) as data:
        counts,sums,groups,ids,names=(data[k] for k in ('conteos','sumas','grupos','documentos','contrastes'))
    assert counts.shape==(222,20) and sums.shape==(222,20,9)
    assert (counts>=0).all() and counts.sum()==601 and np.isfinite(sums).all()
    assert list(ids)==[f'D{i+1:03}' for i in range(222)] and list(names)==NAMES
    theta=evaluate(np.ones((1,222)),counts,sums)[0]
    reference=pd.read_csv(PUBLICO/'resultados/F7_bandas_sensibilidad.csv')
    reference=reference.query('configuracion=="principal"').set_index(['mecanismo','contraste'])
    diagnostics=pd.read_csv(PUBLICO/'resultados/F7_pesos_diagnostico.csv').set_index('mecanismo')
    rejection_table=pd.read_csv(PUBLICO/'resultados/F7_rechazos.csv')
    results,checks=[],[]
    for mechanism in CONFIG['mecanismos']:
        weights,attempts,rejected=draw_weights(counts,groups,CONFIG['replicas'],
                                               CONFIG['semilla_'+mechanism],mechanism)
        ref_diag=diagnostics.loc[mechanism]
        assert (attempts,len(weights),len(rejected))==tuple(ref_diag[['intentos','aceptadas','rechazadas']])
        assert rejected==rejection_table.query('mecanismo==@mechanism')[['intento','grupos_ausentes']].to_dict('records')
        assert hash_weights(weights)==ref_diag.pesos_sha256
        replicas=evaluate(weights,counts,sums)
        np.testing.assert_allclose(replicas[:,0::3]+replicas[:,2::3],replicas[:,1::3],atol=TOL,rtol=0)
        se,critical,order,_=calibrate(theta,replicas)
        assert order==19000
        obtained=pd.DataFrame(dict(estimador=theta,SE=se,cuantil=critical,
                                   inferior=theta-critical*se,superior=theta+critical*se),index=NAMES)
        expected=reference.loc[mechanism,obtained.columns].loc[NAMES]
        np.testing.assert_allclose(obtained,expected,atol=TOL,rtol=0)
        checks.append(dict(mecanismo=mechanism,intentos=attempts,rechazadas=len(rejected),
                           **{f'error_max_{k}':v for k,v in (obtained-expected).abs().max().items()}))
        results.extend(dict(mecanismo=mechanism,contraste=name,**row.to_dict())
                       for name,row in obtained.iterrows())
    return results,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida',type=Path,default=PUBLICO/'build'/'bandas_reproducidas.csv')
    args=parser.parse_args()
    rows,checks=reproducir()
    args.salida.parent.mkdir(parents=True,exist_ok=True)
    with args.salida.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for record in checks:
        print(record)
    print(f'{len(rows)} bandas verificadas a tolerancia absoluta {TOL:g}.')


if __name__=='__main__':
    main()

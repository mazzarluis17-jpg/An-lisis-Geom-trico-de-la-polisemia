import unittest

import numpy as np

from reproducir_bandas import TOL, reproducir
from rutas import PUBLICO


class ReproduccionParcial(unittest.TestCase):
    def test_insumo_sin_localizadores(self):
        with np.load(PUBLICO/'datos/F7_agregados_principal.npz',allow_pickle=False) as data:
            self.assertEqual(set(data.files),{'conteos','sumas','grupos','documentos','contrastes'})
            self.assertEqual(data['documentos'].tolist(),[f'D{i:03}' for i in range(1,223)])
            self.assertEqual(data['conteos'].sum(),601)

    def test_contrastes_y_bandas(self):
        rows,checks=reproducir()
        self.assertEqual(len(rows),18)
        self.assertEqual({r['mecanismo'] for r in rows},{'multinomial','exponencial'})
        self.assertTrue(all(v <= TOL for r in checks for k,v in r.items() if k.startswith('error_max_')))
        self.assertEqual({r['mecanismo']:r['rechazadas'] for r in checks},
                         {'multinomial':1,'exponencial':0})


if __name__=='__main__':
    unittest.main()

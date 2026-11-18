"""Reponderación documental con puntuaciones y geometría congeladas."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f0 import digest, verify_hashes
from f7_nucleo import calibrate, draw_weights, evaluate, hash_weights
from f45 import group_metrics
from geometria import vecinos
from rutas import EXP, HISTORICAL, REVISION

OUT = REVISION / 'resultados'
CONFIG = json.loads((REVISION/'config/F7_v1.json').read_text())
SCOPE = CONFIG['alcance']
TOL = CONFIG['tolerancia_calculo']
METHODS = ['PCA', 'Isomap', 'UMAP']
METRICS = ['RNX', 'delta_aj', 'S_aj']
PAIRS = [(0, 1), (0, 2), (1, 2)]
NAMES = [f'{m}:{METHODS[a]}-{METHODS[b]}' for m in METRICS for a,b in PAIRS]


def read(name):
    return json.loads((REVISION/'config'/name).read_text())


def verify_manifest(name):
    manifest = REVISION/'logs'/name
    for line in manifest.read_text().splitlines():
        expected, path = line.split('  ', 1)
        assert digest(REVISION/path) == expected, path


def aligned(table, meta):
    assert not table.duplicated(['id','metodo']).any()
    assert len(table) == len(meta)*len(METHODS) and set(table.metodo)==set(METHODS)
    values = []
    for method in METHODS:
        part = table[table.metodo == method].set_index('id')
        assert set(part.index)==set(meta.id)
        part = part.loc[meta.id]
        assert np.array_equal(part[['grupo','document']].to_numpy(),
                              meta[['group','document']].to_numpy())
        values.append(part[METRICS].to_numpy())
    cube = np.stack(values,axis=1)
    assert np.isfinite(cube).all()
    return cube


def scores(x, projections, labels, historical_checks=None):
    reference = vecinos(x)
    weights = 1/np.arange(1,10)
    weights /= weights.sum()
    items=[]
    for z in projections:
        projected = vecinos(z)
        result=np.zeros((len(x),len(METRICS)))
        for k,w in enumerate(weights,1):
            metrics, detail = group_metrics(x,z,reference,projected,labels,k,
                                            historical_checks is not None and k in (3,5))
            result += w*np.column_stack([detail[name] for name in METRICS])
            if historical_checks is not None and k in (3,5):
                historical_checks.append((k,metrics))
        if historical_checks is not None:
            metrics,_=group_metrics(x,z,reference,projected,labels,10,False)
            historical_checks.append((10,metrics))
        items.append(result)
    return np.mean(items,axis=0)


def load_sources():
    verify_hashes()
    verify_manifest('SHA256SUMS_F3_F5.txt')
    f3=read('F3_definitiva_v1.json')
    cp=read('CP0_v1.json')
    assert f3['umap_regla']=='prop0.15' and f3['semillas']==list(range(42,62))
    assert cp['regla']=='min+0' and digest(REVISION/'config/CP0_v1.json')==f3['CP0_sha256']
    assert digest(OUT/'F3_proyecciones.npz')==f3['sha256_proyecciones']
    meta=pd.read_csv(EXP/'common.csv',usecols=['id','group','document','synset'])
    assert meta.id.is_unique and len(meta)==601 and meta.group.nunique()==20 and meta.document.nunique()==222
    rows=pd.read_csv(EXP/'rows_bert.csv')
    assert meta.id.equals(rows.id) and np.array_equal(rows.row,np.arange(len(meta)))
    saved=pd.read_csv(OUT/'F45_AUC_ocurrencias.csv')
    primary=aligned(saved.query('condicion=="principal"'),meta)
    rules=pd.read_csv(OUT/'F3_reglas.csv').pivot(index='grupo',columns='regla',values='n_neighbors')
    alt={rule:primary.copy() for rule in rules.columns}
    c0=np.empty_like(primary)
    metric_records=[]
    with np.load(OUT/'F3_proyecciones.npz',allow_pickle=False) as z, np.load(
            HISTORICAL/'proyecciones_20_grupos.npz',allow_pickle=False) as hist, np.load(
            OUT/'F1_proyecciones.npz',allow_pickle=False) as control:
        assert set(hist.files)==set(control.files)
        for key in hist.files:
            np.testing.assert_array_equal(hist[key],control[key])
        for group,ix in meta.groupby('group').indices.items():
            p=f'{group}::'
            np.testing.assert_array_equal(z[p+'ids'],meta.iloc[ix].id)
            np.testing.assert_array_equal(ix,z[p+'indices'])
            np.testing.assert_array_equal(ix,hist[p+'indices'])
            x=hist[p+'X_unit']
            labels=meta.iloc[ix].synset.to_numpy()
            for method_i,method in enumerate(METHODS):
                checks=[]
                c0[ix,method_i]=scores(x,[hist[p+method]],labels,checks)
                metric_records.extend(dict(grupo=group,metodo=method,k=k,**m) for k,m in checks)
                if method!='UMAP':
                    np.testing.assert_allclose(primary[ix,method_i],scores(x,[z[p+method]],labels),
                                               atol=TOL,rtol=0)
            by_scale={}
            for k in rules.loc[group].unique():
                batch=z[f'{p}UMAP::{k}']
                assert batch.shape==(20,len(ix),2)
                by_scale[k]=scores(x,batch,labels)
            for rule,k in rules.loc[group].items():
                alt[rule][ix,2]=by_scale[k]
            np.testing.assert_allclose(primary[ix,2],alt['prop0.15'][ix,2],atol=TOL,rtol=0)
    check_historical(pd.DataFrame(metric_records))
    np.testing.assert_allclose(c0,aligned(saved.query('condicion=="C0"'),meta),atol=TOL,rtol=0)
    group_indices=list(meta.groupby('group').indices.values())
    macros=pd.read_csv(OUT/'F45_AUC_macro.csv').set_index(['condicion','metodo'])
    for label,cube in [('principal',primary),('C0',c0)]:
        means=np.mean([cube[ix].mean(axis=0) for ix in group_indices],axis=0)
        np.testing.assert_allclose(means,macros.loc[[(label,m) for m in METHODS],METRICS],
                                   atol=TOL,rtol=0)
    abs5=np.mean([alt['abs5'][ix,2].mean(axis=0) for ix in group_indices],axis=0)
    np.testing.assert_allclose(abs5,[.542318,.013852,.199098],rtol=0,
                               atol=CONFIG['tolerancia_redondeo_6_decimales'])
    print('Fuentes validadas. AUC abs5:',abs5.tolist(),flush=True)
    return meta, {'principal':primary,'abs5':alt['abs5'],'C0':c0},alt


def check_historical(actual):
    keys=['grupo','metodo','k']
    for name,old,new in [('R','R_k','R'),('T','trustworthiness','T'),('delta','delta_pureza','delta')]:
        reference=pd.read_csv(OUT/f'F1_{name}.csv')
        merged=reference.merge(actual,on=keys,validate='one_to_one')
        assert len(merged)==len(reference)
        np.testing.assert_allclose(merged[new],merged[old+'_historico'],atol=TOL,rtol=0)
    p=pd.read_csv(OUT/'F1_P.csv').rename(columns={'representacion':'metodo'})
    original=p.query('metodo=="Original"').merge(actual.query('metodo=="PCA"'),on=['grupo','k'])
    reduced=p.query('metodo!="Original"').merge(actual,on=keys)
    assert len(original)==60 and len(reduced)==180
    np.testing.assert_allclose(original.P_original,original.pureza_historico,atol=TOL,rtol=0)
    np.testing.assert_allclose(reduced.P,reduced.pureza_historico,atol=TOL,rtol=0)


def differences(cube):
    return np.column_stack([cube[:,a,j]-cube[:,b,j]
                            for j in range(len(METRICS)) for a,b in PAIRS])


def aggregate(values, document, group, n_docs, n_groups):
    counts=np.zeros((n_docs,n_groups),dtype=int)
    sums=np.zeros((n_docs,n_groups,values.shape[1]))
    np.add.at(counts,(document,group),1)
    np.add.at(sums,(document,group),values)
    return counts,sums



def direct(weights, values, document, group):
    return np.stack([np.average(values[group==g],axis=0,weights=weights[document[group==g]])
                     for g in np.unique(group)]).mean(axis=0)



def main():
    from rutas import exigir_trabajo_local
    exigir_trabajo_local()
    meta, configurations, scales=load_sources()
    docs, document=np.unique(meta.document,return_inverse=True)
    groups, group=np.unique(meta.group,return_inverse=True)
    internal=np.array([f'D{i+1:03}' for i in range(len(docs))])
    values={key:differences(value) for key,value in configurations.items()}
    counts, _=aggregate(values['principal'],document,group,len(docs),len(groups))
    width=len(docs)
    ones=np.ones((1,width))
    all_weights,draw_summary,rejections={},[],[]
    for mechanism in CONFIG['mecanismos']:
        weights,n,invalid=draw_weights(counts,groups,CONFIG['replicas'],
                                       CONFIG['semilla_'+mechanism],mechanism)
        np.testing.assert_array_equal(weights,draw_weights(counts,groups,CONFIG['replicas'],
                                      CONFIG['semilla_'+mechanism],mechanism)[0])
        all_weights[mechanism]=weights
        draw_summary.append(dict(mecanismo=mechanism,intentos=n,aceptadas=len(weights),
                                 rechazadas=len(invalid),pesos_sha256=hash_weights(weights)))
        rejections.extend(dict(mecanismo=mechanism,**item) for item in invalid)
    descriptive=[]
    for scale,cube in scales.items():
        v=differences(cube)
        estimate=direct(ones[0],v,document,group)
        descriptive.extend(dict(regla_umap=scale,contraste=name,estimador=x,
                                alcance='Descriptivo. Geometría fija y documentos observados.')
                           for name,x in zip(NAMES,estimate))
    bands,monte_carlo,omissions,group_signs=[],[],[],[]
    for condition,v in values.items():
        _,sums=aggregate(v,document,group,width,len(groups))
        theta=evaluate(ones,counts,sums)[0]
        np.testing.assert_allclose(theta,direct(ones[0],v,document,group),atol=TOL,rtol=0)
        for g,tag in enumerate(groups):
            row=v[group==g].mean(axis=0)
            group_signs.extend(dict(configuracion=condition,grupo=tag,contraste=name,estimador=x)
                               for name,x in zip(NAMES,row))
        for mechanism,weights in all_weights.items():
            reps=evaluate(weights,counts,sums)
            for i in (0,len(weights)//2,len(weights)-1):
                np.testing.assert_allclose(reps[i],direct(weights[i],v,document,group),atol=TOL,rtol=0)
            np.testing.assert_allclose(reps[:,::3]+reps[:,2::3],reps[:,1::3],atol=TOL,rtol=0)
            se,c,r,halves=calibrate(theta,reps)
            monte_carlo.append(dict(configuracion=condition,mecanismo=mechanism,cuantil=c,
                                    estadistico_orden=r,cuantil_mitad_1=halves[0],
                                    cuantil_mitad_2=halves[1],diferencia_abs=abs(halves[0]-halves[1]),
                                    alcance=SCOPE))
            for name,est,sd in zip(NAMES,theta,se):
                margin=c*sd
                bands.append(dict(configuracion=condition,mecanismo=mechanism,contraste=name,
                                  estimador=est,SE=sd,cuantil=c,margen=margin,amplitud=2*margin,
                                  inferior=est-margin,superior=est+margin,
                                  incluye_cero=bool(est-margin<=0<=est+margin),
                                  pesos_sha256=hash_weights(weights),alcance=SCOPE))
            if condition=='principal' and mechanism=='multinomial':
                omitted=evaluate(1-np.eye(width),counts,sums)
                for idx,(doc,row) in enumerate(zip(internal,omitted)):
                    for j,(name,new) in enumerate(zip(NAMES,row)):
                        sign=bool(theta[j]*new<0)
                        below=abs(new)<c*se[j]
                        omissions.append(dict(documento=doc,contraste=name,estimador_completo=theta[j],
                                              estimador_omision=new,cambio=new-theta[j],
                                              cambio_abs=abs(new-theta[j]),margen_original=c*se[j],
                                              cambia_signo=sign,bajo_margen_original=below,
                                              fragilidad_documental=sign or below,
                                              n_ocurrencias=int(counts[idx].sum()),
                                              n_grupos=int((counts[idx]>0).sum()),alcance=SCOPE))
    b=pd.DataFrame(bands)
    assert len(b)==54 and not b.duplicated(['configuracion','mecanismo','contraste']).any()
    assert (b.groupby('mecanismo').pesos_sha256.nunique()==1).all()
    main_b=b.query('configuracion=="principal" and mecanismo=="multinomial"').copy()
    influence=pd.DataFrame(omissions)
    sc=pd.DataFrame(descriptive)
    fragility=[]
    for row in main_b.itertuples():
        other=b[(b.contraste==row.contraste)&~((b.configuracion=='principal')&(b.mecanismo=='multinomial'))]
        inclusive=other[other.incluye_cero]
        directions=other[other.estimador*row.estimador<0].configuracion.unique().tolist()
        grid_changes=sc[(sc.contraste==row.contraste)&(sc.estimador*row.estimador<0)].regla_umap.tolist()
        omission=influence[influence.contraste==row.contraste]
        fragility.append(dict(contraste=row.contraste,banda_principal_excluye_cero=not row.incluye_cero,
                              fragilidad_inferencial=not row.incluye_cero and not inclusive.empty,
                              sensibilidades_incluyen_cero='|'.join(inclusive.configuracion+'/'+inclusive.mecanismo),
                              fragilidad_direccional=bool(directions or grid_changes),
                              configuraciones_signo_opuesto='|'.join(directions),reglas_signo_opuesto='|'.join(grid_changes),
                              omisiones_signo_opuesto=int(omission.cambia_signo.sum()),
                              omisiones_bajo_margen=int(omission.bajo_margen_original.sum()),
                              omisiones_fragilidad=int(omission.fragilidad_documental.sum()),
                              alcance=SCOPE))
    exp=b.query('mecanismo=="exponencial"')
    multi=b.query('mecanismo=="multinomial"')
    keys=['configuracion','contraste']
    mech=multi[keys+['SE','amplitud','incluye_cero']].merge(
        exp[keys+['SE','amplitud','incluye_cero']],on=keys,suffixes=('_multi','_exp'),validate='one_to_one')
    mech['dif_SE_exp_menos_multi']=mech.SE_exp-mech.SE_multi
    mech['dif_amplitud_exp_menos_multi']=mech.amplitud_exp-mech.amplitud_multi
    mech['cambia_inclusion_cero']=mech.incluye_cero_exp!=mech.incluye_cero_multi
    mech['alcance']=SCOPE
    np.savez_compressed(OUT/'F7_pesos.npz',documentos=internal,**all_weights)
    for name,frame in {
        'estimadores_principales':main_b[['contraste','estimador','alcance']],
        'bandas_principales':main_b,'bandas_sensibilidad':b,
        'escalas_umap':sc,'influencia_documental':influence,
        'fragilidad':pd.DataFrame(fragility),'monte_carlo':pd.DataFrame(monte_carlo),
        'pesos_diagnostico':pd.DataFrame(draw_summary),
        'rechazos':pd.DataFrame(rejections,columns=['mecanismo','intento','grupos_ausentes']),
        'comparacion_pesos':mech,'diferencias_grupo':pd.DataFrame(group_signs)
    }.items():
        frame.to_csv(OUT/f'F7_{name}.csv',index=False)
    verify_hashes()
    print(pd.DataFrame(draw_summary).to_string(index=False),flush=True)


if __name__=='__main__':
    main()

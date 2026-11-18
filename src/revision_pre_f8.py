"""Derivaciones descriptivas pre-F8 con proyecciones y resultados congelados."""
import json

import numpy as np
import pandas as pd

from f0 import verify_hashes, digest
from f7 import verify_manifest
from geometria import vecinos
from rutas import EXP, REVISION

OUT = REVISION / 'resultados'
TOL = 1e-12


def rnx_curve(x, y):
    n = len(x)
    result=[]
    for k in range(1,10):
        hits=(x[:,:k,None]==y[:,None,:k]).any(axis=2).mean()
        c=k/(n-1)
        result.append((hits-c)/(1-c))
    return np.array(result)


def scale_tables():
    rules = pd.read_csv(OUT/'F3_reglas.csv')
    configs=json.loads((REVISION/'config/F3_definitiva_v1.json').read_text())
    assert configs['umap_regla']=='prop0.15' and configs['semillas']==list(range(42,62))
    assert digest(OUT/'F3_proyecciones.npz')==configs['sha256_proyecciones']
    assert digest(REVISION/'config/CP0_v1.json')==configs['CP0_sha256']
    assert rules.query('regla=="prop0.15"').set_index('grupo').n_neighbors.to_dict()==configs['umap_vecinos']
    parts=[]
    with np.load(OUT/'F3_proyecciones.npz',allow_pickle=False) as z, np.load(
            OUT/'F1_proyecciones.npz',allow_pickle=False) as ref:
        for group, spec in rules.groupby('grupo'):
            p=f'{group}::'
            np.testing.assert_array_equal(z[p+'indices'],ref[p+'indices'])
            rank=vecinos(ref[p+'X_unit'])
            pca,iso=(rnx_curve(rank,vecinos(z[p+m])) for m in ('PCA','Isomap'))
            umaps={}
            for k in spec.n_neighbors.unique():
                batch=z[f'{p}UMAP::{k}']
                assert batch.shape==(20,len(rank),2)
                umaps[k]=np.mean([rnx_curve(rank,vecinos(v)) for v in batch],axis=0)
            for row in spec.itertuples():
                for k,(a,b,c) in enumerate(zip(pca,iso,umaps[row.n_neighbors]),1):
                    parts.append((group,row.regla,k,a,b,c))
    long=pd.DataFrame(parts,columns=['grupo','regla_umap','k','PCA','Isomap','UMAP'])
    test=long.query('regla_umap=="prop0.15"').melt(
        id_vars=['grupo','k'],value_vars=['PCA','Isomap','UMAP'],var_name='metodo',value_name='reconstruido')
    f4=pd.read_csv(OUT/'F4_grupos.csv').query('condicion=="principal" and k<=9')
    check=test.merge(f4,on=['grupo','metodo','k'],validate='one_to_one')
    assert len(check)==len(test)==len(f4)
    np.testing.assert_allclose(check.reconstruido,check.RNX,atol=TOL,rtol=0)
    by_k=long.groupby(['regla_umap','k'])[['PCA','Isomap','UMAP']].mean()
    summaries=[]
    for rule, d in by_k.groupby(level=0):
        d=d.droplevel(0)
        weights=1/d.index.to_numpy()
        for label,v in [('AUC_1_sobre_k',np.average(d,axis=0,weights=weights)),
                        ('AUC_uniforme',d.mean()),('k5',d.loc[5]),('k9',d.loc[9])]:
            summaries.append((rule,label,*v))
    result=pd.DataFrame(summaries,columns=['regla_umap','escala','PCA','Isomap','UMAP'])
    result['UMAP_menos_PCA']=result.UMAP-result.PCA
    result['UMAP_menos_Isomap']=result.UMAP-result.Isomap
    for label,filename in [('AUC_uniforme','F45_AUC_macro_uniforme.csv'),
                           ('AUC_1_sobre_k','F45_AUC_macro.csv')]:
        ref=pd.read_csv(OUT/filename).query('condicion=="principal"').set_index('metodo').RNX
        row=result.query('regla_umap=="prop0.15" and escala==@label').iloc[0]
        np.testing.assert_allclose(row[ref.index].to_numpy(dtype=float),ref,rtol=0,atol=TOL)
    f7=pd.read_csv(OUT/'F7_escalas_umap.csv')
    prim=result.query('escala=="AUC_1_sobre_k"').set_index('regla_umap')
    for name,col in [('RNX:PCA-UMAP','UMAP_menos_PCA'),('RNX:Isomap-UMAP','UMAP_menos_Isomap')]:
        ref=f7.query('contraste==@name').set_index('regla_umap').estimador
        np.testing.assert_allclose(prim.loc[ref.index,col],-ref,rtol=0,atol=TOL)
    for frame in (by_k,long):
        frame['UMAP_menos_PCA']=frame.UMAP-frame.PCA
        frame['UMAP_menos_Isomap']=frame.UMAP-frame.Isomap
    return result,by_k.reset_index(),long


def selectivity(frame):
    d=frame.copy()
    active=d.eventos_sustitucion>0
    assert (active == (d.fraccion_sustituida>TOL)).all()
    np.testing.assert_allclose(d.loc[~active,'S'],0,atol=TOL,rtol=0)
    d['sigma']=d.S.div(d.fraccion_sustituida.where(active))
    formula=(d.P_in-d.q_ponderada)-(d.P_out-d.P_original_ponderada)
    np.testing.assert_allclose(d.sigma,formula,atol=TOL,rtol=0,equal_nan=True)
    np.testing.assert_allclose(d.fraccion_sustituida*d.sigma.fillna(0),d.S,atol=TOL,rtol=0)
    d['sin_sustituciones']=~active
    return d


def sigma_tables():
    d=selectivity(pd.read_csv(OUT/'F5_descomposicion_grupo.csv'))
    ns=pd.read_csv(OUT/'F0_grupos.csv').set_index('group').n
    reps=np.where((d.metodo=='UMAP') & (d.condicion!='C0'),20,1)
    d['oportunidades']=d.grupo.map(ns)*d.k*reps
    np.testing.assert_allclose(d.eventos_sustitucion/d.oportunidades,d.fraccion_sustituida,
                               atol=TOL,rtol=0)
    rows=[]
    for keys,g in d.groupby(['condicion','metodo','k']):
        v=g.fraccion_sustituida.to_numpy()
        s=g.S.to_numpy()
        e=g.eventos_sustitucion.to_numpy()
        active=e>0
        events=int(e.sum())
        sigma=g.sigma.to_numpy()
        pooled=np.average(sigma[active],weights=e[active]) if events else np.nan
        slot_fraction=events/g.oportunidades.sum()
        slot_selectivity=np.average(s,weights=g.oportunidades)
        if events:
            np.testing.assert_allclose(slot_fraction*pooled,slot_selectivity,atol=TOL,rtol=0)
        rows.append((*keys,len(g),int(active.sum()),int((~active).sum()),events,int(g.oportunidades.sum()),
                     v.mean(),s.mean(),g.sigma.mean(),pooled,slot_fraction,slot_selectivity))
    columns=['condicion','metodo','k','n_grupos','n_activos','n_sin_sustituciones',
             'eventos','oportunidades','fraccion_macro','S_macro','sigma_macro_activos',
             'sigma_agrupada_eventos','fraccion_agrupada','S_agrupada']
    return d,pd.DataFrame(rows,columns=columns)


def military():
    data=pd.read_csv(EXP/'common.csv',usecols=['id','group','document'])
    docs=np.sort(data.document.unique())
    code={doc:f'D{i:03}' for i,doc in enumerate(docs,1)}
    incidence=pd.crosstab(data.group,data.document).reindex(columns=docs)
    counts=incidence.loc['military#a']
    active=counts[counts>0]
    assert len(active)==11 and active.sum()==23
    n_docs=(incidence>0).sum(axis=1)
    diagnosis=pd.read_csv(OUT/'F7_rechazos.csv')
    assert diagnosis.grupos_ausentes.tolist()==['military#a']
    sharing=pd.DataFrame(dict(documento=active.index.map(code),ocurrencias_military=active.to_numpy(),
                              ocurrencias_total=incidence[active.index].sum(axis=0).to_numpy(),
                              grupos_total=(incidence[active.index]>0).sum(axis=0).to_numpy()))
    deltas=pd.read_csv(OUT/'F7_diferencias_grupo.csv').query('configuracion=="principal"')
    full=pd.read_csv(OUT/'F7_bandas_principales.csv').set_index('contraste')
    doc_influence=pd.read_csv(OUT/'F7_influencia_documental.csv')
    rows=[]
    for name,d in deltas.groupby('contraste',sort=False):
        v=d.set_index('grupo').estimador
        other=v.drop('military#a')
        mu=v.mean()
        np.testing.assert_allclose(mu,full.at[name,'estimador'],rtol=0,atol=TOL)
        without=other.mean()
        value=v.at['military#a']
        np.testing.assert_allclose(without-mu,(mu-value)/19,atol=TOL,rtol=0)
        ddocs=doc_influence[(doc_influence.contraste==name)&doc_influence.documento.isin(sharing.documento)]
        rows.append(dict(contraste=name,military=value,macro_20=mu,macro_sin_military=without,
                         cambio=without-mu,mediana_otros=other.median(),min_otros=other.min(),
                         max_otros=other.max(),rango_ascendente=int((v<value).sum()+1),
                         omisiones_documentos_military_cambia_signo=int(ddocs.cambia_signo.sum()),
                         max_abs_omision_documento_military=ddocs.cambio_abs.max()))
    stats=dict(documentos=len(docs),documentos_military=len(active),ocurrencias_military=int(active.sum()),
               docs_por_grupo_mediana=float(n_docs.median()),docs_por_grupo_min=int(n_docs.min()),
               docs_por_grupo_max=int(n_docs.max()),military_docs_multigrupo=int((sharing.grupos_total>1).sum()),
               prob_ausencia_military=(1-len(active)/len(docs))**len(docs),
               intento_rechazado=int(diagnosis.intento.iloc[0]))
    return sharing,pd.DataFrame(rows),stats


def main():
    from rutas import exigir_trabajo_local
    exigir_trabajo_local()
    verify_hashes()
    for manifest in ('SHA256SUMS_entrega.txt','SHA256SUMS_F3_F5.txt','SHA256SUMS_F7.txt'):
        verify_manifest(manifest)
        print(f'Manifiesto validado: {manifest}')
    scales,curves,groups=scale_tables()
    sigma,summary=sigma_tables()
    docs,influence, stats=military()
    for name,data in {
        'RNX_sensibilidad':scales,'RNX_curvas':curves,'RNX_por_grupo':groups,
        'selectividad_grupo':sigma,'selectividad_resumen':summary,
        'military_documentos':docs,'military_contrastes':influence,
    }.items():
        data.to_csv(OUT/f'preF8_{name}.csv',index=False)
    (OUT/'preF8_military_resumen.json').write_text(json.dumps(stats,indent=2)+'\n')
    verify_hashes()
    print(f'Filas de selectividad: {len(sigma)}. Sin sustituciones: {sigma.sin_sustituciones.sum()}.')


if __name__=='__main__':
    main()

"""Fit once on train only, then cache field-free covariates for every split."""
from common import *
from features import geometry,inputs
from calibration import fit_training

def run():
    verify_freeze();assert read(OUT/'data_complete.json')['count']==180
    records=read(OUT/'geometry_manifest.json')['records'];tr=[r for r in records if r['split']=='train']
    if (OUT/'calibration.json').exists():
        cal=read(OUT/'calibration.json');assert cal['passed']
        assert [r['geometry_id'] for r in cal['rows']]==[r['geometry_id'] for r in tr]
        assert all(sha(OUT/'data'/r['geometry_id']/'sample.npz')==r['sample_sha256'] for r in cal['rows'])
    else:cal=fit_training(tr)
    stats=[];(OUT/'cache').mkdir(exist_ok=True)
    for r in records:
        path=OUT/'cache'/(r['geometry_id']+'.npz')
        # Only geometry/material/source keys: do not load test Ez here.
        with np.load(OUT/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
            ep=z['eps'];sm=z['src_mask'];x=z['x_um'];y=z['y_um'];src=z['source_center_um']
        d,cost,f=geometry(r['params'],x,y,src,r['input_port'])
        assert cost['v2_max_axis_gradient']<=5,'Frozen v2 geometry gate failed; pause without substitution'
        beta=float(np.polyval(cal['coefficients_rad_um'],ep.max()));assert beta>0
        inp,carriers=inputs(ep,sm,x,y,src,r['input_port'],beta,d)
        X,Y=np.meshgrid(x,y);dist=f['projection']['distance'];w=r['params']['wg_width_um'];core=dist<=w/2
        bend=core&(X>=f['bend_um'][0,0])&(X<=f['bend_um'][-1,0])
        masks={'core':core,'bend_core':bend,'halo':(dist>w/2)&(dist<=w/2+.5),'far':dist>w/2+.5}
        if path.exists():raise RuntimeError('Existing geometry cache; do not silently overwrite')
        np.savez_compressed(path,x=inp,**{'carrier_'+k:v for k,v in carriers.items()},**{'region_'+k:v for k,v in masks.items()},beta_rad_um=beta)
        stats.append({'geometry_id':r['geometry_id'],'split':r['split'],'beta_rad_um':beta,**cost,'cache_sha256':sha(path)})
    write('cache_complete.json',{'time_utc':now(),'records':stats,'calibration_sha256':sha(OUT/'calibration.json'),
       'test_fields_loaded':False,'common_features_for_all_methods':True})
    print('TRAINING PREPARATION PASSED; beta relative RMSE='+str(cal['relative_rmse']),flush=True)
if __name__=='__main__':run()

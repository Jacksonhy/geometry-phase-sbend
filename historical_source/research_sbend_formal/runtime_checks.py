"""Analytic-field calibration and metrics checks, no new geometry data."""
import ast
import numpy as np
from common import *
import calibration,evaluate
def main():
    for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    r=read(DEV/'sbend_index.json')[0];x=np.arange(480)*.05-11.975;y=np.arange(160)*.05-3.975
    fits=[]
    for ip in [1,2]:
        direction=1 if ip==1 else -1;E=np.broadcast_to(np.exp(1j*direction*9.25*x)[None],(160,480))
        q=calibration.extract(E,x,y,r['params'],ip);assert abs(q['beta_rad_um']-9.25)<1e-10;fits.append(q)
    true=np.ones((160,480),np.complex128);pred=true*1.1
    regions={'core':np.ones(true.shape,bool),'bend_core':np.ones(true.shape,bool),'halo':np.ones(true.shape,bool),'far':np.ones(true.shape,bool)}
    fm=evaluate.field_metrics(pred,true,regions);assert max(abs(v-.1) for v in fm.values())<1e-12
    proxy=evaluate.proxy_function()
    with np.load(DEV/read(DEV/'sbend_index.json')[0]['sample'],allow_pickle=False) as z:
        c=proxy(z['Ez'],z['eps'],z['port_masks'],z['port_ids'],int(z['input_port']),1.55,.05)
        exported_eps_difference=float(np.max(abs(c-z['S_Ez_proxy'])))
        # Historical generator projected with float64 simulation epsilon before
        # exporting its float32 copy. Reproduce with the actual original input.
        original=proxy(z['Ez'],z['full/eps'][20:-20,20:-20],z['port_masks'],z['port_ids'],int(z['input_port']),1.55,.05)
        assert np.max(abs(original-z['S_Ez_proxy']))<1e-12
    write('runtime_checks.json',{'passed':True,'time_utc':now(),'analytic_beta_both_directions':fits,
        'analytic_relative_error':fm,'development_proxy_original_inputs_exact_reproduction':True,
        'exported_float32_eps_proxy_difference':exported_eps_difference,'formal_fields_read':False})
    print('ANALYTIC RUNTIME CHECKS PASSED',flush=True)
if __name__=='__main__':main()

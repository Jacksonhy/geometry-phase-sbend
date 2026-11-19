import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from support import OUT,V1,OLD,read
def main():
    edge=read(OUT/'gradient_checks.json');probe=read(OUT/'continuity_probes.json');summary=read(OUT/'phase_summary.json')
    methods=['path','path_v2','local_tangent'];labels=['Path v1','Path v2','LocalTangent']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for m,label in zip(methods,labels):
        vals=[max(r['max_abs_coordinate_gradient'] for r in edge if r['geometry_id']==rec['geometry_id'] and r['method']==m and r['band']=='all') for rec in read(OLD/'sbend_index.json')]
        axes[0].plot(np.arange(1,17),vals,'o-',label=label)
        eps=sorted({r['epsilon_um'] for r in probe},reverse=True)
        vals=[max(r['max_abs_distance_difference_um'] for r in probe if r['method']==m and r['kind']=='old_worst_cut' and r['epsilon_um']==e) for e in eps]
        axes[1].loglog(eps,vals,'o-',label=label)
    axes[0].set(xlabel='Development geometry',ylabel='Maximum coordinate gradient',yscale='log')
    axes[1].set(xlabel='Probe half spacing (um)',ylabel='Max distance-coordinate difference (um)')
    for i,m in enumerate(['linear',*methods]):
        vals=[]
        for axis in ['propagation','normal']:
            s=next(r for r in summary if r['frame']=='curvilinear' and r['axis']==axis and r['band']=='core' and r['zone']=='all' and r['gate']=='amplitude_1pct' and r['group']=='all' and r['baseline']==('path' if m=='path_v2' else m))
            vals.append(s['v2_mean'] if m=='path_v2' else s['baseline_mean'])
        axes[2].bar(np.arange(2)+(i-1.5)*.18,vals,width=.18,label={'linear':'Linear','path':'Path v1','path_v2':'Path v2','local_tangent':'LocalTangent'}[m])
    axes[2].set_xticks([0,1],['Along s','Normal']);axes[2].set_ylabel('Core residual weighted RMS (rad/um)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'continuity_and_core.png',dpi=170);plt.close(fig)
    # The historical worst geometry is fixed independently of the candidate.
    worst=max((r for r in edge if r['method']=='path' and r['band']=='all'),key=lambda r:r['max_abs_coordinate_gradient'])
    gid=worst['geometry_id']
    with np.load(V1/'maps'/f'{gid}.npz') as old,np.load(OUT/'maps'/f'{gid}.npz') as new:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,m,label in zip(axes,methods,labels):
            D=new['path_v2_um'] if m=='path_v2' else old[m+'_um']
            gx=np.gradient(D,.05,axis=1);gy=np.gradient(D,.05,axis=0);grad=np.hypot(gx,gy)
            im=ax.imshow(np.log10(np.maximum(grad,.01)),extent=[old['x_um'][0],old['x_um'][-1],old['y_um'][0],old['y_um'][-1]],origin='lower',aspect='auto',vmin=-.1,vmax=2,cmap='magma')
            poly=old['polyline_um'];ax.plot(poly[:,0],poly[:,1],'c-',lw=.6);ax.set(title=label,xlabel='x (um)',ylabel='y (um)')
        fig.colorbar(im,ax=axes.tolist(),label='log10 coordinate gradient norm',fraction=.02)
        fig.suptitle(gid+' (same geometry; no field-dependent carrier)')
        fig.savefig(OUT/'gradient_maps.png',dpi=170,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()

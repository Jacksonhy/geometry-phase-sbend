import csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,OLD,read

def main():
    g=read(OUT/'geometry_checks.json');records=read(OLD/'sbend_index.json');cal=read(OUT/'calibration_transfer.json')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ip,marker in [(1,'o'),(2,'s')]:
        sel=[r for r in cal if r['input_port']==ip]
        axs[0].scatter([r['width_um'] for r in sel],[100*r['relative_error_Y_vs_MPB'] for r in sel],marker=marker,label=f'input {ip}, vs MPB')
        axs[0].scatter([r['width_um'] for r in sel],[100*r['relative_error_Y_vs_field'] for r in sel],marker=marker,facecolors='none',edgecolors='black',label=f'input {ip}, vs FDTD slope')
    axs[0].axhline(0,color='gray',lw=.8);axs[0].set_xlabel('Guide width (um)');axs[0].set_ylabel('Y calibration relative difference (%)');axs[0].legend(fontsize=7);axs[0].set_title('Two propagation-constant references')
    sums=read(OUT/'phase_summary.json')
    for method in ['linear','column_arc','local_tangent']:
        values=[]
        for axis in ['propagation','normal']:
            r=next(r for r in sums if r['frame']=='curvilinear' and r['axis']==axis and r['zone']=='all' and r['band']=='core' and r['gate']=='amplitude_1pct' and r['group']=='all' and r['metric']=='weighted_rms' and r['baseline']==method)
            values.append(r['baseline_mean'])
        axs[1].plot([0,1],values,'o-',label=method)
    r0=next(r for r in sums if r['frame']=='curvilinear' and r['axis']=='propagation' and r['zone']=='all' and r['band']=='core' and r['gate']=='amplitude_1pct' and r['group']=='all' and r['metric']=='weighted_rms' and r['baseline']=='linear')
    r1=next(r for r in sums if r['frame']=='curvilinear' and r['axis']=='normal' and r['zone']=='all' and r['band']=='core' and r['gate']=='amplitude_1pct' and r['group']=='all' and r['metric']=='weighted_rms' and r['baseline']=='linear')
    axs[1].plot([0,1],[r0['path_mean'],r1['path_mean']],'s--',label='path')
    axs[1].set_xticks([0,1],['Propagation (s coordinate)','Transverse normal']);axs[1].set_ylabel('Mean amplitude-weighted RMS (rad/um)');axs[1].legend();axs[1].set_title('16 development geometries, core')
    for ax in axs:ax.grid(alpha=.25)
    fig.savefig(OUT/'calibration_and_phase.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(4,4,figsize=(15,10),constrained_layout=True)
    for rec,ax in zip(records,axs.flat):
        gid=rec['geometry_id']
        with np.load(OUT/'profiles'/f'{gid}.npz') as d:
            i=int(np.argmin(abs(d['n_um'])));s=d['s_um'];E=d['Ez_samples'][:,i];phase=np.unwrap(np.angle(E));points=d['points_um'][:,i,:]
            keep=(points[:,0]>=-11)&(points[:,0]<=11);s=s[keep];phase=phase[keep]
            for method in ['linear','local_tangent','path']:
                z=phase-d[method+'_phase'][keep,i];z=z-z[0]
                ax.plot(s,z,label=method,ls='--' if method=='local_tangent' else '-',lw=1)
            ax.set_title(f"{gid[-6:]} input {rec['input_port']}; R={rec['params']['R_min_um']:.3f}",fontsize=8);ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.supxlabel('Centerline arclength s (um), +x direction');fig.supylabel('Residual phase relative to first ROI sample (rad)')
    fig.savefig(OUT/'all16_centerline_residuals.png',dpi=170);plt.close(fig)
    # Worst global coordinate gradient, explicitly selected to reveal adverse behavior.
    far=read(OUT/'far_field_extension.json');worst=max((r for r in far if r['method']=='path'),key=lambda r:r['max_abs_coordinate_gradient'])
    gid=worst['geometry_id'];rec=next(r for r in records if r['geometry_id']==gid)
    with np.load(OUT/'maps'/f'{gid}.npz') as m,np.load(OLD/rec['sample']) as d:
        x=m['x_um'];y=m['y_um'];extent=[-12,12,-4,4];fig,axs=plt.subplots(3,1,figsize=(11,8),constrained_layout=True,sharex=True)
        a=axs[0].imshow(d['eps'],origin='lower',extent=extent,aspect='auto',cmap='gray_r');poly=m['polyline_um'];axs[0].plot(poly[:,0],poly[:,1],'r-',lw=.8)
        src=m['source_center_um'];axs[0].scatter(src[0],src[1],color='blue',marker='x');axs[0].set_title(f'{gid}: geometry and exact centerline; selected for WORST far-field gradient')
        for ax,name in zip(axs[1:],['path','local_tangent']):
            im=ax.imshow(m[name+'_um'],origin='lower',extent=extent,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label=f'{name} distance (um)')
            ax.plot(poly[:,0],poly[:,1],'w-',lw=.5);ax.scatter(worst['x_um'],worst['y_um'],marker='x',color='red');ax.set_title(name)
        for ax in axs:ax.set_ylabel('y (um)')
        axs[-1].set_xlabel('x (um)');fig.savefig(OUT/'far_field_failure.png',dpi=180);plt.close(fig)
    # Local failures on the worst aggregate local-increase example, all data remain in CSV.
    fails=read(OUT/'local_failure_cases.json');sub=[r for r in fails if r['baseline']=='linear' and r['axis']=='propagation' and r['zone']=='core_bend']
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar(np.arange(16),[100*r['path_worse_weight_fraction'] for r in sub]);ax.set_xticks(np.arange(16),[r['geometry_id'][-6:] for r in sub],rotation=60)
    ax.set_ylabel('Amplitude weight where Path local |gradient| is worse (%)');ax.set_title('Bend/core local failures vs Linear; no samples excluded')
    fig.savefig(OUT/'local_adverse_regions.png',dpi=180);plt.close(fig)
    print('4 diagnostic figures saved, including adverse examples.')
if __name__=='__main__':main()

"""Plots from saved formal metrics only; no additional evaluation."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import *
LABEL=['Direct','Linear','LocalTangent','Path v1','Path v2']
def main():
    summary=read(OUT/'test_summary.json');lookup={(r['method'],r['checkpoint'],r['group']):r for r in summary}
    with (OUT/'test_per_geometry.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    data={m:{r['geometry_id']:float(r['rel_l2']) for r in rows if r['method']==m and r['checkpoint']=='best'} for m in METHODS}
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);xx=np.arange(5)
    for c,shift in [('best',-.18),('last',.18)]:
        ax.bar(xx+shift,[lookup[m,c,'all']['rel_l2'] for m in METHODS],.35,label=c)
    ax.set_xticks(xx,LABEL);ax.set_ylabel('Mean per-geometry complex RelL2');ax.legend();ax.set_title('100 new S-bend geometries; training seed 2026')
    fig.savefig(OUT/'formal_five_methods.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,base in zip(axs,['linear','local_tangent','path_v1']):
        ids=sorted(data[base]);a=np.array([data[base][g] for g in ids]);b=np.array([data['path_v2'][g] for g in ids])
        ax.scatter(a,b,c=np.where(b<a,'tab:blue','tab:red'),s=15,alpha=.8)
        lim=max(a.max(),b.max())*1.03;ax.plot([0,lim],[0,lim],'k--',linewidth=.8);ax.set_xlim(0,lim);ax.set_ylim(0,lim)
        ax.set_xlabel(base+' RelL2');ax.set_ylabel('Path v2 RelL2');ax.set_title(str(int(sum(b<a)))+'/100 improved')
    fig.savefig(OUT/'paired_geometry_scatter.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,key in enumerate(['input1','input2']):
        axs[0].bar(xx+(i-.5)*.35,[lookup[m,'best',key]['rel_l2'] for m in METHODS],.35,label=key)
    axs[0].set_xticks(xx,LABEL,rotation=15);axs[0].set_ylabel('Mean RelL2');axs[0].legend()
    keys=['core_rel_l2','bend_core_rel_l2','halo_rel_l2','far_rel_l2']
    for i,m in enumerate(METHODS):axs[1].bar(np.arange(4)+(i-2)*.15,[lookup[m,'best','all'][k] for k in keys],.15,label=LABEL[i])
    axs[1].set_xticks(np.arange(4),['core','bend core','halo','far']);axs[1].set_ylabel('Regional RelL2');axs[1].legend(fontsize=8)
    fig.savefig(OUT/'direction_and_region_results.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()

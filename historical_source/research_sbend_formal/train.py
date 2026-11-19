"""Five equal budgets; never opens test fields or test caches."""
import time,random,traceback
import torch
from torch import nn
from common import *
import model

def capture_rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def savept(p,obj):
    p=Path(p);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)
def load_data():
    records=[r for r in read(OUT/'geometry_manifest.json')['records'] if r['split'] in ['train','val']]
    records.sort(key=lambda r:0 if r['split']=='train' else 1)
    assert len(records)==80 and all(r['split']=='train' for r in records[:64])
    cal=read(OUT/'calibration.json');xx=[];yy=[];cc={m:[] for m in METHODS}
    for r in records:
        with np.load(OUT/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
            xx.append(z['x'])
            for m in METHODS:cc[m].append(z['carrier_'+m])
        with np.load(OUT/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
            assert str(z['split']) in ['train','val'];yy.append((z['Ez'][None]/cal['scale']).astype(np.complex64))
    return {'x':torch.from_numpy(np.stack(xx)),'y':torch.from_numpy(np.stack(yy)),
        'c':{m:torch.from_numpy(np.stack(a)) for m,a in cc.items()},'records':records,'meta':cal}
@torch.no_grad()
def score(m,data,ids,name):
    m.eval();values=[]
    for j in range(0,len(ids),2):
        ind=ids[j:j+2];p=model.forward(m,data['x'][ind].cuda(),data['c'][name][ind].cuda())
        values.extend(model.rel_mse(p,data['y'][ind].cuda()).sqrt().cpu().tolist())
    return np.array(values)
def train_one(data,name):
    run=OUT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
    if (run/'summary.json').exists():
        s=read(run/'summary.json');assert sha(run/'best.pt')==s['best_sha256'] and sha(run/'last.pt')==s['last_sha256'];return s
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=model.new();initial=model.state_hash(m)
    expected=read(OUT/'preflight.json')['model_checks'][0]['initial_state_sha256'];assert initial==expected
    batches=model.batches();bh=ahash(batches);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2400,eta_min=.0001)
    history=[];best=float('inf');best_step=0;elapsed=0;first=1;resumes=0
    if (run/'last.pt').exists():
        ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert ck['initial_state_sha256']==initial and ck['batches_sha256']==bh
        m.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
        history=ck['history'];best=ck['best_val'];best_step=ck['best_step'];elapsed=ck['seconds'];first=ck['step']+1
        resumes=ck.get('resumes',0)+1;restore_rng(ck['rng']);del ck
    elif any(run.iterdir()):raise RuntimeError('Incomplete run has no restorable state')
    start=time.perf_counter()
    for step in range(first,2401):
        ids=batches[step-1];m.train();opt.zero_grad(set_to_none=True)
        pred=model.forward(m,data['x'][ids].cuda(),data['c'][name][ids].cuda())
        loss=model.rel_mse(pred,data['y'][ids].cuda()).mean();assert torch.isfinite(loss)
        loss.backward();grad=nn.utils.clip_grad_norm_(m.parameters(),5);assert torch.isfinite(grad)
        opt.step();sch.step()
        if step==1 or step%50==0:
            values=score(m,data,np.arange(64,80),name);val=float(values.mean())
            row={'step':step,'loss':float(loss.detach()),'val_rel_l2':val,'seconds':elapsed+time.perf_counter()-start};history.append(row)
            if val<best:
                best=val;best_step=step
                savept(run/'best.pt',{'state':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'step':step,'method':name,
                    'initial_state_sha256':initial,'batches_sha256':bh,'val_rel_l2':best,'config':model.CONFIG,'seed':SEED})
            write(run/'history.json',history);print(name+' '+__import__('json').dumps(row),flush=True)
        if step%200==0:
            savept(run/'last.pt',{'state':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'rng':capture_rng(),
                'step':step,'method':name,'initial_state_sha256':initial,'batches_sha256':bh,'history':history,
                'best_val':best,'best_step':best_step,'seconds':elapsed+time.perf_counter()-start,'resumes':resumes,'config':model.CONFIG,'seed':SEED})
    seconds=elapsed+time.perf_counter()-start;last=score(m,data,np.arange(64,80),name)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['state']);replay=score(m,data,np.arange(64,80),name)
    assert abs(float(replay.mean())-best)<1e-7
    train=score(m,data,np.arange(64),name)
    s={'method':name,'steps':2400,'seed':SEED,'initial_state_sha256':initial,'batches_sha256':bh,'best_step':best_step,
       'best_val':best,'last_val':float(last.mean()),'best_train':float(train.mean()),'val_replay_error':abs(float(replay.mean())-best),
       'train_wall_seconds':seconds,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
       'best_sha256':sha(run/'best.pt'),'last_sha256':sha(run/'last.pt'),'test_evaluated':False,'resumes':resumes}
    write(run/'summary.json',s);del m,opt,sch,ck;return s
def main():
    verify_freeze();assert (OUT/'cache_complete.json').exists();model.configure();data=load_data();results=[]
    for name in METHODS:
        results.append(train_one(data,name));write('training_progress.json',results)
    assert len({r['initial_state_sha256'] for r in results})==len({r['batches_sha256'] for r in results})==1
    write('training_complete.json',{'time_utc':now(),'results':results,
        'weight_hashes':{str(p.relative_to(OUT)):sha(p) for p in (OUT/'runs').glob('*/*.pt')},
        'calibration_sha256':sha(OUT/'calibration.json'),'test_evaluated':False})
if __name__=='__main__':
    try:main()
    except Exception as e:write('training_failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'time_utc':now()});raise

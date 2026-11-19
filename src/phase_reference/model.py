"""Original Fourier bodies, shared initialization and common physical loss."""
import sys,random
import numpy as np
import torch
from .paths import ROOT
SEED=2026
from .compat.official_architecture import load_original_model_class
CLASS=load_original_model_class()
CONFIG=dict(in_channels=14,out_channels=2,dim=64,kernel_list=[64]*12,kernel_size_list=[1]*12,
  padding_list=[1]*12,hidden_list=[256],mode_list=[(40,70)]*12,act_func='GELU',dropout_rate=.1,
  drop_path_rate=0,eps_min=2.0851359,eps_max=6.2851359,pos_encoding='exp',conv_stem=True,
  aug_path=False,ffn=True,ffn_dwconv=True,with_cp=False)
def configure():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
def new(seed=2026, device="cuda"):
    global SEED
    SEED=seed
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    return CLASS(**CONFIG,device=torch.device(device)).to(device)
def state_hash(m):
    import hashlib
    h=hashlib.sha256()
    for n,p in m.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def batches():
    rng=np.random.default_rng(SEED)
    return np.stack([rng.choice(64,2,replace=False) for _ in range(2400)])
def forward(m,x,c):
    w=torch.full((len(x),1),1.55,device=x.device);g=torch.full((len(x),2),.05,device=x.device)
    return m(x,w,g).to(torch.complex64)*c
def rel_mse(p,y):return abs(p-y).square().sum((1,2,3))/abs(y).square().sum((1,2,3))

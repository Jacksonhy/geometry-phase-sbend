"""Offline compatibility harness for original NeurOLight classes (not a trained baseline).

MIT, original copyright (c) 2022 Jiaqi Gu. Source class bodies are not edited.
Only dependency names are supplied locally; GELU path and drop_path_rate=0 smoke
do not exercise Swish or the replaced DropPath implementation.
"""
import ast
import functools
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.checkpoint import checkpoint
from .official_bridge import ROOT, SOURCE, OriginalOperator


class CompatDropPath(nn.Module):
    """Standard per-example stochastic depth; compatibility replacement, not upstream timm."""
    def __init__(self,drop_prob=0.):
        super().__init__(); self.drop_prob=float(drop_prob)
    def forward(self,x):
        if self.drop_prob==0 or not self.training: return x
        keep=1-self.drop_prob
        mask=torch.empty((x.shape[0],)+(1,)*(x.ndim-1),dtype=x.dtype,device=x.device).bernoulli_(keep)
        return x*mask/keep


def unavailable_legacy_seed(*args,**kwargs):
    raise RuntimeError('Legacy pyutils deterministic helper is not reproduced; seed PyTorch externally and use reset_parameters(None).')


def load_original_model_class():
    path=SOURCE/'core/models/layers/fno_conv2d.py'
    spec=importlib.util.spec_from_file_location('original_fno_operator',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    env=dict(torch=torch,nn=nn,F=F,np=np,lru_cache=functools.lru_cache,
             FNOConv2d=m.FNOConv2d,NeurOLightConv2d=OriginalOperator,
             set_torch_deterministic=unavailable_legacy_seed,
             to_2tuple=_pair,DropPath=CompatDropPath,Swish=nn.SiLU,checkpoint=checkpoint)
    for path in [SOURCE/'core/models/pde_base.py',SOURCE/'core/models/neurolight_cnn.py']:
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module,str(path),'exec'),env)
    return env['NeurOLight2d']





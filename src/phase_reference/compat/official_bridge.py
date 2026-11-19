"""Offline loading of original NeurOLight routines, with no full legacy stack.

Original code is MIT licensed, Copyright (c) 2022 Jiaqi Gu.
See references/NeurOLight-source/NeurOLight-main/LICENSE.
"""
import ast
import functools
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
SOURCE=Path(__file__).resolve().parents[3]/'third_party/NeurOLight'


def original_prior_class():
    path=SOURCE/'core/models/pde_base.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PDE_NN_BASE')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_get_linear_pos_enc','get_grid']]
    newcls=ast.ClassDef(name='OriginalPrior',bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),newcls],type_ignores=[])
    ast.fix_missing_locations(module)
    env=dict(nn=nn,torch=torch,np=np,lru_cache=functools.lru_cache)
    exec(compile(module,str(path),'exec'),env)
    return env['OriginalPrior']


OriginalPrior=original_prior_class()


def original_operator_class():
    path=SOURCE/'core/models/layers/neurolight_conv2d.py'
    spec=importlib.util.spec_from_file_location('original_neurolight_operator',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NeurOLightConv2d


OriginalOperator=original_operator_class()


def wave_prior(epsilon,wavelength=1.55,dx=.05,dy=.05):
    # Original ordering: x axis (image columns), real/imag, then y axis real/imag.
    a=torch.as_tensor(epsilon,dtype=torch.float32)
    if a.ndim==3: a=a[:,None]
    n=a.shape[0]
    prior=OriginalPrior()
    return prior.get_grid(a.shape,a.device,mode='exp',epsilon=a,
        wavelength=torch.full((n,1),wavelength,dtype=a.dtype,device=a.device),
        grid_step=torch.tensor([dx,dy],dtype=a.dtype,device=a.device).repeat(n,1))





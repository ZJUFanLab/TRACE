# -*- encoding: utf-8 -*-
from __future__ import absolute_import, annotations, print_function
from typing import Optional
import tempfile, shutil, logging, time
import contextlib, os
from multiprocessing import Pool

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import random
import numpy as np

logger = logging.getLogger(__name__)
# logging
separator = ">" * 30
line = "-" * 30

@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
  """创建上下文管理器，创建和删除临时文件"""
  tmpdir = tempfile.mkdtemp(dir=base_dir)
  try:
    yield tmpdir
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

@contextlib.contextmanager
def timing(msg: str):#计算时间差
  logger.info('Started %s', msg)
  tic = time.time()
  yield
  toc = time.time()
  logger.info('Finished %s in %.3f seconds', msg, toc - tic)
  

def mol_conformers(mol:Mol):
  mol.RemoveAllConformers()
  ps = AllChem.ETKDGv2()
  id = AllChem.EmbedMolecule(mol, ps) # 生成3维几何构象
  if id == -1:
    logger.info('rdkit pos could not be generated without using random pos. using random pos now.')
    ps.useRandomCoords = True
    AllChem.EmbedMolecule(mol, ps)
    AllChem.MMFFOptimizeMolecule(mol, confId=0)
  return mol

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map
  
  
class dict_to_object(object):
    """
    对 dict 对象的属性样式访问，并且完全按照 OP 的要求进行操作
    Args:
        object (_type_): _description_
    """
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(key, (list, tuple)):
                setattr(self, key, [dict_to_object(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, dict_to_object(value) if isinstance(value, dict) else value)
                
class config_to_object(object):
    def __init__(self, d):
        for key, value in d.items():
            setattr(self, key, value) 
            
                 
def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, dict):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(obj))
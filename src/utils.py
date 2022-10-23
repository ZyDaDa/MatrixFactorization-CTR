from doctest import script_from_examples
import os
import torch
import random
import numpy as np
import torch

from sklearn.metrics import roc_auc_score

def metrics(true, score):

    auc = roc_auc_score(true, score)

    return {'auc':auc}



def fix_seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

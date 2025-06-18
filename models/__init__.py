# -*- encoding: utf-8 -*-


from __future__ import absolute_import, print_function, annotations

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, LinearLR

from .losses import LabelSmoothingLoss, FocalLoss, DiceLoss, GHMLoss, MDNLoss
from models.graph_transformer import GraphTransformerNetwork


loss_mapping = {
    "LabelSmoothingLoss":LabelSmoothingLoss,
    "FocalLoss" : FocalLoss,
    "DiceLoss": DiceLoss,
    "GHMLoss" : GHMLoss,
    "CELoss": nn.CrossEntropyLoss,
    "BCELoss": nn.BCEWithLogitsLoss,
    "MSELoss": nn.MSELoss,
    "MDNLoss": MDNLoss
}

activtion_maping = {
    "relu":nn.ReLU(),
    "silu": nn.SELU(),
    "leakyrelu":nn.LeakyReLU()
}


graph_mapping = {
    "graphtransformer": GraphTransformerNetwork
}


scheduler_mapping = {
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CyclicLR": CyclicLR,
    "LinearLR": LinearLR
}

__all__ = ['loss_mapping', 'activtion_maping', 'pretrain_mapping', 'graph_mapping', 'scheduler_mapping']

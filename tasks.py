# -*- encoding: utf-8 -*-
from __future__ import annotations, print_function, absolute_import

import torch as th
import torch.nn as nn
import math
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np

from torch_geometric.nn import Set2Set, global_mean_pool

from utils import common
from models.layers import MultiLayerPerceptron as MLP
from models.losses import LabelSmoothingLoss
from models import loss_mapping

class Task(nn.Module):
            
    def preprocess(self, train_dataset,  val_dataset=None):
        return train_dataset, val_dataset, {}
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def metric(self, *args, **kwargs):
        raise NotImplementedError
        
class InteractionPrediction(Task):
    def __init__(self, 
                 model,
                 num_class:int,
                 model2=None,
                 criterion:str="CELoss",
                 mlp_num_layers:int=1,
                 mlp_dropout_rate:float=0.5,
                 mlp_activation = "relu",
                 mlp_short_cut:bool=False,
                 mlp_batch_norm:bool=False,
                 mlp_hidden_size:int=2048,
                 **kwargs
                 ) -> None:
        super(InteractionPrediction, self).__init__()
        self.model = model
        self.model2 = model2 if model2 else model
        self.criterion = criterion
        self.num_class = num_class
        self.num_layers = mlp_num_layers
        self.dropout_rate = mlp_dropout_rate
        self.short_cut = mlp_short_cut
        class_weight_dir = kwargs.pop("class_weight_dir", None)
        label_smoothing = kwargs.pop("label_smoothing", 0.0)
        
        if class_weight_dir is not None:
            class_weight = pd.read_csv(class_weight_dir)["weight"].tolist()
            class_weight = th.tensor(class_weight,dtype=th.float32)
        else:
            class_weight = None
            
        if criterion=="BCELoss":
            self.criterion_fn = loss_mapping[criterion](weight=class_weight)
            
        elif criterion == "CELoss":
            self.criterion_fn = loss_mapping[criterion](weight=class_weight, label_smoothing=label_smoothing)
            
        elif criterion == "LabelSmoothingLoss":
            self.criterion_fn = loss_mapping[criterion](classes=num_class, smoothing=label_smoothing)
            
        elif criterion == "FocalLoss":
            if class_weight_dir is None:
                raise ValueError
            self.criterion_fn = loss_mapping[criterion](alpha=class_weight, gamma=2, with_logits=True) 
        elif criterion == "DiceLoss":
            self.criterion_fn = loss_mapping[criterion](square_denominator=True, set_level=True, with_logits=True)
        
        elif criterion == "GHMLoss":
            if num_class>2:
                raise ValueError
            self.criterion_fn = loss_mapping[criterion](bins=10,  momentum=0, with_logits=True)

            
            
        hidden_dims = [mlp_hidden_size] * (self.num_layers - 1)
        self.final_mlp = MLP(self.model.output_dim + self.model2.output_dim, 
                            hidden_dims + [self.num_class], 
                            activation=mlp_activation,
                            dropout=mlp_dropout_rate,
                            short_cut=mlp_short_cut,
                            batch_norm=mlp_batch_norm)

    def forward(self, batch):
        metric = {}
        smiles1_id, smiles2_id, g1, g2, labels = batch
        
            
        features1 = {'node': g1.ndata['atom'], 'edge': g1.edata['bond']}
        features2 = {'node': g2.ndata['atom'], 'edge': g2.edata['bond']}
        h1= self.model(g1, features1)
        h2 = self.model2(g2, features2) 
        h = th.cat([h1, h2], dim=1)
        
        logits = self.final_mlp(h)
        
        loss = None
        results = (loss, metric, labels)
        if labels is not None:
            if self.num_class == 2:
                label_one_hot = labels
                if len(labels.shape)==1:
                    label_one_hot = th.nn.functional.one_hot(labels, num_classes=2)
                loss = self.criterion_fn(logits, label_one_hot.float())
            elif self.num_class == 86:
                loss = self.criterion_fn(logits, labels)    
            metric['loss'] = loss.item()
            results = (loss, metric, labels)

        return results
        
    def predict(self, batch):
        smiles1_id, smiles2_id, g1, g2, labels = batch
            
        features1 = {'node': g1.ndata['atom'], 'edge': g1.edata['bond']}
        features2 = {'node': g2.ndata['atom'], 'edge': g2.edata['bond']}
        h1 = self.model(g1, features1)
        h2 = self.model2(g2, features2)      
        h = th.cat([h1, h2], dim=1)
        logits = self.final_mlp(h)
        _, pred = th.max(logits.data, 1)
        return pred, logits, labels
    
    
    def metric(self, y_pred, y_true, logits):
        batch_iter = len(y_pred)
        #y_pred = common.cat(y_pred)
        y_true = common.cat(y_true)
        y_logits = common.cat(logits)
        
        metric = {}

        if  self.num_class == 2: # binary classification 
            y_pred = common.cat(y_pred)
            label_one_hot = y_true
            if len(y_true.shape)==1:
                label_one_hot = th.nn.functional.one_hot(y_true, num_classes=2)
            loss = self.criterion_fn(y_logits, label_one_hot.float())
            metric["loss"] = loss.item() / batch_iter
            y_scores = th.sigmoid(y_logits)

            pos_probs = y_scores[:, 1].detach().cpu().numpy()
            y_true_np = y_true.cpu().numpy()

            if len(np.unique(y_true_np)) > 1:  
                metric['macro_auc'] = roc_auc_score(y_true_np, pos_probs)
                metric['macro_aupr'] = average_precision_score(y_true_np, pos_probs)


            if th.is_tensor(y_pred):
                y_pred = y_pred.cpu().numpy()
            if th.is_tensor(y_true):
                y_true = y_true.cpu().numpy()
            
            metric_report = classification_report(y_true=y_true, 
                                                y_pred=y_pred, 
                                                output_dict=True)
            for k,v in metric_report.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk != "support":
                            metric[k+"_"+kk]=vv
                else:
                    metric[k] = v
            
        elif self.num_class == 86:# multi-class classification
            y_pred = common.cat(y_pred)
            loss = self.criterion_fn(y_logits, y_true)
            metric["loss"] = loss.item() / batch_iter
            y_scores = F.softmax(y_logits, dim=1)
            if th.is_tensor(y_pred):
                y_pred = y_pred.cpu().numpy()
            if th.is_tensor(y_true):
                y_true = y_true.cpu().numpy()
            y_true_one_hot = label_binarize(y_true, classes=range(self.num_class))
            metric_report = classification_report(y_true=y_true, 
                                                y_pred=y_pred, 
                                                output_dict=True)
            for k,v in metric_report.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk != "support":
                            metric[k+"_"+kk]=vv
                else:
                    metric[k] = v
            
            y_scores = y_scores.cpu()
            auc_scores = []
            aupr_scores = []
            for i in range(self.num_class):
                if len(np.unique(y_true_one_hot[:, i])) > 1:  # Ensure at least two classes are present
                    auc_scores.append(roc_auc_score(y_true_one_hot[:, i], y_scores[:, i].numpy()))  # Convert to numpy after ensuring it's on CPU
                    aupr_scores.append(average_precision_score(y_true_one_hot[:, i], y_scores[:, i].numpy()))

            # Append average AUC and AUPR to metrics
            if auc_scores:
                metric['macro_auc'] = np.mean(auc_scores)
            if aupr_scores:
                metric['macro_aupr'] = np.mean(aupr_scores)

        #y_scores = F.softmax(y_logits, dim=1) if self.num_class > 2 else th.sigmoid(y_logits)
 
        
        return metric
    
    def preprocess(self, dataset):
        
        return super().preprocess(dataset)
    
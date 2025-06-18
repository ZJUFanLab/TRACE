# -*- encoding: utf-8 -*-
from __future__ import annotations, print_function, absolute_import
import os
import json
import numpy as np
import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.smiles2dgl import smiles_to_dgl, graph_construction_and_featurization


def collate(samples):
    smiles1_id, smiles2_id, graph1, graph2, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graph1)
    batched_graph2 = dgl.batch(graph2)
    labels = torch.tensor(labels, dtype=torch.long)
    
    smiles1_id = torch.tensor(smiles1_id)
    smiles2_id = torch.tensor(smiles2_id)
    return smiles1_id, smiles2_id, batched_graph1, batched_graph2, labels

class DrugInteractionDataset(Dataset):
    def __init__(self, 
                 csv_file, 
                 graph_file=None, 
                 id_to_index_file:str=None, 
                 num_class = 2,):
        self.data_frame = pd.read_csv(csv_file)
        self.smiles1_ids = self.data_frame['smiles_1'].tolist()
        self.smiles2_ids = self.data_frame['smiles_2'].tolist()
        self.num_class = num_class
        self.labels = self.data_frame['label'].tolist()

        self.id_to_index = self.load_id_to_index(id_to_index_file)
        
             
        # If a file with preprocessed graphs is provided, try to load it
        if graph_file and os.path.isfile(graph_file):
            self.id_to_graph = torch.load(graph_file)
            self.graph1 = [self.id_to_graph[s1] for s1 in self.smiles1_ids]
            self.graph2 = [self.id_to_graph[s2] for s2 in self.smiles2_ids]
            self.labels = [l for g1, g2, l in list(zip(self.graph1, self.graph2, self.labels)) if g1 is not None and g2 is not None]
            self.smiles1_ids = [s1 for g1, g2, s1 in list(zip(self.graph1, self.graph2, self.smiles1_ids)) if g1 is not None and g2 is not None]
            self.smiles2_ids = [s2 for g1, g2, s2 in list(zip(self.graph1, self.graph2, self.smiles2_ids)) if g1 is not None and g2 is not None]

    def smiles_to_graph(self, smiles):
        graph = smiles_to_dgl(smiles)
        if graph is None:
            print("Warning: Invalid SMILES string:", smiles)
        return graph

    
    
    def load_id_to_index(self, load_id_to_index_file):
        id_to_index = {}
        if load_id_to_index_file is None:
            return id_to_index
        
        # with open(load_id_to_index_file, "r", encoding="utf-8") as fr:
        #     lines = fr.readlines()
        #     for line in lines:
        #         line = line.strip()
        #         if line:
        #             line_list = line.split("\t")
        #             if line_list[0] not in id_to_index:
        #                 id_to_index[line_list[0]] = len(id_to_index)

        # 使用pandas读取csv，假设第一列为ID列
        df = pd.read_csv(load_id_to_index_file)
        id_col = df.columns[0]  # 取第一列
        for idx, val in enumerate(df[id_col]):
            if val not in id_to_index:
                id_to_index[val] = idx
        return id_to_index

    
    def __getitem__(self, idx):
        smiles1_id = torch.tensor(self.id_to_index[self.smiles1_ids[idx]], dtype=torch.long)
        smiles2_id = torch.tensor(self.id_to_index[self.smiles2_ids[idx]], dtype=torch.long)
        graph1 = self.graph1[idx]
        graph2 = self.graph2[idx]
        #label = torch.tensor(self.labels[idx], dtype=torch.long)
        label = self.labels[idx]
        return smiles1_id, smiles2_id, graph1, graph2, label
    
    def __len__(self):
        return len(self.labels)
import os
import json
import torch
import csv
import argparse
from tqdm import tqdm
from utils.smiles2dgl import smiles_to_dgl, graph_construction_and_featurization
from utils.io import read_config_yaml
from utils.common import dict_to_object

class GraphPreprocessor:
    def __init__(self, smiles_file, graph_file):
        self.smiles_file = smiles_file
        self.graph_file = graph_file
        self.id_to_graph = {}
        
    def preprocess(self):
        try:
            with open(self.smiles_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                lines = [(f"{row['DrugID']}", row["Smiles"]) for row in reader]
                
            for cid, smiles in tqdm(lines, desc="Processing SMILES", ncols=100):
                try:
                    graph = smiles_to_dgl(smiles)
                    
                    if graph is not None:
                        self.id_to_graph[cid] = graph
                except Exception as e:
                    print(f"Error processing SMILES {smiles}: {e}")
                    
            torch.save(self.id_to_graph, self.graph_file)
            print(f"Graphs saved to {self.graph_file}")
                
        except Exception as e:
            print(f"Error reading SMILES file: {e}")

# 初始化GraphPreprocessor并进行预处理
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ddi model') 
    parser.add_argument('--config_file', type=str, required=True, help='配置路径')
    config = parser.parse_args()
    args = read_config_yaml(config.config_file)
    args = dict_to_object(args)
    smiles_file = args.id_to_index_dir
    graph_file = args.data_bin_dir    
    graph_preprocessor = GraphPreprocessor(smiles_file, graph_file)
    graph_preprocessor.preprocess()
# -*- encoding: utf-8 -*-

from __future__ import annotations, print_function, absolute_import
import argparse

from torch.utils.data import random_split

from data import DrugInteractionDataset
from models.graph_transformer import GraphTransformerNetwork
from tasks import InteractionPrediction
from utils.common import dict_to_object, seed_everything
from utils.io import read_config_yaml
from trainer import TaskTrainer

def main(config):
    args = read_config_yaml(config.config_file)
    args = dict_to_object(args)
    seed_everything(args.seed)
    dataset = DrugInteractionDataset(args.data_dir, 
                                     graph_file=args.data_bin_dir,
                                     id_to_index_file=args.id_to_index_dir,
                                     num_class = args.num_class)

    num_total = len(dataset)
    num_train = int(num_total * 0.6)  # 60% of the total data for the training set
    num_val = int(num_total * 0.2)    # 20% of the total data for the validation set
    num_test = num_total - num_train - num_val  # Remaining 20% for the test set

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])
    # train_dataset = DrugInteractionDataset(args.train_data_dir,
    #                                        graph_file=args.data_bin_dir,
    #                                        id_to_index_file=args.id_to_index_dir,
    #                                        num_class=args.num_class)
    # val_dataset = DrugInteractionDataset(args.valid_data_dir,
    #                                        graph_file=args.data_bin_dir,
    #                                        id_to_index_file=args.id_to_index_dir,
    #                                        num_class=args.num_class)
    # test_dataset = DrugInteractionDataset(args.test_data_dir,
    #                                        graph_file=args.data_bin_dir,
    #                                        id_to_index_file=args.id_to_index_dir,
    #                                        num_class=args.num_class)

    # dataset = DrugInteractionDataset(args.data_dir, 
    #                                  graph_file=args.data_bin_dir,
    #                                  id_to_index_file=args.id_to_index_dir,
    #                                  num_class = args.num_class)

    # num_total = len(dataset)
    # num_train = int(num_total * 0.8)  # 60% of the total data for the training set
    # num_val = int(num_total * 0.2)    # 20% of the total data for the validation set

    # train_dataset, val_dataset = random_split(dataset, [num_train, num_val])


            
    model = GraphTransformerNetwork(in_feats=args.in_feats,
                                    in_edge_feats=args.in_edge_feats,
                                    hidden_size=args.hidden_size,
                                    num_layers=args.num_layers)
    # print(model)
    task = InteractionPrediction(model=model, 
                                 model2=model, 
                                 **args.__dict__)
    
    trainer = TaskTrainer(task=task, args=args)
    # trainer.train(train_dataset=train_dataset, 
    #               val_dataset=val_dataset)
    trainer.train(train_dataset=train_dataset, 
                  val_dataset=val_dataset, 
                  test_dataset = test_dataset)
    trainer.test(test_dataset=test_dataset)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ddi model') 
    parser.add_argument('--config_file', type=str, required=True, help='配置路径')
    config = parser.parse_args()
    main(config)
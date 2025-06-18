# -*- encoding: utf-8 -*-
from __future__ import annotations, print_function, absolute_import
from collections import Counter
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from utils.logger import LoggingLogger
from utils.early_stopping import EarlyStopping
from data import collate as default_collate
from tasks import Task
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import hamming_loss


class TaskTrainer(object):
    def __init__(self, task:Task, args, current_fold = None) -> None:
        self.model = task
        self.args = args
        self.current_fold = current_fold
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.gpu else "cpu")
        self.model.to(self.device)
          
        self.logger = LoggingLogger()
            
        self.logger.log_config(args.__dict__)
        self.early_stop = False
        if self.args.patience > 0:
            self.stopper = EarlyStopping(patience=self.args.patience, 
                                         mode=self.args.early_stop_mode, 
                                         checkpoint_dir=self.args.checkpoint_dir,
                                         filename=self.args.experiment_name)
    
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=args.learning_rate) 
        
    def train(self, 
              train_dataset, 
              val_dataset=None, 
              test_dataset=None,
              collate_fn=default_collate):
        
        
        # train_dataset, val_dataset, statistics = self.model.preprocess(train_dataset, 
        #                                                                val_dataset)
        ## 统计数据
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.args.batch_size, 
                                  shuffle=True, 
                                  collate_fn=collate_fn)
        
        best_epoch_info = {}  
        best_score = None  
        best_model = None
        best_test_metrics = {}
        best_test_epoch_info = {}  
        for epoch in range(self.args.num_epochs):
            self.model.train()
            total_loss = 0
            all_labels = []
            for step, batch in enumerate(train_loader):
                batch_step = epoch *len(train_loader) + step
                batch = tuple(d.to(self.device) for d in batch)
                loss, metrics, labels= self.model(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                all_labels.extend(labels.cpu().tolist())
                total_loss += loss.item()
                #self.logger.log(metrics, step_id=epoch, category="train/batch")

            avg_loss = total_loss / len(train_loader)
            self.logger.log({"average loss": avg_loss}, step_id=epoch, category="train/epoch")
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.args.num_epochs, avg_loss))
            if epoch == 0:
                label_count = Counter(all_labels)
                label_count = [[k, v] for k, v in label_count.items()]
                label_count = sorted(label_count, key=lambda x : x[0])
                self.logger.logger.warning(label_count)
            if val_dataset:
                metric_evaluate = self.evaluate(dataset=val_dataset, collate_fn=collate_fn)
                         
                self.logger.log(record=metric_evaluate, step_id=epoch, category="valid/epoch")
            

            test_metrics = None
            if test_dataset is not None:
                test_metrics = self.evaluate(dataset=test_dataset, collate_fn=collate_fn)
                self.logger.log(record=test_metrics, step_id=epoch, category="test/epoch")

            
            current_score = metric_evaluate[self.args.early_stop_criteria]
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_model = self.model.state_dict()
                torch.save(best_model, self.args.best_model_dir)
                print(f'Saved new best model with {self.args.early_stop_criteria}: {best_score} at epoch {epoch}')
                best_epoch_info = {
                    'epoch': epoch,
                    'metrics': metric_evaluate
                }
            


            if self.args.patience > 0:
                self.early_stop = self.stopper.step(model=self.model, score=metric_evaluate[self.args.early_stop_criteria])

            if self.stopper.best_score is not None and (best_score is None or self.stopper.best_score > best_score):
                    best_score = self.stopper.best_score
                    best_epoch_info = {
                        'epoch': epoch,
                        'metrics': metric_evaluate
                    }
            

            
            if self.early_stop:
                break

        
        self.logger.close()    
 
    def test(self, 
            test_dataset=None,
            collate_fn=default_collate):
        self.model.load_state_dict(torch.load(self.args.best_model_dir, map_location=self.device))
        self.model.to(self.device)
        test_metrics = self.evaluate(dataset=test_dataset, collate_fn=collate_fn)
        # for key, value in test_metrics.items():
        #     print(f"{key}: {value}")
        print("test metrics:")
        self.logger.summary(test_metrics,None) 
        self.logger.close()
  
                            
    def evaluate(self, 
                 dataset, 
                 collate_fn=default_collate):
        self.model.eval()  
        
        loader = DataLoader(dataset, 
                                batch_size=self.args.batch_size, 
                                shuffle=False, 
                                collate_fn=collate_fn)
        y_pred = []
        y_true = []
        y_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = tuple(d.to(self.device) for d in batch)
                pred, logits, labels = self.model.predict(batch)
                y_pred.append(pred)
                y_true.append(labels)
                y_logits.append(logits)
        metric = self.model.metric(y_pred, y_true, y_logits)
        
        return metric
    
    @staticmethod
    def predict(dataset):
        pass
    
    
    def load_model(self):
        pass

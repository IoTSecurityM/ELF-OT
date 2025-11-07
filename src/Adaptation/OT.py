# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

import ot

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from mal_dataset import load_processed_features, load_laplaOT_procFea
from model import ClassClassifier

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def emd_mapping(solu_name, ot_name, net, train_X, train_y, test_X, test_y, model_config):
    
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=train_X.detach().cpu().numpy(), Xt=test_X.detach().cpu().numpy()) 
    
    train_X_mapped_numpy = ot_emd.transform(Xs=train_X.detach().cpu().numpy())
    train_X_mapped_tensor = torch.tensor(train_X_mapped_numpy, dtype=torch.float32)
        
    dataset = TensorDataset(train_X_mapped_tensor, train_y)
    dataloader = DataLoader(dataset, batch_size=model_config['batch_size'], shuffle=True)
    
    class_counts = Counter(train_y.numpy())
    total_samples = len(train_y.numpy())
    class_weight = {cls: total_samples / count for cls, count in class_counts.items()}
    class_weights_tensor = torch.tensor([class_weight[i] for i in sorted(class_weight.keys())], dtype=torch.float)
    
    class_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=model_config["lr"], weight_decay=model_config["weight_decay"])
    
    for epoch in range(500):
        total_loss, epoch_step = 0, 0 
        net.train()
        for xb, yb in dataloader:
            optimizer.zero_grad()
            outputs = net(xb)
            loss = class_criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_step += 1
        
        
def others_train(experiment_path, solu_name, ot_name, net, train_X, train_y, test_X, test_y, config, checkpoint_dir=None):

        # ********************************* Dataset *********************************
        
        # ---------- Load Data
        train_X, train_y, test_X, test_y = load_processed_features(config['model_config']['arch_name'], config['model_config']['train_year'], config['model_config']['test_year'], config['model_config']['dt_index'])
        
        if config['ot_name'] == "EntropyRegularization":
        
            ot_sinkhorn = ot.da.SinkhornTransport(reg_e=config['lambda'])
            ot_sinkhorn.fit(Xs=train_X.detach().cpu().numpy(), Xt=test_X.detach().cpu().numpy())
           
            train_X_mapped_numpy = ot_sinkhorn.transform(Xs=train_X.detach().cpu().numpy())
            train_X_mapped_tensor = torch.tensor(train_X_mapped_numpy, dtype=torch.float32)
            
        elif config['ot_name'] == "GroupSparsity":
            
            ot_lpl1 = ot.da.SinkhornLpl1Transport(reg_e=config['lambda'], reg_cl=config['eta'])
            ot_lpl1.fit(Xs=train_X.detach().cpu().numpy(), ys=train_y.detach().cpu().numpy(), Xt=test_X.detach().cpu().numpy())
            
            train_X_mapped_numpy = ot_lpl1.transform(Xs=train_X.detach().cpu().numpy())
            train_X_mapped_tensor = torch.tensor(train_X_mapped_numpy, dtype=torch.float32)
            
        elif config['ot_name'] == "LaplacianOT":
            
            train_X_mapped_tensor = load_laplaOT_procFea(config['model_config']['arch_name'], config['model_config']['train_year'], config['model_config']['test_year'], config['model_config']['dt_index'], config['eta'])
        
        train_ds = TensorDataset(train_X_mapped_tensor, train_y)
        train_loader = DataLoader(train_ds, batch_size=config['model_config']['batch_size'], shuffle=True)
        # ********************************* Models *********************************
            
        # ---------- Initiate Models
        net = ClassClassifier(train_X.shape[1], config['model_config']['class_classifier_hidden_size_1'], config['model_config']['class_classifier_dropout_rate_1'], config['model_config']['class_classifier_hidden_size_2'], 3)
        
        # ---------- Loss criteria + Optimization
        class_counts = Counter(train_y.numpy())
        total_samples = len(train_y.numpy())
        class_weight = {cls: total_samples / count for cls, count in class_counts.items()}
        class_weights_tensor = torch.tensor([class_weight[i] for i in sorted(class_weight.keys())], dtype=torch.float)
        
        class_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(net.parameters(), lr=config['model_config']["lr"], weight_decay=config['model_config']["weight_decay"])
        
        
        # ---------- Training + Validation     
        for epoch in range(500):  # Adjust epochs
            
            # ---------- Training
            net.train()
            
            total_loss = 0
            epoch_steps = 0  
            predicted_list, label_list = [], []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = net(xb)
                
                _, predicted = torch.max(outputs, 1)
                
                loss = class_criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                epoch_steps += 1
                
                predicted_list.extend(predicted.tolist())
                label_list.extend(yb.tolist())       
    
            
    
    
    
        
  
    
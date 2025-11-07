# -*- coding: utf-8 -*-
import os
import numpy as np
import csv
from collections import Counter
import itertools
import pandas as pd
import json
from scipy.spatial.distance import cdist 

import ot

from functools import partial

import sklearn.metrics as ms
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset

from mal_dataset import load_processed_features
from model import ClassClassifier
from withoutAdapt import get_besthyperparams, no_adaption

def train(solu_name, net, Xtest, Yst, class_weights_tensor, config):
    
    dataset = TensorDataset(Xtest, Yst)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    if solu_name == 'jdotM':  
    # -------------- Cross entropy loss
        for epoch in range(300):
            total_loss, epoch_step = 0, 0 
            net.train()
            for xb, yb in dataloader:
                optimizer.zero_grad()
    
                logits = net(xb)
                log_probs = F.log_softmax(logits, dim=1)
               
                per_sample_loss = -torch.sum(yb * log_probs, dim=1) 
                weights = torch.sum(yb * class_weights_tensor, dim=1)  # shape [B]
                per_sample_loss = per_sample_loss * weights  # elementwise scaling
               
                loss = per_sample_loss.mean()
    
                loss.backward()
                optimizer.step()
               
                epoch_step += 1
                total_loss += loss.item()
        
    
    elif solu_name == 'jdotH':
    # -------------- Hinge loss
        for epoch in range(200):
            total_loss, epoch_step = 0, 0 
            net.train()
            for xb, yb in dataloader:
                optimizer.zero_grad()
    
                logits = net(xb)
                
                yb_signed = 2 * yb - 1  # Now in [-1, 1]
    
                margin = 1 - yb_signed * logits  # [B, C]
                hinge_loss = torch.clamp(margin, min=0) ** 2  # [B, C]
                
                weights = torch.sum(yb * class_weights_tensor, dim=1).unsqueeze(1)  # [B, 1]
                hinge_loss = hinge_loss * weights  # [B, C]
                
                loss = hinge_loss.mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_step += 1
                total_loss += loss.item()
        

    return net


def classification(solu_name, config, best_model_path, checkpoint_dir=None):
    
    # ********************************* Dataset *********************************
    
    # ---------- Load Data
    train_X, train_y, test_X, test_y = load_processed_features(config['arch_name'], config['train_year'], config['test_year'], config['dt_index'])
    
    # ---------- Optimal Transport Initialization
    n = train_X.shape[0]
    ntest = test_X.shape[0]
    wa = np.ones((n,))/n
    wb = np.ones((ntest,))/ntest
    
    C0 = cdist(train_X.detach().cpu().numpy(), test_X.detach().cpu().numpy(), metric='sqeuclidean')
    C0 = C0/np.max(C0)
    
    alpha = 1
    
    # ********************************* Models *********************************
  
    # ---------- Initiate Models
    net = ClassClassifier(train_X.shape[1], config['class_classifier_hidden_size_1'], config['class_classifier_dropout_rate_1'], config['class_classifier_hidden_size_2'], 3)

    checkpoint_dir = best_model_path
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_dir"))
    
    net.load_state_dict(model_state)
    
    # ---------- Loss criteria + Optimization
    class_counts = Counter(train_y.numpy())
    total_samples = len(train_y.numpy())
    class_weight = {cls: total_samples / count for cls, count in class_counts.items()}
    class_weights_tensor = torch.tensor([class_weight[i] for i in sorted(class_weight.keys())], dtype=torch.float)
    
    # ********************************* Optimal Transport ********************************* 
    # -------------- Check Train and Test Performance of Trained Model from Non-adaption
    net.eval()
    
    print('Training dataset results')
    with torch.no_grad():
        train_pred = net(train_X)
    
    _, train_ydec_tmp = torch.max(train_pred, 1)
    confmat = ConfusionMatrix(task="multiclass", num_classes=3)
    print(confmat(torch.tensor(train_ydec_tmp), torch.tensor(train_y)))
    
    F1_score = ms.f1_score(train_y.tolist(), train_ydec_tmp.tolist(), average='macro')  
    accuracy = accuracy_score(np.array(train_y), np.array(train_ydec_tmp))   
    
    
    with torch.no_grad():
        test_pred = net(test_X)
        test_pred_probs = F.softmax(test_pred, dim=1)
    
    _, ydec_tmp = torch.max(test_pred, 1)
    confmat = ConfusionMatrix(task="multiclass", num_classes=3)
    print(confmat(torch.tensor(ydec_tmp), torch.tensor(test_y)))
    
    F1_score = ms.f1_score(test_y.tolist(), ydec_tmp.tolist(), average='macro')  
    accuracy = accuracy_score(np.array(test_y), np.array(ydec_tmp))   
    
    # -------------- Transport Cost Calulation
    
    n_classes = 3
    train_y_onehot = np.eye(n_classes)[train_y.detach().cpu().numpy()] 
    
    if solu_name == 'jdotM':
        # ----- MSE loss
        
        # function cost
        fcost = cdist(train_y_onehot, test_pred_probs.detach().cpu().numpy(), metric='sqeuclidean')
        
    elif solu_name == 'jdotH':
        # ----- hinge loss
         
        # Compute hinge loss between every source-target label pair
        test_logits = test_pred.detach().cpu().numpy()
        fcost = np.zeros((train_y_onehot.shape[0], test_logits.shape[0]))
        
        for i in range(train_y_onehot.shape[0]):
            for j in range(test_logits.shape[0]):
                # hinge = max(0, 1 - y * f(x))
                # here both y and f are vectors → use elementwise and square (to match JDOT style)
                loss_vector = np.maximum(0, 1 - train_y_onehot[i] * test_logits[j])
                fcost[i, j] = np.sum(loss_vector ** 2)
    
    C = alpha * C0 + fcost
    
    method='emd'
    reg = 1
    reset_model = True
    k=0
    changeLabels = False
    
    best_loss = float('inf')
    best_f1 = float('-inf')
    best_ot_f1 = float('-inf')
    patience = 10
    patience_counter = 0
    
    while (k < 50):# and not changeLabels:
        print('==============================================')
        k = k + 1
        if method=='sinkhorn':                 # fix f -> y_pred fixed -> C fixed -> calcuate G
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G = ot.emd(wa,wb,C)
            
        print('Output from optimal transport ---')
        
        Yst = ntest * G.T.dot(train_y_onehot)
        Yst = torch.tensor(Yst, dtype=torch.float32)
        
        _, ydec_tmp = torch.max(Yst, 1)
        
        confmat = ConfusionMatrix(task="multiclass", num_classes=3)
        print(confmat(torch.tensor(ydec_tmp), torch.tensor(test_y)))
        
        F1_score = ms.f1_score(test_y.tolist(), ydec_tmp.tolist(), average='macro')  
        accuracy = accuracy_score(np.array(test_y), np.array(ydec_tmp))   
        
        if F1_score > best_ot_f1:
            best_ot_f1 = F1_score
        
        if reset_model:
            net = ClassClassifier(train_X.shape[1], config['class_classifier_hidden_size_1'], config['class_classifier_dropout_rate_1'], config['class_classifier_hidden_size_2'], 3)
        
        test_copy_X = test_X.clone().detach()
        net = train(solu_name, net, test_copy_X, Yst, class_weights_tensor, config) # G fixed -> Yst(y_pred) pixed -> train f
        print(f"iter {k} trainig done --------")
        
        net.eval()
        print('Output from Classifer ---')
        
        if solu_name == 'jdothingL':
        # ----- MSE loss
            with torch.no_grad():
                test_pred = net(test_copy_X)
                test_pred_probs = F.softmax(test_pred, dim=1)
            # function cost
            fcost = cdist(train_y_onehot, test_pred_probs.detach().cpu().numpy(), metric='sqeuclidean')
        
        elif solu_name == 'jdotRealHing':
        # ----- hinge loss
            with torch.no_grad():
                test_pred = net(test_copy_X)  # logits, not softmax
                test_logits = test_pred.detach().cpu().numpy()  # shape: (n_target, n_classes)
                train_labels = train_y_onehot  # already in one-hot form, shape: (n_source, n_classes)    
            # Compute hinge loss between every source-target label pair
            fcost = np.zeros((train_labels.shape[0], test_logits.shape[0]))
            for i in range(train_labels.shape[0]):
                for j in range(test_logits.shape[0]):
                    # hinge = max(0, 1 - y * f(x))
                    # here both y and f are vectors → use elementwise and square (to match JDOT style)
                    loss_vector = np.maximum(0, 1 - train_labels[i] * test_logits[j])
                    fcost[i, j] = np.sum(loss_vector ** 2)
        
        C = alpha * C0 + fcost
     
        _, ydec_tmp = torch.max(test_pred, 1)
        
        confmat = ConfusionMatrix(task="multiclass", num_classes=3)
        print(confmat(torch.tensor(ydec_tmp), torch.tensor(test_y)))
        
        F1_score = ms.f1_score(test_y.tolist(), ydec_tmp.tolist(), average='macro')  
        accuracy = accuracy_score(np.array(test_y), np.array(ydec_tmp))   
        
        
        
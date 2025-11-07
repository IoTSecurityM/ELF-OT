# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.spatial.distance import cdist 

import ot

import sklearn.metrics as ms
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mal_dataset import MalConvProcDataset, MalwareImageProcDataset
from models import MalConv, FrozenXceptionClassifier
from withoutAdaptionXception import get_besthyperparams, no_adaption

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def classification(fea_name, config, best_model_path, checkpoint_dir=None):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    results_track = {'train_f1': 0, 'train_acc': 0, 'test_f1': 0, 'test_acc': 0, 'adaptedTest_f1': 0, 'adaptedTest_acc': 0, 'loss':[]}
      
    # ********************************* Dataset *********************************
    
    # ---------- Load Data
    if fea_name == "malconv":
        train_dt = MalConvProcDataset(config['arch_name'], config['train_year'], config['dt_index'])
        test_dt = MalConvProcDataset(config['arch_name'], config['test_year'], config['dt_index'])
    elif fea_name == "xception":
        train_dt = MalwareImageProcDataset(config['arch_name'], config['train_year'], config['dt_index'])
        test_dt = MalwareImageProcDataset(config['arch_name'], config['test_year'], config['dt_index'])
    
    train_loader = DataLoader(train_dt, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dt, batch_size=config['batch_size'], shuffle=True)
    
    # ---------- Initiate Models
    if fea_name == "malconv":
       net = MalConv().to(device)
    elif fea_name == "xception":
       net = FrozenXceptionClassifier(config['infer_fea']).to(device)
       
    # ---------- Loss criteria + Optimization
    class_weight = {0: 18/139, 1: 120/139, 2: 1/139}
    class_weights_tensor = torch.tensor([class_weight[i] for i in sorted(class_weight.keys())], dtype=torch.float).to(device)
    
    class_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    checkpoint_dir = best_model_path
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_dir"), map_location=device)
    
    net.load_state_dict(model_state)
    
    # -------------- Check Train and Test Performance of Trained Model from Non-adaption
    net.eval()
    
    print('Training dataset results')
    predicted_list, label_list = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, outputs = net(xb) 
            
            _, predicted = torch.max(outputs, 1)
            
            predicted_list.extend(predicted.tolist())
            label_list.extend(yb.tolist())                                     

    F1_score = ms.f1_score(np.array(label_list), np.array(predicted_list), average='macro')  
    accuracy = accuracy_score(np.array(label_list), np.array(predicted_list)) 
    
    results_track['train_f1'] = F1_score
    results_track['train_acc'] = accuracy
    
    print(f'f1: {F1_score}, accuracy: {accuracy}')
    
    print('Testing dataset results')
    predicted_list, label_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, outputs = net(xb) 
            
            _, predicted = torch.max(outputs, 1)
            
            predicted_list.extend(predicted.tolist())
            label_list.extend(yb.tolist())        
    
    F1_score = ms.f1_score(np.array(label_list), np.array(predicted_list), average='macro')  
    accuracy = accuracy_score(np.array(label_list), np.array(predicted_list)) 
    
    results_track['test_f1'] = F1_score
    results_track['test_acc'] = accuracy
   
    print(f'f1: {F1_score}, accuracy: {accuracy}')
    
    # -------------- Transport Cost Calulation
    alpha = 0.001
    lambda_t = 0.0001
    n_classes = 3
    best_model = 0
    
    # ---------- Optimal Transport Initialization
    for epoch in range(500):  # Adjust epochs
        
        # ---------- Training
        net.train()
        total_loss, epoch_step = 0, 0
        for (train_X, train_y), (test_X, _) in zip(train_loader, test_loader):
            
            train_X, train_y = train_X.to(device), train_y.to(device)
            test_X = test_X.to(device)
            
            train_feat, train_pred = net(train_X)
            test_feat, test_pred = net(test_X)
            
            test_prob = F.softmax(test_pred, dim=1)
            
            # --------------- Update Transport Plan
            
            # ----- Cost on Features
            
            train_feat_numpy = train_feat.detach().numpy()
            test_feat_numpy = test_feat.detach().numpy()
            
            wa, wb = ot.unif(len(train_feat_numpy)), ot.unif(len(test_feat_numpy))
            C0 = ot.dist(train_feat_numpy.reshape(len(train_feat_numpy), -1), test_feat_numpy.reshape(len(test_feat_numpy), -1))
            
            # ----- Cost on labels
            
            train_y_onehot_numpy = np.eye(n_classes)[train_y.detach().numpy()]
            test_prob_numpy = test_prob.detach().numpy()
            
            fcost = cdist(train_y_onehot_numpy, test_prob_numpy, metric='sqeuclidean')
                    
            # ----- Calculate the transport plan 
            
            C = alpha * C0 + lambda_t * fcost
            
            G = ot.emd(wa,wb,C)         
            
            # --------------- Update Model
            optimizer.zero_grad()
            
            loss_cls = class_criterion(train_pred, train_y)
            
            C0_model = torch.cdist(train_feat, test_feat, p=2)
            
            train_y_onehot = F.one_hot(train_y, num_classes=n_classes).float()
            fcost_model = torch.zeros(train_y_onehot.size(0), test_prob.size(0))
            
            for i in range(train_y_onehot.size(0)):
                for j in range(test_prob.size(0)):
                    # Cross-entropy cost: -y * log(p)
                    loss_vector = -train_y_onehot[i] * torch.log(torch.clamp(test_prob[j], min=1e-10))
                    fcost_model[i, j] = loss_vector.sum()
    
            loss_trans = (torch.tensor(G) * (alpha * C0_model + lambda_t * fcost_model)).sum()
            
            loss = loss_cls + loss_trans
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_step += 1
            
        results_track['loss'] = total_loss/epoch_step
            
        print(f"epoch {epoch} train average loss {total_loss/epoch_step}")  
        
    best_model.eval()
    predicted_list, label_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            _, outputs = best_model(xb) 
            
            _, predicted = torch.max(outputs, 1)
            
            predicted_list.extend(predicted.tolist())
            label_list.extend(yb.tolist())                                     

    F1_score = ms.f1_score(np.array(label_list), np.array(predicted_list), average='macro')  
    accuracy = accuracy_score(np.array(label_list), np.array(predicted_list))   
    
            
            
#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score

class MultiModalTrainer(object):
    def __init__(self, net, optimizer, lrsch, loss, train_loader, val_loader, logger, start_epoch,
                 save_interval=10, mode=2, Ldiff_weight=0.1):
        '''
        mode:   0: only single task--combine 
                1: multi task added, three losses are simply added together.
                2: Ldiff between two extracted features
        
        '''
        self.net = net
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.loss = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.logger.global_step = start_epoch
        self.save_interval = save_interval
        self.logger.add_graph(self.net)
        self.mode = mode
        self.Ldiff_weight = Ldiff_weight
            
    def train(self):
        self.net.train()
        self.logger.update_step()
        avg_loss = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, ascii=True, ncols=60)):
            inputs = [x.cuda() for x in inputs]
            targets = targets.cuda()
            pred_single, pred_combine = self.net(*inputs)
            self.optimizer.zero_grad()
            loss = self.loss(pred_combine, targets.view(-1))
            if self.mode >= 1: #multi-task added
                for pred in pred_single:
                    loss += self.loss(pred, targets.view(-1))
            if self.mode >= 2: #Ldiff added
                feats = []
                for idx, x in enumerate(inputs):
                    x = self.net.backbone[idx](x)
                    x = self.net.avgpool(x)
                    feats.append(x)
                ##TODO: generalize to multi-modal
                mul = feats[0] * feats[1]
                Ldiff = torch.abs(mul.sum(dim=1)) / (feats[0].norm(dim=1) * feats[1].norm(dim=1))
                Ldiff = Ldiff.sum()
                loss += self.Ldiff_weight * Ldiff
            
            loss.backward()
            self.optimizer.step()
            avg_loss.append(loss.item())
        avg_loss = sum(avg_loss) / (len(avg_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
            self.lrsch.step(avg_loss)
        else:
            self.lrsch.step()
        if not (self.logger.global_step % self.save_interval):
            self.logger.save(self.net, self.optimizer, self.lrsch, self.loss)

    def test(self):
        self.net.eval()
        pred_list_single = []
        prob_list_single = []
        pred_list_combine = []
        prob_list_combine = []
        target_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.val_loader, ascii=True, ncols=60)):
                target_list.append(targets.detach().numpy().ravel())
                inputs = [x.cuda() for x in inputs]
                targets = targets.cuda()
                self.optimizer.zero_grad()
                pred_single, pred_combine = self.net(*inputs)
                pred_list_single.append([pred_single[i].argmax(1).cpu().detach().numpy().ravel() for i in range(len(pred_single))])
                prob_list_single.append([pred_single[i][:,1].cpu().detach().numpy().ravel() for i in range(len(pred_single))])
                pred_list_combine.append(pred_combine.argmax(1).cpu().detach().numpy().ravel())
                prob_list_combine.append(pred_combine[:,1].cpu().detach().numpy().ravel())

        self.log_metric("grey", [x[0] for x in pred_list_single], [x[0] for x in prob_list_single], target_list)
        self.log_metric("SWE", [x[1] for x in pred_list_single], [x[1] for x in prob_list_single], target_list)
        self.log_metric("combine", pred_list_combine, prob_list_combine, target_list)

    def log_metric(self, prefix, pred, prob, target):
        pred_list = np.concatenate(pred)
        prob_list = np.concatenate(prob)
        target_list = np.concatenate(target)
        cls_report = classification_report(target_list, pred_list, output_dict=True)
        acc = accuracy_score(target_list, pred_list)
        auc_score = roc_auc_score(target_list, prob_list)
        self.logger.log_scalar(prefix+'/'+'Accuracy', acc, print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)




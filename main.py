#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from data.BMDataset import BMDataset
from data.PatientBags import PatientBags
from data.ss_bag import SSBags
from data.bag import BMBags
from data.ruijin import RuijinBags
from data.DirectBags import DirectBags
from data.PatientBags import PatientBags
from data.PatientBags_old import PatientBags_old
from models.attentionMIL import Attention, GatedAttention,H_Attention, S_H_Attention, S_H_Attention2, Res_Attention
from models.graph_attention import H_Attention_Graph, H_Attention_GraphV2
from trainers.MILTrainer import MILTrainer
from trainers.BaseTrainer import BaseTrainer
from utils.logger import Logger
from torch.optim import Adam
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os

parser = argparse.ArgumentParser(description='Ultrasound CV Framework')
parser.add_argument('--config',type=str,default='grey_SWE_gcn')

class base_config(object):
    def __init__(self,log_root,args):
        self.net = getattr(import_module('models.attentionMIL'),args.net)()
        print(self.net)
        self.net = self.net.cuda()
        self.train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])

        self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)

        self.trainbag = PatientBags_old(args.data_root, pre_transform=self.train_transform, crop_mode=False,
                    sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold])
        self.testbag = PatientBags_old(args.data_root, pre_transform=self.test_transform, crop_mode=False,sub_list = [args.test_fold])
        self.train_loader = DataLoader(self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = BaseTrainer(self.net, self.optimizer, self.lrsch, None, self.train_loader, self.val_loader, self.logger, 0)

        self.save_config(args)
    
    def save_config(self,args):
        config_file = './saved_configs/'+args.config_name+'.txt'
        f = open(config_file,'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])


class graph_config(base_config):
    def __init__(self, log_root, args):
        self.net = getattr(import_module('models.graph_attention'),args.net)()
        print(self.net)
        self.net = self.net.cuda()
        self.train_transform = transforms.Compose([
                    transforms.Resize((112,112)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
                    transforms.Resize((112,112)),
                    transforms.ToTensor()
        ])
        self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        self.trainbag = DirectBags(args.data_root, pre_transform=self.train_transform, 
                    sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold])
        self.testbag = DirectBags(args.data_root, pre_transform=self.test_transform, sub_list = [args.test_fold])
        self.train_loader = DataLoader(self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = MILTrainer(self.net, self.optimizer, self.lrsch, None, self.train_loader, self.val_loader, self.logger, 0)
        self.save_config(args)


if __name__=='__main__':
    #configs = getattr(import_module('configs.'+args.config),'Config')()
    #configs = configs.__dict__

    # update configss from arg parse
    parser.add_argument('--data_root',type=str,default='/media/hhy/data/USdata/MergePhase1/5folds_ALNM')
    parser.add_argument('--log_root',type=str)
    parser.add_argument('--test_fold',type=int,default=0)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--resume',type=int,default=-1)
    parser.add_argument('--batchsize',type=int,default=1)
    parser.add_argument('--net',type=str,default='H_Attention_Graph')
    parser.add_argument('--config_name',type=str)

    # parse parameters
    args = parser.parse_args()
    log_root = os.path.join('/media/hhy/data/gcn_results',args.log_root)
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    # choose model
    base_nets = ['Attention','Res_Attention']
    graph_nets = ['H_Attention_Graph']

    if args.net in base_nets:
        config_object = base_config(log_root,args)
    if args.net in graph_nets:
        config_object = graph_config(log_root,args)
    
    

    # train and eval
    for epoch in range(config_object.logger.global_step, args.epoch):
        print('Now epoch {}'.format(epoch))
        config_object.trainer.train()
        config_object.trainer.test()
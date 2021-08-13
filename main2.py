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
    def __init__(self,log_root,net,args):
        self.net = getattr(import_module('models.attentionMIL'),net)()
        print(self.net)
        self.net = self.cuda()
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
        self.trainer = BaseTrainer(self.net, self.optimizer, self.lrsch, None, self.train_loader, self.val_loader, logger, 0)



if __name__=='__main__':
    #configs = getattr(import_module('configs.'+args.config),'Config')()
    #configs = configs.__dict__

    # update configss from arg parse
    parser.add_argument('--data_root',type=str,default='/media/hhy/data/USdata/MergePhase1/5folds_BM')
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
    net = args.net

    if (net == 'H_Attention_Graph') or (net =='H_Attention_GraphV2'):
        net = getattr(import_module('models.graph_attention'),net)()
    else:
        net = getattr(import_module('models.attentionMIL'),net)()
    '''
    net = getattr(import_module('models.attentionMIL'),net)()
    '''

    '''
    print(net)
    net = net.cuda()

    optimizer = Adam(net.parameters(), lr=args.lr)
    lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
    logger = Logger(log_root)
    train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])

    trainbag = DirectBags(args.data_root, pre_transform=train_transform, 
                    sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold])
    testbag = DirectBags(args.data_root, pre_transform=test_transform, sub_list = [args.test_fold])

    trainbag = PatientBags_old(args.data_root, pre_transform=train_transform, crop_mode=False,
                    sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold])
    testbag = PatientBags_old(args.data_root, pre_transform=test_transform, crop_mode=False,sub_list = [args.test_fold])


    train_loader = DataLoader(trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
    val_loader = DataLoader(testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
    trainer = BaseTrainer(net, optimizer, lrsch, None, train_loader, val_loader, logger, 0)
    config_name = args.config_name
    '''
    ## save file
    config_file = './saved_configs/'+config_name+'.txt'
    f = open(config_file,'a+')
    argDict = args.__dict__
    for arg_key, arg_value in argDict.items():
        f.writelines(arg_key+':'+str(arg_value)+'\n')
    f.close()
    logger.auto_backup('./')
    logger.backup_files([config_file])

    # train and eval
    for epoch in range(logger.global_step, args.epoch):
        print('Now epoch {}'.format(epoch))
        trainer.train()
        trainer.test()
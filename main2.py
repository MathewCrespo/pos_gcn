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
from models.attentionMIL import Attention, GatedAttention,H_Attention, S_H_Attention, S_H_Attention2
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

if __name__=='__main__':
    #configs = getattr(import_module('configs.'+args.config),'Config')()
    #configs = configs.__dict__

    # update configss from arg parse
    parser.add_argument('--data_root',type=str,default='/remote-home/gyf/Ultrasound_CV/data/MergePhase1/5folds_ALNM')
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
    log_root = os.path.join('/remote-home/gyf/hhy/Ultrasound_MIL/gcn_pos/data_results',args.log_root)
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
    print(net)
    net = net.cuda()

    optimizer = Adam(net.parameters(), lr=args.lr)
    lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
    logger = Logger(log_root)
    train_transform = transforms.Compose([
                    transforms.Resize((28,28)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
                    transforms.Resize((28,28)),
                    transforms.ToTensor()
        ])
    
    trainbag = DirectBags(args.data_root, pre_transform=train_transform, 
                    sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold])
    testbag = DirectBags(args.data_root, pre_transform=test_transform, sub_list = [args.test_fold])
    '''
    trainbag = PatientBags_old(args.data_root, pre_transform=train_transform, crop_mode=False,
                    sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold])
    testbag = PatientBags_old(args.data_root, pre_transform=test_transform, crop_mode=False,sub_list = [args.test_fold])
    '''

    train_loader = DataLoader(trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
    val_loader = DataLoader(testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
    trainer = MILTrainer(net, optimizer, lrsch, None, train_loader, val_loader, logger, 0)
    config_name = args.config_name

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
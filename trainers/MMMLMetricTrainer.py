#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os,sys
sys.path.append('../')
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utility as utility
from utils.utility import AverageMeter, LossTracker
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import utils.common_functions as c_f

class MMMLMetricTrainer(object):
    """
    Multi Modal Multi Label Metric Learning Trainer with 
    classification loss.

    Notes:
        1. Each modality has it own backbone and a shared backbone is optional
            (which input is merger_pre(grey, swe), could be simply cat together, or
            just do nothing)

            grey -> backbone_grey -> feat_grey;
            merger_pre(grey, pure) -> backbone_shared -> feat_share; (optional)
            pure -> backbone_pure -> feat_pure;

        2. For feature merging: Basically, 
            a merger could be easily designed as a function of torch.cat()
        
        3. Classifier: support multi-label classification; support loss between
            logits.
        
        4. Models, optimizers, lrschs, losses, loss_weights are designed as
            dictionaries!
        
        5. Metrics: classification metrics + retrieval top-k recall/specificity.

        6. Data structures:
            data: {"grey": [N,3,H,W], "pure": [N,3,H,W]}
            labels: {"cls": [N], "metric": [N]}

    Args:
        models: {"backbones", "classifiers", "embedders", "mergers",}
            if "embedders" is None, feature is output of backbone
        optimizers: {"backbones", "classifiers", "embedders"}
        lrschs: {"backbone", "classifier", "embedder"}
        losses: {"cls": "metric":, "feats", "logits"}
        loss_weights: the key should be corresponding to losses
    """
    def __init__(self,
                 models,
                 optimizers,
                 losses,
                 lrschs=None,
                 loss_weights=None,
                 train_data=None,
                 test_data=None,
                 mining_funcs=None,
                 gradient_clippers=None,
                 save_interval=5,
                 logger=None,
                 end_of_iteration_hook=None,
                 end_of_epoch_hook=None,
                 process_labels=None,
                 batch_size=64,
                 dataloader_num_workers=8,
                 sampler=None,
                 collate_fn=None,
                 data_device="cuda",
                 cls_tasks=["BM", "ALNM"],
                 cls_thresholds={"BM":0.5, "ALNM":0.1},
                 ):
        self.models = models
        self.optimizers = optimizers
        self.lrschs = lrschs
        self.losses = losses # loss functions
        self.loss_weights = loss_weights
        self.train_data = train_data
        self.test_data = test_data
        self.mining_funcs = mining_funcs
        self.gradient_clippers = gradient_clippers
        self.save_interval = save_interval
        self.logger = logger
        self.data_device = data_device
        self.end_of_iteration_hook = end_of_iteration_hook
        self.end_of_epoch_hook = end_of_epoch_hook
        self.process_labels = process_labels
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.loss_names = list(self.losses.keys())
        self.epoch = 0
        self.global_iter = 0
        self.inner_iter = 0
        self.cls_tasks = cls_tasks
        self.cls_thresholds = cls_thresholds
        ## initialization
        self.custom_setup()
        self.initialize_loaders()
        self.initialize_models()
        self.initialize_loss_tracker()
        self.initialize_loss_weights()
        self.initialize_hooks()
        self.initialize_lr_schedulers()
        self.show_recall_k = [1,2,4,8,16]
        self.initialize_data_device()

    def custom_setup(self):
        self.feat_used_for_mining = "feat_gp"

    def train(self):
        self.set_to_train()
        self.epoch += 1
        self.reset_meters()
        self.reset_banks()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, labels) in enumerate(tbar):
            data, labels = self.maybe_do_batch_mining(data, labels)
            output_dict = self.forward_and_backward(data, labels)
            self.describe_pbar(tbar)
            self.collect_outputs(output_dict, labels)
            self.end_of_iteration_hook(self)
            self.global_iter += 1
            self.inner_iter += 1
        
        self.end_of_epoch_hook(self)
        self.inner_iter = 0
        self.step_lr_schedulers()
        self.merge_outputs()
        self.evaluate()
        self.log_losses("train")
        self.log_metrics("train")
        self.save()
            
    def test(self):
        self.set_to_eval()
        self.reset_meters()
        self.reset_banks()
        tbar = tqdm(self.test_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tbar):
                output_dict = self.forward(data)
                self.collect_outputs(output_dict, labels)

        self.merge_outputs()
        self.evaluate()
        self.log_metrics("test")
        self.save_results("test")

    def describe_pbar(self, pbar):
        pbar.set_description('TRAIN ({}) | Loss: {:.3f} | Cls Loss: {:.3f} | Metric Loss: {:.3f} | Logit Loss: {:.3f} | Feat Loss: {:.3f}'.format(
                self.epoch, self.loss_meters["total_loss"].average,
                self.loss_meters["cls_loss"].average if "cls_loss" in self.loss_meters else 0,
                self.loss_meters["metric_loss"].average if "metric_loss" in self.loss_meters else 0,
                self.loss_meters["logit_loss"].average if "logit_loss" in self.loss_meters else 0,
                self.loss_meters["feat_loss"].average if "feat_loss" in self.loss_meters else 0))
    
    def save(self):
        """
        save networks, optimizers, criterions, lrschs
        """
        save_dir = os.path.join(self.logger.logdir, "ckpt")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_dict = {"models": {}, "optimizers":{}, "losses":{}, "lrschs": {}}
        for k in self.models.keys():
            save_dict['models'][k] = self.models[k].state_dict()
        
        for k in self.optimizers.keys():
            save_dict["optimizers"][k] = self.optimizers[k].state_dict()
        
        for k in self.losses.keys():
            save_dict["losses"][k] = self.losses[k].state_dict()
        
        for k in self.lrschs.keys():
            save_dict["lrschs"][k] = self.lrschs[k].state_dict()
        
        torch.save(save_dict, os.path.join(save_dir, "ckpt_{}.pth".format(self.epoch)))

    def save_results(self, prefix="test"):
        """
        save embeddings and labels
        """
        save_dir = os.path.join(self.logger.logdir, "res_{}".format(prefix))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(self.embedding_banks, os.path.join(save_dir, "outputs_{}.pth".format(self.epoch)))
        torch.save(self.label_banks, os.path.join(save_dir, "labels_{}.pth".format(self.epoch)))

    def merge_outputs(self):
        
        for k in self.embedding_banks.keys():
            self.embedding_banks[k] = torch.cat(self.embedding_banks[k], dim=0)
        
        for k in self.label_banks.keys():
            self.label_banks[k] = torch.cat(self.label_banks[k], dim=0)

    def collect_outputs(self, output_dict, labels):
        """
        output_dict: dict of tensor
        labels: dict of tensor
        """
        for k, v in output_dict.items():
            if k not in self.embedding_banks:
                self.embedding_banks[k] = []
            self.embedding_banks[k].append(v.detach().cpu())
        
        for k, v in labels.items():
            if k not in self.label_banks:
                self.label_banks[k] = []
            self.label_banks[k].append(v.detach().cpu())

    def evaluate(self):
        self.cls_evaluate()
        self.do_knn_and_evaluate()

    def cls_evaluate(self):
        if "cls" in self.label_banks:
            labels = self.label_banks["cls"]
            ## Multi-Label / Single-Label
            if labels.dim() == 2:
                if labels.shape[-1] > 1:
                    for i in range(labels.shape[-1]):
                        label = labels[:, i].view(-1)
                        for name, v in self.embedding_banks.items():
                            if name.startswith("logit"):
                                self.cls_meters.update(self.cls_report(v.sigmoid()[:, i], label, "{}/{}".format(self.cls_tasks[i],name.split("_")[-1]),
                                self.cls_thresholds[self.cls_tasks[i]]))
                else:
                    label = labels.view(-1)
                    for name, v in self.embedding_banks.items():
                        if name.startswith("logit"):
                            i = 0
                            self.cls_meters.update(self.cls_report(v.sigmoid()[:, i], label, "{}/{}".format(self.cls_tasks[i],name.split("_")[-1]),
                            self.cls_thresholds[self.cls_tasks[i]]))
            else:
                label = labels.view(-1)
                for name, v in self.embedding_banks.items():
                    if name.startswith("logit"):
                        i = 0
                        self.cls_meters.update(self.cls_report(v.sigmoid()[:, i], label, "{}/{}".format(self.cls_tasks[i],name.split("_")[-1]),
                        self.cls_thresholds[self.cls_tasks[i]]))

    def do_knn_and_evaluate(self):
        """
        Simply emplemented retrieval evaluation.
        Notes:
            1. Based on dot similarity (make sure embeddings have been correctly normalized)
            2. KNN retrieval, Recall@K returned.
            3. KNN classification (TODO)
        """
        if "metric" in self.label_banks:
            label = self.label_banks["metric"]

            for name, v in self.embedding_banks.items():
                if name.startswith("feat"):
                    embeddings = v
                    idxs  = c_f.knn_dot_self(embeddings, max(self.show_recall_k), False)
                    retrieved_results = label[idxs.view(-1)].view(-1, max(self.show_recall_k))
                    for k in self.show_recall_k:
                        correct_num = sum([1 for (result, l) in zip(retrieved_results, label) if l in result[:k]])
                        recall = correct_num / len(label)
                        self.metric_meters["{}_Recall@{}".format(name, k)] = recall
        
    def maybe_mine_embeddings(self, embeddings, labels):
        if "tuple_miner" in self.mining_funcs:
            return self.mining_funcs["tuple_miner"](embeddings, labels)
        return None

    def maybe_do_batch_mining(self, data, labels):
        if self.mining_funcs is not None:
            if "subset_batch_miner" in self.mining_funcs:
                with torch.no_grad():
                    self.set_to_eval()
                    embeddings = self.compute_embeddings(data)
                    idx = self.mining_funcs["subset_batch_miner"](embeddings, labels["metric"])
                    self.set_to_train()
                    for k in data.keys():
                        data[k] = data[k][idx]
                    for k in labels.keys():
                        labels[k] = labels[k][idx]
        return data, labels

    def clip_gradients(self):
        if self.gradient_clippers is not None:
            for v in self.gradient_clippers.values():
                v()

    def compute_embeddings(self, data):
        output_dict = self.forward(data)
        embeddings = output_dict[self.feat_used_for_mining]
        return embeddings

    def log_losses(self, prefix="train"):
        for k, v in self.loss_meters.items():
            self.logger.log_scalar(prefix+"/"+k, v.average, self.epoch, False)


    def log_metrics(self, prefix="train"):
        for k, v in self.cls_meters.items():
            self.logger.log_scalar(prefix+"/"+k, v, self.epoch, False)
        
        for k, v in self.metric_meters.items():
            self.logger.log_scalar(prefix+"/"+k, v, self.epoch, False)

    def cls_report(self, y_pred, y_true, prefix, cls_threshold=0.5):
        """
        A combination of sklearn.metrics function and our logger (use tensorboard)
        """
        ##make hard prediction
        y_pred_hard = (y_pred>cls_threshold).float()
        cls_report = classification_report(y_true, y_pred_hard, output_dict=True)
        acc = accuracy_score(y_true, y_pred_hard)
        auc_score = roc_auc_score(y_true, y_pred)
        return_dict = {}
        return_dict[prefix+"/"+"ACC"] = acc
        return_dict[prefix+"/"+"AUC"] = auc_score
        return_dict[prefix+"/"+"Recall"] = cls_report['1.0']['recall']
        return_dict[prefix+"/"+"Specificity"] = cls_report['0.0']['recall']
        return return_dict

    def forward(self, data):
        """
        Do a complete forward pass through the whole architecture and returns 
        as dictionaries.
        output_dict:
            {
                "feat_grey", "logit_grey",
                "feat_pure", "logit_pure",
                "feat_shared", "logit_shared",
                "feat_gp", "logit_gp",
                "feat_gps", "logit_gps",
            }
        """
        output_dict = {}
        ## Basic
        if "grey" in data:
            f_grey = self.models["backbone_grey"](data["grey"].to(self.data_device))
            logit_grey = self.models["classifier_grey"](f_grey)
            feat_grey = self.models["embedder_grey"](f_grey)
            output_dict["logit_grey"] = logit_grey
            output_dict["feat_grey"]  = feat_grey

        if "pure" in data:
            f_pure = self.models["backbone_pure"](data["pure"].to(self.data_device))
            logit_pure = self.models["classifier_pure"](f_pure)
            feat_pure = self.models["embedder_pure"](f_pure)
            output_dict["logit_pure"] = logit_pure
            output_dict["feat_pure"] = feat_pure
        
        ## Dual Modal
        if "grey" in data and "pure" in data:
            dual_input = self.models["merger_pre"](data["pure"].to(self.data_device), data["grey"].to(self.data_device))
            if "backbone_shared" in self.models:
                f_shared = self.models["backbone_shared"](dual_input)
                feat_shared = self.models["embedder_shared"](f_shared)
                output_dict["feat_shared"] = feat_shared
                if "classifier_shared" in self.models:
                    logit_shared = self.models["classifier_shared"](feat_shared)
                    output_dict["logit_shared"] = logit_shared
                
                f_gps = self.models["merger_gps"](f_grey,f_pure,f_shared)
                feat_gps = self.models["embedder_gps"](f_gps)
                if "classifier_gps" in self.models:
                    logit_gps = self.models["classifier_gps"](f_gps)
                    output_dict["logit_gps"] = logit_gps
                output_dict["feat_gps"] = feat_gps
            
            f_gp = self.models["merger_gp"](f_grey, f_pure)
            feat_gp = self.models["embedder_gp"](f_gp)
            output_dict["feat_gp"] = feat_gp
            if "classifier_gp" in self.models:
                logit_gp = self.models["classifier_gp"](f_gp)
                output_dict["logit_gp"] = logit_gp

        return output_dict

    def backward(self):
        self.loss_bank["total_loss"].backward()

    def forward_and_backward(self, data, labels):
        """
        labels: (dict) {"cls", "retrieval"}
        """
        self.zero_losses()
        self.zero_grad()
        self.update_loss_weights()
        output_dict = self.calculate_loss(data, labels)
        self.loss_tracker.update(self.loss_weights)
        for name in self.loss_names:
            self.loss_meters[name].update(self.loss_bank[name].item())
        self.backward()
        self.step_optimizers()

        return output_dict

    def update_loss_weights(self):
        pass

    def calculate_loss(self, data, labels):
        """
        data: dict
        labels: dict or torch.tensor
        """
        output_dict = self.forward(data)
        if "cls" in labels:
            self.get_cls_losses(output_dict, labels["cls"].to(self.data_device))
        if "metric" in labels:
            self.get_metric_losses(output_dict, labels["metric"].to(self.data_device))
        
        ## loss on multi-modal features (Ldiff)
        self.get_feats_losses(output_dict)
        ## loss on logits (IH Loss)
        self.get_logits_losses(output_dict)

        return output_dict

    def get_cls_losses(self, output_dict, labels):
        if "cls_loss" in self.loss_names:
            self.loss_bank["cls_loss"] = 0
            if "logit_grey" in output_dict:
                self.loss_bank["cls_loss"] += self.losses["cls_loss"](output_dict["logit_grey"], labels.view(output_dict["logit_grey"].shape))
            if "logit_pure" in output_dict:
                self.loss_bank["cls_loss"] += self.losses["cls_loss"](output_dict["logit_pure"], labels.view(output_dict["logit_pure"].shape))
            if "logit_gp" in output_dict:
                self.loss_bank["cls_loss"] += self.losses["cls_loss"](output_dict["logit_gp"], labels.view(output_dict["logit_gp"].shape))
            if "logit_shared" in output_dict:
                self.loss_bank["cls_loss"] += self.losses["cls_loss"](output_dict["logit_shared"], labels.view(output_dict["logit_shared"].shape))
            if "logit_gps" in output_dict:
                self.loss_bank["cls_loss"] += self.losses["cls_loss"](output_dict["logit_gps"], labels.view(output_dict["logit_gps"].shape))
        else:
            pass

    def get_metric_losses(self, output_dict, labels):
        if "metric_loss" in self.loss_names:
            self.loss_bank["metric_loss"] = 0
            if "feat_grey" in output_dict:
                self.loss_bank["metric_loss"] += self.losses["metric_loss"](output_dict["feat_grey"], labels)
            if "feat_pure" in output_dict:
                self.loss_bank["metric_loss"] += self.losses["metric_loss"](output_dict["feat_pure"], labels)
            if "feat_gp" in output_dict:
                self.loss_bank["metric_loss"] += self.losses["metric_loss"](output_dict["feat_gp"], labels)
            if "feat_shared" in output_dict:
                self.loss_bank["metric_loss"] += self.losses["metric_loss"](output_dict["feat_shared"], labels)
            if "feat_gps" in output_dict:
                self.loss_bank["metric_loss"] += self.losses["metric_loss"](output_dict["feat_gps"], labels)
        else:
            pass

    def get_feats_losses(self, output_dict):
        if "feat_loss" in self.loss_names:
            self.loss_bank["feat_loss"] = self.losses["feat_loss"](output_dict["feat_grey"], output_dict["feat_pure"])
        else:
            pass

    def get_logits_losses(self, output_dict):
        if "logit_loss" in self.loss_names:
            self.loss_bank["logit_loss"] = 0
            if "logit_grey" in output_dict:
                self.loss_bank["logit_loss"] += self.losses["logit_loss"](output_dict["logit_grey"])
            if "logit_pure" in output_dict:
                self.loss_bank["logit_loss"] += self.losses["logit_loss"](output_dict["logit_pure"])
            if "logit_gp" in output_dict:
                self.loss_bank["logit_loss"] += self.losses["logit_loss"](output_dict["logit_gp"])
            if "logit_shared" in output_dict:
                self.loss_bank["logit_loss"] += self.losses["logit_loss"](output_dict["logit_shared"])
            if "logit_gps" in output_dict:
                self.loss_bank["logit_loss"] += self.losses["logit_loss"](output_dict["logit_gps"])
        else:
            pass

    def zero_losses(self):
        for k in self.loss_bank.keys():
            self.loss_bank[k] = 0

    def zero_grad(self):
        for v in self.models.values():
            v.zero_grad()
        for v in self.optimizers.values():
            v.zero_grad()

    def step_lr_schedulers(self):
        if self.lrschs is not None:
            for v in self.lrschs.values():
                v.step()

    def step_optimizers(self):
        for v in self.optimizers.values():
            v.step()

    def initialize_loaders(self):        
        self.train_loader = c_f.get_train_dataloader(
            self.train_data,
            self.batch_size,
            self.sampler,
            self.dataloader_num_workers,
            self.collate_fn,
        )
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_num_workers)

    def reset_meters(self):
        self.cls_meters = {}
        self.metric_meters = {}
        self.loss_meters = {}
        for name in self.loss_names:
            self.loss_meters[name] = AverageMeter()
    
    def reset_banks(self):
        self.label_banks = {}
        self.embedding_banks = {}

    def initialize_loss_tracker(self):
        self.loss_tracker = utility.LossTracker(self.loss_names)
        self.loss_bank = self.loss_tracker.losses
     
        
    def initialize_data_device(self):
        if self.data_device is None:
            self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_loss_weights(self):
        if self.loss_weights is None:
            self.loss_weights = {k: 1 for k in self.loss_names}

    def initialize_hooks(self):
        if self.end_of_iteration_hook is None:
            self.end_of_iteration_hook = c_f.return_input
        if self.end_of_epoch_hook is None:
            self.end_of_epoch_hook = c_f.return_input

    def initialize_lr_schedulers(self):
        if self.lrschs is None:
            self.lrschs = {}

    def initialize_models(self):
        if "embedder_grey" not in self.models:
            self.models["embedder_grey"] = c_f.Normalize()
        if "embedder_pure" not in self.models:
            self.models["embedder_pure"] = c_f.Normalize()
        if "embedder_gp" not in self.models:
            self.models["embedder_gp"] = c_f.Normalize()
        if "embedder_shared" not in self.models:
            self.models["embedder_shared"] = c_f.Normalize()
        if "embedder_gps" not in self.models:
            self.models["embedder_gps"] = c_f.Normalize()

        if "merger_pre" not in self.models:
            self.models["merger_pre"] = c_f.CatMerger()
        if "merger_gp" not in self.models:
            self.models["merger_gp"] = c_f.CatMerger()
        if "merger_gps" not in self.models:
            self.models["merger_gps"] = c_f.CatMerger()

        for k in self.models.keys():
            self.models[k].to(self.data_device)

    def set_to_train(self):
        for v in self.models.values():
            v.train()

    def set_to_eval(self):
        for v in self.models.values():
            v.eval()


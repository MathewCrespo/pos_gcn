from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random
import pandas as pd
from random import randint,sample


class RuijinBags(Dataset):
    def __init__(self, root, sub_list, pre_transform, crop_mode=False, mix_mode=0,
                 label_name="手术淋巴结情况（0未转移；1转移）"):
        self.root = root
        self.label_name = label_name
        self.sub_list = sub_list
        self.pre_transform = pre_transform
        self.patient_info = [] # each patient is a dict {'ID':ID,'label','image_path':, 'fold':}
        self.table = pd.read_excel('/remote-home/my/Ultrasound_CV/data/Ruijin/phase1/data.xlsx')
        # create patient_info based on 5 fold
        for fold in self.sub_list:
            self.scan(fold)

    def scan(self,fold):  # to create entire patient info (list)
        fold_table = self.table[self.table['fold']==fold].reset_index(drop=True)
        for k in range(len(fold_table)):
            ID = fold_table.loc[k,'ID']
            #print(ID)
            img_path = os.path.join(self.root, ID)
            imgs = os.listdir(img_path)
            label = fold_table.loc[k, self.label_name]
            label = label -1 if self.label_name== "pLN分组3（1为0-2枚淋巴结转移；2为＞2枚淋巴结转移）" else label
            now_patient={
                'ID':ID,
                'imgs':[img_path + '/'+ img for img in imgs],
                'label':label

            }
            self.patient_info.append(now_patient)
            
    def __getitem__(self,idx):
        now_patient = self.patient_info[idx]
        label = now_patient['label']
        imgs = []
        for img_path in now_patient['imgs']:
            img = Image.open(img_path).convert('RGB')
            if self.pre_transform is not None:
                img = self.pre_transform(img)
            imgs.append(img)
        
        return torch.stack([x for x in imgs],dim=0),label
    
    def __len__(self):
        return len(self.patient_info)


if __name__ == '__main__':
    root = '/remote-home/my/Ultrasound_CV/data/Ruijin/clean'
    pre_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
    all_set = RuijinBags(root, [0,1,2,3,4],pre_transform,label_name='手术淋巴结情况（0未转移；1转移）')
    
    label_set = []
    for i in range(len(all_set)):
        label = all_set[i][1]
        label_set.append(label)
    print(label_set.count(1))
    print(label_set)


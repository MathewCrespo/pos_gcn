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


class non_tumorBag(Dataset):
    def __init__(self, root, sub_list, pre_transform, crop_mode=False):
        self.root = root
        self.sub_list = sub_list
        self.pre_transform = pre_transform
        self.patient_info = [] # each patient is a dict {'ID':ID,'label','image_path':, 'fold':}
        # create patient_info based on 5 fold
        for fold in self.sub_list:
            self.scan(fold)
        #print(len(self.patient_info))

    def scan(self,fold):  # to create entire patient info (list)
        b_path = os.path.join(self.root,str(fold)+'/benign')
        m_path = os.path.join(self.root,str(fold)+'/malignant')
        for path in [b_path,m_path]:
            if path == b_path:
                label = 0.0
            else:
                label = 1.0
            patients = os.listdir(path)
            for p in patients:
                patient_path = os.path.join(path,p)
                imgs = os.listdir(patient_path+'/clean')
                now_patient={
                'imgs':[patient_path + '/clean/'+ img for img in imgs],
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
    root = '/remote-home/share/first_hos/non_tumor/5folds'
    pre_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
    all_set = non_tumorBag(root, [0,1,2,3,4],pre_transform)
    img = all_set[0][0]
    label = all_set[0][1]
    print(img.shape)
    print(label)
    '''
    label_set = []
    for i in range(len(all_set)):
        label = all_set[i][1]
        label_set.append(label)
    print(label_set.count(1))
    print(label_set)
    '''


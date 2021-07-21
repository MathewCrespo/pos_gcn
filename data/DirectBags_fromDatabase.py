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
import cv2
import selectivesearch

'''
DirectBags -- do selective search in advance
Two tasks available: Benign and Malignant classifcation and axillary lymph node metastasis (ALNM)
'''

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def is_xml_file(filename):
    return filename.endswith('xml')

class DirectBags_fromDatabase(Dataset):
    '''
    This Dataset is for doing benign/maglignant classification with US or SWE or two modal together

    Args:
        root: (str) data root
        sub_list: (int) sub path list [0, 1, 2, 3, 4] indicating which subset to use
        pre_transform: (torchvision.transform) the transform used
        modality: (int) indicate which modality to use. 0: US 1: SWE 2: US & SWE
    '''
    def __init__(self, pre_transform=None, sub_list = [0], task = 'BM'):
        self.img_list = [] ## a list of tuple (grey_filename, swe_filename)
        self.pre_transform = pre_transform
        self.patient_dict = [] ## a dict of information: {name: {label:, images: {grey:, pure:}}} 
        self.label_list = []
        self.metric_label_list = []
        self.ALNM_num = 1
        self.num_pbag = 0
        self.sub_list = sub_list
        self.task = task
        for i in self.sub_list:
            tmp = database.get('patient_dict{}'.format(str(i)))
            tmp = pickle.loads(tmp)
            self.patient_dict.extend(tmp)


    def __getitem__(self, idx):  # for a single patient
        ### start from patient_dict and create a dict
        #print(idx)
        now_patient = self.patient_dict[idx]
        if self.task == 'BM':
            label = now_patient['label'][0]
        if self.task == 'ALNM':
            label = now_patient['label'][1]
        grey_img_path = now_patient['images']['grey']
        grey_imgs = []
        img_info = []
        idx_list = [0]
        idx_temp = 0
        for path in grey_img_path:   # different ultrasound image in a single patient
            img_path = path.split('.')[0]
            ins_list = os.listdir(img_path)
            lesion_list = []
            pos_list = []
            region_num = 0
            img_info_path = os.path.join(img_path,'original.txt')
            img_info_data = database.get(img_info_path)
            img_w, img_h = pickle.loads(img_info_data)            
            for ins in ins_list:     # different lesion area in a single ultrasound image
                if is_image_file(ins):
                    region_num  += 1
                    ins_info = ins.split('.')[0]+'.txt'
                    ins_path = os.path.join(img_path,ins)
                    log_path = os.path.join(img_path,ins_info)
                    # read image
                    ins_img_data = database.get(ins_path)
                    ins_img = pickle.loads(ins_img_data)
                    ins_img = self.pre_transform(ins_img)
                    lesion_list.append(ins_img)
                    # parse bbox info
                    ins_info_data = database.get(log_path)
                    x,y,w,h = pickle.loads(ins_info_data)
                    pos_info = torch.Tensor([x,y,w,h,img_w,img_h])
                    pos_list.append(pos_info)
                    
            ins_stack = torch.stack([x for x in lesion_list], dim=0)
            pos_stack = torch.stack([x for x in pos_list], dim = 0)

            grey_imgs.append(ins_stack)
            img_info.append(pos_stack)
            idx_temp += region_num
            idx_list.append(idx_temp)
        bag_imgs = torch.cat([x for x in grey_imgs], dim=0)
        bag_pos = torch.cat([x for x in img_info], dim=0)
        return bag_imgs, bag_pos, label, idx_list

    def __len__(self):
        return len(self.patient_dict)
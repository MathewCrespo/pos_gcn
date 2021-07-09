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

class DirectBags(Dataset):
    '''
    This Dataset is for doing benign/maglignant classification with US or SWE or two modal together

    Args:
        root: (str) data root
        sub_list: (int) sub path list [0, 1, 2, 3, 4] indicating which subset to use
        pre_transform: (torchvision.transform) the transform used
        modality: (int) indicate which modality to use. 0: US 1: SWE 2: US & SWE
    '''
    def __init__(self, root, pre_transform=None, sub_list = [0], task = 'BM'):
        self.root = root
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
            self.scan(os.path.join(self.root,str(i)))
        self.cut_null()
    def cut_null(self):  # delete empty bags
        null_record = []
        for i in range(len(self.patient_dict)):
            now_patient = self.patient_dict[i]
            if len(now_patient['images']['grey']) == 0:  # empty bag
                null_index = i
                null_record.append(null_index)

        self.patient_dict = [self.patient_dict[i] for i in range(len(self.patient_dict)) if i not in null_record]
        #print(null_record)
    
    def get_neg_ins (self):
        neg_ins = []
        for i in range(len(self.patient_dict)):
            now_patient = self.patient_dict[i]
            label = now_patient['label'][0]
            if label == 0:  # benign
                neg_ins_path = now_patient['images']['grey']
                neg_ins.extend(neg_ins_path)
        return neg_ins, len(neg_ins)

    def parse_img_info(self, img_info_path):
        f = open(img_info_path,'r')
        line = f.readline().strip('\n')
        img_w = int(line.split(',')[0])
        img_h = int(line.split(',')[1])
        f.close()
        return img_w, img_h
    
    def parse_ins_info(self,log_path):
        f = open(log_path,'r')
        line = f.readline().strip('\n')
        x = int(line.split(',')[0])
        y = int(line.split(',')[1])
        w = int(line.split(',')[2])
        h = int(line.split(',')[3])
        f.close()
        return x,y,w,h

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
            img_w, img_h = self.parse_img_info(img_info_path)            
            for ins in ins_list:     # different lesion area in a single ultrasound image
                if is_image_file(ins):
                    region_num  += 1
                    ins_info = ins.split('.')[0]+'.txt'
                    ins_path = os.path.join(img_path,ins)
                    log_path = os.path.join(img_path,ins_info)
                    # read image 
                    ins_img = Image.open(ins_path)
                    ins_img = self.pre_transform(ins_img)
                    lesion_list.append(ins_img)
                    # parse bbox info
                    x,y,w,h = self.parse_ins_info(log_path)
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

    def scan(self, now_root):
        # 1- malignant  0-benign
        self.M_path = os.path.join(now_root, "Malignant")
        self.B_path = os.path.join(now_root, "Benign")
        # number of M
        ##scan benign path
        idx = 0
        for path in [self.B_path, self.M_path]:
            for patient_dir in os.listdir(path):
                patient_path = os.path.join(path, patient_dir)
                if os.path.isdir(patient_path):
                    ##assign labels
                    patient_info = {}
                    if path == self.M_path:
                        label = torch.Tensor([1, self.get_ALNM(patient_path)])
                    else:
                        label = torch.Tensor([0, 0])
                    
                    patient_info['dir'] = patient_path
                    patient_info['label'] = label
                    patient_info['images'] = {"grey": [], "pure": []}
                    grey_path = os.path.join(patient_path, "grey")
                    swe_path = os.path.join(patient_path, "pure")
                    ##scan grey folder as the reference (assume that grey file exists for sure)
                    for file_name in os.listdir(grey_path):
                        if is_image_file(file_name):
                            grey_file = os.path.join(grey_path, file_name)
                            swe_file = os.path.join(swe_path, file_name)
                            self.label_list.append(label)
                            
                            if os.path.exists(swe_file):
                                self.img_list.append([grey_file, swe_file])
                                patient_info["images"]["grey"].append(grey_file)
                                patient_info["images"]["pure"].append(swe_file)
                            else:
                                self.img_list.append([grey_file, None])
                                patient_info["images"]["grey"].append(grey_file)
                                patient_info["images"]["pure"].append(None)
                    ##update patient dict
                    self.patient_dict.append(patient_info)
                    idx += 1 
    


    def get_ALNM(self, path):
        """
        Get ALNM label from the report file: path/report.csv

        Returns:
            ALNM_label: (int) 0: no ALNM, 1: have ALNM
        """
        report_file = os.path.join(path, "report.csv")
        data = pd.read_csv(report_file)
        ALNM_info = data.loc[0, "淋巴结转移"]
        ##The rule of ALNM information: always start with digit, ex. 0/3, 13/29
        if str(ALNM_info)[0].isdigit():
            if self.ALNM_num is None:
                ALNM_label = int(str(ALNM_info).split('/')[0])
            else: 
                ALNM_label = (int(str(ALNM_info).split('/')[0])>=self.ALNM_num)
        else:
            ALNM_label = 0
        return ALNM_label

    def __len__(self):
        return len(self.patient_dict)

##Test Code
if __name__=="__main__":

    root =  '/remote-home/gyf/Ultrasound_CV/data/MergePhase1/5folds_pre'
    pre = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])

    trainset = DirectBags(root, pre_transform=pre,sub_list=[0],task='ALNM')
    print(trainset[0][0].shape)
    print(trainset[0][1].shape)
    print(trainset[0][2])
    print(trainset[0][3])
    '''
    bag_state = []
    
    # statistics
    for i in range(len(trainset)):
        bag_label = trainset[i][1].int().tolist()
        bag_state.append(bag_label)
    maglinant_num_train = bag_state.count(1)

    testset = PatientBags(root+'/test', pre_transform=pre)
    bag_state = []
    for i in range(len(testset)):
        bag_label = testset[i][1].int().tolist()
        bag_state.append(bag_label)
    maglinant_num_test = bag_state.count(1)
    print('{} maglinant bags out of {} in trainset'.format(maglinant_num_train, len(trainset)))
    #print('{} maglinant bags out of {} in testset'.format(maglinant_num_test, len(testset)))
    '''

    
    


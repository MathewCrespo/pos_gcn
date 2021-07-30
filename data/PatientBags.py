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

'''
BMBags create bags by each patient rather than by random instances.
'''

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def is_xml_file(filename):
    return filename.endswith('xml')

class PatientBags(Dataset):
    '''
    This Dataset is for doing benign/maglignant classification with US or SWE or two modal together

    Args:
        root: (str) data root
        sub_list: (int) sub path list [0, 1, 2, 3, 4] indicating which subset to use
        pre_transform: (torchvision.transform) the transform used
        modality: (int) indicate which modality to use. 0: US 1: SWE 2: US & SWE
    '''
    def __init__(self, root, pre_transform=None, modality=0, cls_task='BM', 
    metric_mode='patient', crop_mode = False, mix_mode=0, sub_list = [0],crop_list = False):
        
        self.modality = modality
        self.root = root
        self.img_list = [] ## a list of tuple (grey_filename, swe_filename)
        self.pre_transform = pre_transform
        self.patient_dict = [] ## a dict of information: {name: {label:, images: {grey:, pure:}}} 
        self.label_list = []
        self.metric_label_list = []
        self.ALNM_num = 1
        self.cls_task = cls_task
        self.metric_mode = metric_mode
        self.num_pbag = 0

        self.sub_list = sub_list
        for i in self.sub_list:
            self.scan(os.path.join(self.root,str(i)))
        self.neg_ins, self.num_neg_ins = self.get_neg_ins()
        #print(len(self.neg_ins))
        self.cut_null()
        #print(self.patient_dict)
        #print(len(self.patient_dict[155]['images']['grey']))
        
        self.crop_mode = crop_mode
        
        self.crop_list = crop_list

        # different ways to mix bags
        self.mix_mode = mix_mode

    def cut_null(self):  # delete empty bags
        null_record = []
        for i in range(len(self.patient_dict)):
            now_patient = self.patient_dict[i]
            if len(now_patient['images']['grey']) == 0:  # empty bag
                null_index = i
                null_record.append(null_index)

        self.patient_dict = [self.patient_dict[i] for i in range(len(self.patient_dict)) if i not in null_record]
        #print(null_record)

    def crop(self, img, size=112):
            crop_bag = []
            for i in range(img.shape[1]//size):
                for j in range(img.shape[2]//size):
                    crop_bag.append(img[:, (i)*size:(i+1)*size, (j)*size:(j+1)*size])

            
            #eturn crop_bag
            
            return torch.stack([x for x in crop_bag], dim=0)      #original code  
    
    def get_neg_ins (self):
        neg_ins = []
        for i in range(len(self.patient_dict)):
            now_patient = self.patient_dict[i]
            label = now_patient['label'][0]
            if label == 0:  # benign
                neg_ins_path = now_patient['images']['grey']
                neg_ins.extend(neg_ins_path)
        return neg_ins, len(neg_ins)


    def __getitem__(self, idx):
        ### start from patient_dict and create a dict
        #print(idx)
        now_patient = self.patient_dict[idx]
        label = now_patient['label'][0]
        grey_img_path = now_patient['images']['grey']
        grey_imgs = []
        if self.mix_mode == 0:  # no operations
            idx_list = [0]
            idx_temp = 0
            for path in grey_img_path:
                grey_img = Image.open(path).convert('RGB')
                if self.pre_transform is not None:
                    grey_img = self.pre_transform(grey_img)
                if self.crop_mode:
                    grey_img = self.crop(grey_img)
                grey_imgs.append(grey_img)

                idx_temp += 64
                idx_list.append(idx_temp)
            bag_imgs = torch.cat([x for x in grey_imgs], dim=0)
            if self.crop_list:
                return bag_imgs,label, idx_list
            else:
                return bag_imgs, label

        elif self.mix_mode == 1:  # mix neg instances
            
            if label.item()==1:
                num_pos_ins = len(grey_img_path)
                if num_pos_ins >2:
                    replace_num = randint(1,num_pos_ins-1)
                else:
                    replace_num = num_pos_ins - 1
                

                replace_neg = sample(self.neg_ins, replace_num)
                #print(replace_neg)
                grey_img_path = grey_img_path[0:len(grey_img_path)-replace_num]
                grey_img_path.extend(replace_neg)
            else:
                replace_num = 0
            
            for path in grey_img_path:
                grey_img = Image.open(path).convert('RGB')
                if self.pre_transform is not None:
                    grey_img = self.pre_transform(grey_img)
                if self.crop_mode:
                    grey_img = self.crop(grey_img)
                    
                grey_imgs.append(grey_img)

            bag_imgs = torch.stack([x for x in grey_imgs], dim=0)
            return bag_imgs, label, replace_num   # when training and testing, rember to add (imgs, label, _) in ...
        '''
         # label
        if self.cls_task == "BM":
            label = self.label_list[idx][0]
        elif self.cls_task == "ALNM":
            label = self.label_list[idx][1]
        else:
            label = self.label_list[idx]

        # image modality
        img_grey, img_swe = self.img_list[idx]

        if self.modality == 0:
            img = Image.open(img_grey).convert('RGB')
            
        
            if self.pre_transform is not None:
                img = self.pre_transform(img)
            
            if self.crop_mode:
                img = self.crop(img)
            return img, label

        elif self.modality == 1:
            img = Image.open(self.img_swe).convert('RGB')
        
            if self.pre_transform is not None:
                img = self.pre_transform(img)
            img = self.crop(img)
            return img, label
        else:
            img_grey = Image.open(img_grey).convert('RGB')
            img_pure = Image.open(img_swe).convert('RGB')
            
        
            if self.pre_transform is not None:
                img_grey = self.pre_transform(img_grey)
                img_pure = self.pre_transform(img_pure)
            img_grey = self.crop(img_grey)
            img_pure = self.crop(img_pure)
            return [img_grey, img_pure], label
        '''


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
                        label = torch.Tensor([1, 0])
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
                            ## different metric mode
                            if self.metric_mode == "patient":
                                self.metric_label_list.append(idx)
                            elif self.metric_mode == "BM":
                                self.metric_label_list.append(label[0].long())
                            elif self.metric_mode == "ALNM":
                                self.metric_label_list.append(label[1].long())
                            else:
                                self.metric_label_list.append(idx)
                            
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

    root =  '/remote-home/my/Ultrasound_CV/data/MergePhase1/5folds'
    pre = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])

    trainset = PatientBags(root,pre_transform= pre,sub_list=[1,2,3,4],crop_list=True)
    print(trainset[0][0].shape)
    print(trainset[0][1])
    print(trainset[0][2])



    
    


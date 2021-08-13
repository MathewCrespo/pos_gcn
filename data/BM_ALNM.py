from PIL import Image
import torch
import pandas as pd
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class BM_ALNM(Dataset):
    '''
    This dataset is for doing ALNM and B/M multi-task with grey and pure two modals
    
    The dataset folder is organized as:
        ./root 
            Benign.xlsx
            Malignant.xlsx
            Benign/
                name1/
                    report.xlsx
                    grey/
                        name1_img1.jpg
                        name1_img2.jpg
                    pure/
                        name1_img1.jpg
                        name1_img2.jpg
                name2/
            Malignant/ 
                (the same as Benign/)
    Args:
        root: (str) the root of the data
        pre_transform: (torchvision.transform) the image transformed used

    Notes:
        1. the grey image file and the swe image file is with the same name.
        2. Use return mode to control returned images
    '''
    def __init__(self, root, pre_transform=None, ALNM_num=1, target_transform=None, return_mode="both"):
        self.root = root
        self.ALNM_num=ALNM_num
        self.pre_transform = pre_transform
        self.target_transform = target_transform
        self.label_list = []
        self.img_list = [] ## a list of tuple (grey_filename, swe_filename)
        self.patient_dict = {} ## a dict of information: {name: {label:, images: {grey:, pure:}}}
        assert return_mode in ["grey", "pure", "both"]
        self.return_mode = return_mode
        self._scan()
        
    def _scan(self):
        self.M_path = os.path.join(self.root, "Malignant")
        self.B_path = os.path.join(self.root, "Benign")
        ##scan benign path
        for patient_dir in os.listdir(self.B_path):
            patient_path = os.path.join(self.B_path, patient_dir)
            if os.path.isdir(patient_path):
                ##assign labels
                patient_info = {}
                label = torch.Tensor([0, 0])
                if self.target_transform is not None:
                    label = self.target_transform(label)
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
                self.patient_dict[patient_dir] = patient_info
        
        ##scan malignant path
        for patient_dir in os.listdir(self.M_path):
            patient_path = os.path.join(self.M_path, patient_dir)
            if os.path.isdir(patient_path):
                ##assign labels
                patient_info = {}
                label = torch.Tensor([1, self.get_ALNM(patient_path)])
                if self.target_transform is not None:
                    label = self.target_transform(label)
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
                self.patient_dict[patient_dir] = patient_info

    def __getitem__(self, idx):
        """
        Returns:
            img_grey: (torch.Tensor) [3, H, W] grey image tensor
            img_pure: (torch.Tensor) [3, H, W] swe image tensor
            label: (torch.Tensor) [2] (0, 0) for benign, (1, 0) malignant and no ALNM, (1, 1) malignant and with ALNM
        
        Notes:
            1. if one modality is lost, return None, read _scan() for more details
            2. Can use a target_transform() to transform the multi-label label to multi-task label
        """
        img_grey, img_swe = self.img_list[idx]
        if img_grey is not None:
            img_grey = Image.open(img_grey).convert('RGB')
            if self.pre_transform is not None:
                img_grey = self.pre_transform(img_grey)
        
        if img_swe is not None:
            img_swe = Image.open(img_swe).convert('RGB')
            if self.pre_transform is not None:
                img_swe = self.pre_transform(img_swe)
        
        label = self.label_list[idx]
        
        if self.return_mode == "grey":
            return img_grey, label
        elif self.return_mode == "pure":
            return img_swe, label
        else:
            return img_grey, img_swe, label

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
                ALNM_label = (int(str(ALNM_info).split('/')[0])>self.ALNM_num)
        else:
            ALNM_label = 0
        return ALNM_label
    
    def __len__(self):
        return len(self.label_list)

        
if __name__=='__main__':
    dataset = BM_ALNM("/remote-home/gyf/Ultrasound_CV/data/ABD/val/", None)
    print(dataset[0])
    malignant_count = 0
    ALNM_count = 0
    print(len(dataset.patient_dict))
    for key, value in dataset.patient_dict.items():
        if value['label'][0].item() > 0:
            malignant_count += 1
        if value['label'][1].item() > 0:
            ALNM_count += 1
    
    print('Malignant patients:', malignant_count)
    print('ALNM patients:', ALNM_count)

    label_number = len(dataset.label_list)
    maglinant_number = 0
    ALNM_number = 0
    print("image number:", label_number)
    for label in dataset.label_list:
        if label[0] > 0:
            maglinant_number += 1
        if label[1] > 0:
            ALNM_number += 1

    print("malignant images:", maglinant_number)
    print("ALNM images:", ALNM_number)
    

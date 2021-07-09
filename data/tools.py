'''
tools.py is for playing fun in patient's dataset
Different methods to create ultrasound bags
'''

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

'''
BMBags create bags by each patient rather than by random instances.
'''

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def is_xml_file(filename):
    return filename.endswith('xml')

class Mixed_Data(Dataset):
    '''
    This Dataset is for doing benign/maglignant classification with US or SWE or two modal together

    Args:
        root: (str) data root
        sub_list: (int) sub path list [0, 1, 2, 3, 4] indicating which subset to use
        pre_transform: (torchvision.transform) the transform used
        modality: (int) indicate which modality to use. 0: US 1: SWE 2: US & SWE
    '''
    def __init__(self, root, pre_transform=None, modality=0, cls_task='BM', 
    metric_mode='patient', ratio, crop_mode = False):
        
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
        self.ratio = ratio
        self.scan()

        
        self.cut_null()
        #print(self.patient_dict)
        #print(len(self.patient_dict[155]['images']['grey']))
        
        self.crop_mode = crop_mode

    def cut_null(self):  # delete empty bags
        null_record = []
        for i in range(len(self.patient_dict)):
            now_patient = self.patient_dict[i]
            if len(now_patient['images']['grey']) == 0:  # empty bag
                null_index = i
                null_record.append(null_index)

        self.patient_dict = [self.patient_dict[i] for i in range(len(self.patient_dict)) if i not in null_record]
        #print(null_record)

    def crop(self, img, size=28):
            crop_bag = []
            for i in range(img.shape[1]//size):
                for j in range(img.shape[2]//size):
                    crop_bag.append(img[:, (i)*size:(i+1)*size, (j)*size:(j+1)*size])

            
            #eturn crop_bag
            
            return torch.stack([x for x in crop_bag], dim=0)      #original code  
    
    # def create_bags(self)   bag function?

    #def get_all (self):


    def __getitem__(self, idx):
        ### start from patient_dict and create a dict
        #print(idx)
        now_patient = self.patient_dict[idx]
        label = now_patient['label'][0]
        grey_img_path = now_patient['images']['grey']
        grey_imgs = []
        for path in grey_img_path:
            grey_img = Image.open(path).convert('RGB')
            if self.pre_transform is not None:
                grey_img = self.pre_transform(grey_img)
            grey_imgs.append(grey_img)
        bag_imgs = torch.stack([x for x in grey_imgs], dim=0)
        return bag_imgs, label
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


    def scan(self):
        # 1- malignant  0-benign
        self.M_path = os.path.join(self.root, "Malignant")
        self.B_path = os.path.join(self.root, "Benign")

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

    root =  '/media/hhy/data/USdata/MergePhase1/test_0.3'
    pre = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])

    trainset = PatientBags(root+'/train',pre_transform= pre)
    testset = PatientBags(root+'/test', pre_transform=pre)

    bag_state = []

    for i in range(len(trainset)):
        bag_label = trainset[i][1].int().tolist()
        bag_state.append(bag_label)
    maglinant_num_train = bag_state.count(1)

    bag_state = []
    for i in range(len(testset)):
        bag_label = testset[i][1].int().tolist()
        bag_state.append(bag_label)
    maglinant_num_test = bag_state.count(1)

    print('{} maglinant bags out of {} in trainset'.format(maglinant_num_train, len(trainset)))
    print('{} maglinant bags out of {} in testset'.format(maglinant_num_test, len(testset)))
        
        


    
    


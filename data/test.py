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



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def is_xml_file(filename):
    return filename.endswith('xml')

class SSBags(Dataset):
    '''
    This Dataset is for doing benign/maglignant classification with US or SWE or two modal together

    Args:
        root: (str) data root
        sub_list: (int) sub path list [0, 1, 2, 3, 4] indicating which subset to use
        pre_transform: (torchvision.transform) the transform used
        modality: (int) indicate which modality to use. 0: US 1: SWE 2: US & SWE
    '''
    def __init__(self, root, pre_transform=None, modality=0, cls_task='BM', 
    metric_mode='patient', selective_mode = True, mix_mode=0, sub_list = [0]):
        
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
        
        self.cut_null()
        self.selective_mode = selective_mode
        
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

    def crop(self, img, size=28):
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

    
    def s_search(self,img, scale=1, sigma=0.9,min_size=100):
        img_lbl, regions = selectivesearch.selective_search(
        img, scale, sigma, min_size)
        candidates = set()
        bbox = []
        area_list = []
        
        for r in regions:
        # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
        # excluding regions smaller than 2000 pixels
            if r['size'] < 1000:
                continue
        # distorted rects    
            x, y, w, h = r['rect']
            
            
            if w/h >1.2 or h/w>1.2:
                continue
            temp = img[y:y+h,x:x+w,:]
            area = r['size']
            area_ratio = area/img.size
            #print(x,y,w,h)
            standard_img = cv2.resize(temp,(224,224))
            standard_img = Image.fromarray(standard_img)
            standard_img = self.pre_transform(standard_img)
            #standard_img = F.interpolate(temp,size=(28,28))
            bbox.append(standard_img)
            area_list.append(area_ratio)
        
        if len(bbox) == 0:
            standard_img = cv2.resize(img,(224,224))
            standard_img = Image.fromarray(standard_img)
            standard_img = self.pre_transform(standard_img)
            bbox.append(standard_img)
            area_list.append(1)

            
        #print(len(bbox))
        return torch.stack([ x for x in bbox], dim = 0), len(bbox), area_list

    def __getitem__(self, idx):
        ### start from patient_dict and create a dict
        #print(idx)
        now_patient = self.patient_dict[idx]
        label = now_patient['label'][0]
        grey_img_path = now_patient['images']['grey']
        grey_imgs = []
        idx_list = [0]
        area_ls = []
        l_ls = []
        idx_temp = 0
        if self.mix_mode == 0:  # no operations
            for path in grey_img_path:
                grey_img = cv2.imread(path)
                if self.selective_mode:
                    grey_img, region_num, area_list = self.s_search(grey_img)

                grey_imgs.append(grey_img)
                l_ls.append(region_num)
                idx_temp += region_num
                idx_list.append(idx_temp)
                print(area_list)
                area_ls.extend(area_list)
            bag_imgs = torch.cat([x for x in grey_imgs], dim=0)
            return bag_imgs, label, idx_list, area_ls, l_ls

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

    def __len__(self):
        return len(self.patient_dict)


##Test Code
if __name__=="__main__":

    root =  '/remote-home/gyf/Ultrasound_CV/data/MergePhase1/5folds'
    p = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor()
        ])

    trainset = SSBags(root, pre_transform=p,sub_list=[0,1,2,3,4])
    a,b,c,d,l_list = trainset[0]
    print(c)
    print(d)
    print(l_list)
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

    
    


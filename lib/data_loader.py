import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from PIL import Image

figsize = 64

def GetLabelFromFn(fn):
    fnList = fn.split('/')
    folder_name = fnList[-2]
    img_fn = fnList[-1]
    phone, session, human_id, access = [int(num)-1 for num in folder_name.split('_')]
    photo_id = int(img_fn.split('.')[0])
    #print(phone, session, human_id, access, photo_id)
    return phone, session, human_id, access, photo_id

class OuluNpu(Dataset):
    def __init__(self, root):
        self.root = root
        self.DictOfFolder = {}
        fnList = glob.glob(os.path.join(self.root,'*','*.png'))
        
        for fn in fnList:
            folder_name = fn.split('/')[-2]
            fnSubList = self.DictOfFolder.get(folder_name)
            self.DictOfFolder[folder_name] = [] if not fnSubList else fnSubList
            self.DictOfFolder[folder_name].append(fn)
            
        for key in self.DictOfFolder:
            self.DictOfFolder[key] = sorted(self.DictOfFolder[key])
            
        self.folderNameList = [key for key in self.DictOfFolder]
        self.len = len(self.DictOfFolder)
        self.transforms = transforms.Compose([
            transforms.Resize(figsize),
            transforms.ToTensor(),
            transforms.Normalize(mean = (.485, .456, .406),
                                 std  = (.229, .224, .225))
        ])
       
    def __getitem__(self, idx):
        folder_name = self.folderNameList[idx]
        fnList = self.DictOfFolder[folder_name]
        imgList = [Image.open(fn) for fn in fnList]
        imgList = [self.transforms(img) for img in imgList]
        #imgList = [img.view(1, 3, figsize, figsize) for img in imgList]
        #img = torch.cat(imgList, 0)
        phone, session, human_id, access, photo_id = GetLabelFromFn(fnList[0])
        return imgList, phone, session, human_id, access, photo_id
    
    def __len__(self):
        return self.len



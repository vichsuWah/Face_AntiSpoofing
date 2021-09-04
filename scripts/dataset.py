import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
def readLabel(path, order = 'single', binary = True):
    folders = sorted(os.listdir(path))
    phones = []
    sessions = []
    ids = []
    labels = []
    for i, folder in enumerate(folders):
        image_path = os.path.join(path, folder)
        image_dir = sorted(os.listdir(image_path))
        split = image_path.split('/')[-1].split('_')
        phone = int(split[0])-1
        session = int(split[1])-1
        id = int(split[2])-1
        label = int(split[3])-1
        if binary:
            label = 0 if label==0 else 1
        else:
            if label==0:
                label = 0
            elif (label==1 or  label==2):
                label = 1 
            elif (label==3 or  label==4):
                label = 2  
        if order == 'group':
            phones.append(phone)
            sessions.append(session)
            ids.append(id)
            labels.append(label)
        elif order == 'single':
            phones.extend([phone]*(11))
            sessions.extend([session]*(11))
            ids.extend([id]*(11))
            labels.extend([label]*(11))
    return np.array(phones), np.array(sessions), np.array(ids), np.array(labels)
def readImg(path, order = 'single', binary = True,size = 256):
    folders = sorted(os.listdir(path))
    if order == 'single':
        x = np.zeros([len(folders*11),size,size,3], dtype=np.uint8)
    elif order == 'group':
        x = np.zeros([len(folders),11,size,size,3], dtype=np.uint8)
    sess = []
    for i, folder in enumerate(tqdm(folders)):
        image_path = os.path.join(path, folder)
        image_dir = sorted(os.listdir(image_path))
        for j, file in enumerate(image_dir):
            if order == 'single':
                x[i*11+j,:,:] = np.array(Image.open(os.path.join(image_path, file)).resize((size, size), Image.ANTIALIAS))[:,:,:3]
            elif order == 'group':
                x[i,j,:,:] = np.array(Image.open(os.path.join(image_path, file)).resize((size, size), Image.ANTIALIAS))[:,:,:3]
    return x
class ImgDataset(Dataset):
    def __init__(self, x, y=None, sess=None, transform=None, order = 'single', binary = True):
        self.x = x
        # label is required to be a LongTensor
        self.order = order
        self.binary = binary
        self.y = y
        self.sess = sess
        self.transform = transform

    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        imgs = self.x[index]
        if self.order == 'group':
            tmp = torch.zeros([imgs.shape[0],3,256,256])
            for i, img in enumerate(imgs):
                # img = Image.open(file).resize((512, 512), Image.ANTIALIAS)
                if self.transform is not None:
                    tmp[i,:,:,:] = self.transform(img)   
            imgs = tmp
        elif self.order == 'single':
            # imgs = Image.open(image_dir).resize((512, 512), Image.ANTIALIAS)
            # imgs = image_dir
            if self.transform is not None:
                imgs = self.transform(imgs)
        if self.y is not None:
            Y = torch.tensor(self.y[index],dtype=(torch.float if self.binary else torch.long))
            S = torch.tensor(self.sess[index],dtype=torch.long)
            return imgs, Y, S
        else:
            return imgs
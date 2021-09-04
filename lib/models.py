import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import copy
from torchvision.utils import save_image
import PIL
import pandas as pd
import matplotlib.pyplot as plt
import random

figsize = 64
framesize = 20

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet= models.resnet18(pretrained=False)
        self.fc = nn.Sequential(
            nn.Linear(11000, 5000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 5)
        )
            

    def forward(self, imgList):
        featList = [self.resnet(img) for img in imgList]
        featList = [feat.view(-1,1,1000) for feat in featList]
        feat = torch.cat(featList, 1)
        feat = feat.reshape(-1,11000)
        return self.fc(feat)
    

class CooperativeGAN(nn.Module):
    def __init__(self):
        super(CooperativeGAN, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.resnet= models.resnet152()
        self.figsize = figsize
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=1000,
            hidden_size=1000,
            num_layers=5
        )
        self.decoder = nn.Sequential(
            nn.Linear(1200, 64),
            *block(64, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(3*self.figsize**2)),
            nn.Tanh()
        )
        
        self.CondGen = nn.Sequential(
            nn.Linear(3, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        self.CondClass = nn.Sequential(
            nn.Linear(3, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1200, 500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        
    def forward(self, imgList, cond):
        nFake = framesize - len(imgList)
        idxList = [random.randint(0, len(imgList)-1) for i in range(nFake)]
        fakeList = [imgList[idx] for idx in idxList]
        fakeList = [self.ResnetGen(img) for img in fakeList]
        fakeList = [feat + torch.rand(feat.shape).to(feat.device) for feat in fakeList]
        fakeList = [torch.cat([self.CondGen(cond), feat], 1) for feat in fakeList]
        
        fakeList = [self.decoder(feat) for feat in fakeList]
        fakeList = [img.view(-1, 3 ,self.figsize, self.figsize) for img in fakeList]
        imgList += fakeList
        #print(fakeList[0].shape)
        fakeList.clear()
        featList = [self.ResnetClass(img) for img in imgList]
        featList = [feat.view(-1,1,1000) for feat in featList]
        feat = torch.cat(featList,1)
        feat, _ = self.lstm(feat)
        feat = feat[:,-1]
        feat = torch.cat([self.CondClass(cond), feat], 1)
        return self.fc(feat)
        
        
class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.resnet = models.resnet152()
        self.figsize = figsize
        self.decoder = nn.Sequential(
            nn.Linear(1200, 64),
            *block(64, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(3*self.figsize**2)),
            nn.Tanh()
        )
        self.CondGenerator = nn.Sequential(
            nn.Linear(4, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.CondClassifier = nn.Sequential(
            nn.Linear(3, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1200, 500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 5)
        )
        
        
    def forward(self, img, cond, mode):
        assert(mode in ["generator", "classifier"])
        feat = self.resnet(img)
        cond = self.CondGenerator(cond) if mode == "generator" else self.CondClassifier(cond)
        feat = torch.cat([cond, feat], 1)
        if mode == "generator":
            feat += torch.rand(feat.shape).to(img.device)
            output = self.decoder(feat)
            img = output.view(len(output),3 ,self.figsize, self.figsize)
            return img
        else:
            return self.classifier(feat)

            
    
    
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.figsize = figsize
        self.model = nn.Sequential(
            nn.Linear(200+figsize**2*3, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.CondFc = nn.Sequential(
            nn.Linear(4, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
    def forward(self, img, cond):
        cond = self.CondFc(cond)
        feat = torch.cat([img.view(img.size(0), -1), cond], 1)
        validity = self.model(feat)
        return validity
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
    
def saveModel(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def loadModel(checkpoint_path, model, optimizer, log=True):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    if log:
        print('model loaded from %s' % checkpoint_path)
    
def GetCudaDevice(cuda = 1, seed = 123, log=True):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    deviceName = "cuda:"+str(cuda)
    device = torch.device(deviceName if use_cuda else "cpu")
    #device = torch.device('cpu')
    print('Device used:', device)
    return device

def toDevice(dataList, device):
    return [data.to(device) for data in dataList]
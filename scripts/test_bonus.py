import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dataset import ImgDataset, readLabel, readImg
from models import DG_model
def test(args):
    workspace = args.workspace
    test_x = readImg(workspace, order='group')
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 60
    test_set = ImgDataset(test_x, None, None,test_transform, 'group')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = DG_model(model = args.model_type, num_cls = 1 ).cuda()
    model_path = args.checkpoint_real_fake
    model.load_state_dict(torch.load(model_path))
    model.eval()
    real_fake_preds = []
    with torch.no_grad():
        for vids in tqdm(test_loader):
            for vid in vids:
                class_logits, embedded = model(vid.cuda())
                class_logits = torch.mean(class_logits.squeeze()).cpu().numpy()
                real_fake_preds.append(class_logits)
    real_fake_preds = np.where(np.array(real_fake_preds) > 0.5, 1, 0)

    model_path = args.checkpoint_print_replay
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print_replay_preds = []
    with torch.no_grad():
        for vids in tqdm(test_loader):
            for vid in vids:
                class_logits, embedded = model(vid.cuda())
                class_logits = torch.mean(class_logits.squeeze()).cpu().numpy()
                print_replay_preds.append(class_logits)
    print_replay_preds = np.where(np.array(print_replay_preds) > 0.5, 2, 1)
    result = real_fake_preds*print_replay_preds
    image_dir = sorted(os.listdir(workspace))
    df = pd.DataFrame({'video_id':image_dir, 'label': result})
    df.to_csv(args.outfile,index=False)
def parse_args():
    parser = argparse.ArgumentParser(description='Face Anti-spoofing')
    parser.add_argument('--workspace',default='data/oulu_npu_cropped/test', type=str, help='dataset path')
    parser.add_argument('--model_type', default='resnet18', type=str, help='model backbone')
    parser.add_argument('--checkpoint_real_fake', default='models/model-final.pth', type=str, help="checkpoint path")
    parser.add_argument('--checkpoint_print_replay', default='models/model-bonus.pth', type=str, help="checkpoint path")
    parser.add_argument('--outfile', default='bonus.csv', type=str, help='outfile name')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    test(args)
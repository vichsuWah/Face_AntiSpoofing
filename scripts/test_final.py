import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import str2bool
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
    model_path = args.checkpoint
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds = []
    with torch.no_grad():
        for vids in tqdm(test_loader):
            for vid in vids:
                class_logits, embedded = model(vid.cuda())
                class_logits = torch.mean(class_logits.squeeze()).cpu().numpy()
                test_preds.append(class_logits)
    import pandas as pd
    result = np.array(test_preds)*(-1)+1 
    image_dir = sorted(os.listdir(workspace))
    df = pd.DataFrame({'video_id':image_dir, 'label': result})
    df.to_csv(args.outfile,index=False)
def parse_args():
    parser = argparse.ArgumentParser(description='Face Anti-spoofing')
    parser.add_argument('--workspace',default='data/oulu_npu_cropped/test', type=str, help='dataset path')
    parser.add_argument('--model_type', default='resnet18', type=str, help='model backbone')
    parser.add_argument('--outfile', default='oulu.csv', type=str, help='outfile name')
    parser.add_argument('--checkpoint', default='models/model-final.pth', type=str, help="checkpoint path")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    test(args)
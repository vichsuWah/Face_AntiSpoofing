import os
import sys
import argparse
import numpy as np
from dataset import ImgDataset, readLabel, readImg
def preprocess(args):
    oulu_workspace = os.path.join(args.workspace, 'oulu_npu_cropped')

    train_folder = os.path.join(oulu_workspace,'train')
    train_x = readImg(train_folder, order='single')
    np.save(os.path.join(oulu_workspace,'train_x.npy'), train_x)

    # train_rembg_folder = os.path.join(oulu_workspace,'train_rembg')
    # train_x_rembg = readImg(train_rembg_folder, order='single')
    # np.save(os.path.join(oulu_workspace,'train_x_rembg.npy'), train_x_rembg)

    val_folder = os.path.join(oulu_workspace,'val')
    val_x = readImg(val_folder, order='single')
    np.save(os.path.join(oulu_workspace,'val_x.npy'), val_x)

    test_folder = os.path.join(oulu_workspace,'test')
    test_x = readImg(test_folder, order='group')
    np.save(os.path.join(oulu_workspace,'test_x.npy'), test_x)

    siw_workspace = os.path.join(args.workspace, 'siw_test')
    test_x = readImg(siw_workspace, order='group')
    np.save(os.path.join(args.workspace,'siw_test_x.npy'), test_x)

def parse_args():
    parser = argparse.ArgumentParser(description='Face Anti-spoofing')
    parser.add_argument('--workspace',default='data', type=str, help='dataset path')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    preprocess(args)
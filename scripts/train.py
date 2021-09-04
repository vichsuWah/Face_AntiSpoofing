import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from tensorboardX import SummaryWriter

from dataset import ImgDataset, readLabel, readImg
from models import DG_model, DomainClassifier, Discriminator
from utils import auc_acc, str2bool
from hard_triplet_loss import HardTripletLoss
def train(args):
    workspace_dir = args.workspace
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    binary = args.binary
    print('binary:{}'.format(binary))
    bs = args.batch_size
    train_folder = os.path.join(workspace_dir,'train_rembg')
    phones, sess, ids, train_y = readLabel(train_folder, order='single', binary=binary)
    train_x_rembg = np.load(os.path.join(workspace_dir,'train_x_rembg.npy'))
    train_x_orig = np.load(os.path.join(workspace_dir,'train_x.npy'))
    train_x = np.concatenate([train_x_orig, train_x_rembg])
    train_dataset = ImgDataset(train_x, np.tile(train_y,(2)), np.tile(sess,(2)), train_transform, 'single', binary)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    val_folder = os.path.join(workspace_dir,'val')
    val_x = np.load(os.path.join(workspace_dir,'val_x.npy'))
    phones, sess, ids, val_y = readLabel(os.path.join(workspace_dir,'val'), order='single', binary=binary)
    val_set = ImgDataset(val_x, val_y, sess, test_transform, 'single', binary)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)



    model = DG_model(model = args.model_type, num_cls = (1 if binary else 3)).cuda()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    domain_classifier = Discriminator(num_cls=2).cuda()

    criterion = {'class': (nn.BCELoss().cuda() if binary else nn.CrossEntropyLoss().cuda()),
                 'triplet': HardTripletLoss(margin=0.1, hardest=args.hardest).cuda(),
                 'domain': nn.CrossEntropyLoss().cuda()}
    optimizer = torch.optim.SGD(model.parameters(),lr=args.model_lr, weight_decay=args.model_l2)  # optimizer 使用 Adam
    optimizer_D = torch.optim.Adam(domain_classifier.parameters(),lr=args.sess_disc_lr, weight_decay=args.sess_disc_l2)  # optimizer 使用 Adam
    num_epoch = args.epoch
    if args.log:
        logdir = 'logdir/' + str(model.__class__.__name__)
        if logdir == None:
            logdir = datetime.now().strftime("%m-%d-%H:%M:%S")
        else:
            logdir += datetime.now().strftime("_%m-%d-%H:%M:%S")
        writer = SummaryWriter(logdir)
        text_file = open(os.path.join(logdir,'config.txt'), 'w')
        for arg in vars(args):
            text_file.write(arg+' '+str(getattr(args, arg))+'\n')
        text_file.close()
    val_acc_prev = 0
    val_auc_prev = 0
    for epoch in range(num_epoch):
        
        epoch_start_time = time.time()
        train_loss = 0.0
        triplet_loss = 0.0
        domain_loss_total = 0.0
        cls_loss_total = 0.0
        val_loss = 0.0
        D_loss = 0.0
        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        train_logits = []
        train_labels = []
        for data, label, sess in tqdm(train_dataloader):
            data = data.cuda()
            label = label.cuda()
            sess = sess.cuda()
            domain_label = sess
            # train domain classifier
            pred, feature = model(data.cuda())
            domain_logits = domain_classifier(feature.detach())
            
            loss = criterion['domain'](domain_logits, domain_label)
            D_loss+= loss.item()
            
            loss.backward()
            optimizer_D.step()
            # train main model
            pred, feature = model(data.cuda())
            class_logits = pred[:data.shape[0]]
            domain_logits = domain_classifier(feature)

            cls_loss = criterion['class'](class_logits.squeeze(), label)
            triplet = criterion["triplet"](feature, label)
            domain_loss = criterion['domain'](domain_logits, domain_label)
            
            loss =  cls_loss + args.triplet_coef*triplet + args.sess_coef*domain_loss
            loss.backward()
            optimizer.step() 
            
            optimizer.zero_grad() 
            optimizer_D.zero_grad()
            # log
            cls_loss_total += cls_loss.item()
            triplet_loss += args.triplet_coef*triplet 
            domain_loss_total += args.sess_coef*domain_loss.item()
            
            train_loss += loss.item()
            train_logits.append(class_logits)
            train_labels.append(label)

        train_auc, train_acc = auc_acc(torch.cat(train_logits), torch.cat(train_labels), binary)
        # torchvision.utils.save_image(data, logdir + '/data.png', nrow=11)
        model.eval()
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for data, label, sess in tqdm(val_loader):
                class_logits, embedded = model(data.cuda())
                batch_loss = criterion['class'](class_logits, label.cuda())
                val_logits.append(class_logits)
                val_labels.append(label)
                val_loss += batch_loss.item()
            val_auc, val_acc = auc_acc(torch.cat(val_logits), torch.cat(val_labels), binary)

            D_loss /= train_dataloader.__len__()
            train_loss /= train_dataloader.__len__()
            cls_loss_total /= train_dataloader.__len__()
            triplet_loss /= train_dataloader.__len__()
            domain_loss_total /= train_dataloader.__len__()
            val_loss /= val_loader.__len__()
            print('[{:03}/{:03}] {:3.2f} sec(s) Train Acc: {:.6f} Train AUC: {:.6f} Train  loss: {:3.6f} D_loss: {:3.6f}'\
                .format(epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc, train_auc, train_loss, D_loss))
            print(' '*24+'cls  loss: {:3.6f} trip loss: {:3.6f} domain loss: {:3.6f}'\
                .format(cls_loss_total, triplet_loss, domain_loss_total ))
            print(' '*24+'Valid Acc: {:.6f} Valid AUC: {:.6f} Valid  loss: {:3.6f} '\
                .format(val_acc, val_auc, val_loss))
            if args.log:    
                writer.add_scalars('loss', {'train':train_loss,'val':val_loss}, epoch)
                writer.add_scalars('acc', {'train':train_acc,'val':val_acc}, epoch)
                writer.add_scalars('auc', {'train':train_auc,'val':val_auc}, epoch)
                writer.flush()
                if val_acc > 0.93 or val_auc > 0.99:
                    model_path = logdir+'/models'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(), model_path+'/model-{}-{:.6f}-{:.6f}.pth'.format(epoch, val_acc, val_auc)) 
                    if val_acc > val_acc_prev:
                        val_acc_prev = val_acc
                        torch.save(model.state_dict(), model_path+'/model-acc-best.pth') 
                    if val_auc > val_auc_prev:
                        val_acc_prev = val_acc
                        torch.save(model.state_dict(), model_path+'/model-auc-best.pth')
                else:
                    model_path = logdir+'/models'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(), model_path+'/model-last.pth') 
def parse_args():
    parser = argparse.ArgumentParser(description='Face Anti-spoofing')
    parser.add_argument('--workspace',default='data/oulu_npu_cropped', type=str, help='dataset path')
    parser.add_argument('--model_type', default='resnet18', type=str, help='model backbone')
    parser.add_argument('--model_lr', default=1e-4, type=float, help='main model lr')
    parser.add_argument('--model_l2', default=1e-5, type=float, help='main model l2')
    parser.add_argument('--sess_disc_lr', default=1e-5, type=float, help='session discriminator lr')
    parser.add_argument('--sess_disc_l2', default=1e-4, type=float, help='session discriminator l2')
    # parser.add_argument('--phone_disc_lr', default=1e-5, type=float, help='phone discriminator lr')
    # parser.add_argument('--phone_disc_l2', default=1e-4, type=float, help='phone discriminator l2')
    parser.add_argument('--triplet_coef', default=1.0, type=float, help='triplet loss scalar')
    parser.add_argument('--sess_coef', default=0.0, type=float, help='session discriminator loss scalar')
    # parser.add_argument('--phone_coef', default=1.0, type=float, help='phone discriminator loss scalar')
    parser.add_argument('--log', type=str2bool, nargs='?', const=True, default=True, help='log')
    parser.add_argument('--hardest',  type=str2bool, nargs='?', const=True, default=False,help='triplet loss hardest')
    parser.add_argument('--binary', type=str2bool, nargs='?', const=True, default=True, help='binary classification or not')
    parser.add_argument('--batch_size', default=60, type=int, help='batch size')
    parser.add_argument('--epoch', default=100, type=int, help="Num of epoch")
    parser.add_argument('--checkpoint', default=None, type=str, help="checkpoint path")
    parser.add_argument('--val_class', default=1, type=int, help="checkpoint path")
    parser.add_argument('--notes', default=None, type=str, help='some additional notes')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    train(args)
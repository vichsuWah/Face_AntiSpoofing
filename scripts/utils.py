import torch
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
def auc_acc(class_logits, label, binary = True):
    label = label.cpu().numpy()
    if binary:
        class_logits = class_logits.squeeze().cpu().detach().numpy()
        pred = np.where(class_logits > 0.5, 1, 0)
    else:
        pred = np.argmax(class_logits.cpu().detach().numpy(), axis=1)
        class_logits = torch.cat([class_logits[:,:1],torch.mean(class_logits[:,1:], dim = 1, keepdims = True)], dim = 1)
        class_logits = F.softmax(class_logits, dim=1)[:,1].cpu().detach().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(np.clip(label,0,1), class_logits, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.mean(pred == label)
    return auc, acc
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
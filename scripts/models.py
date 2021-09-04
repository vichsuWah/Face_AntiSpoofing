import torch
import torch.nn as nn
import torchvision.models as models
import sys
import numpy as np
from torch.autograd import Variable
import random
import os

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Feature_Generator_ResNet(nn.Module):
    def __init__(self, model):
        super(Feature_Generator_ResNet, self).__init__()
        if model == 'resnet18':
            model_resnet = models.resnet18(pretrained = True)
        elif model == 'resnet50':
            model_resnet = models.resnet50(pretrained = True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return feature

class Feature_Embedder_ResNet(nn.Module):
    def __init__(self, model):
        super(Feature_Embedder_ResNet, self).__init__()
        if model == 'resnet18':
            model_resnet = models.resnet18(pretrained = True)
            outsize = 512
        elif model == 'resnet50':
            model_resnet = models.resnet50(pretrained = True)
            outsize = 2048
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(outsize, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

class Classifier(nn.Module):
    def __init__(self, num_cls):
        super(Classifier, self).__init__()
        self.num_cls = num_cls
        self.classifier_layer = nn.Linear(512, num_cls)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        if self.num_cls == 1:
            classifier_out = self.sigmoid(classifier_out)
        return classifier_out
class DG_model(nn.Module):
    def __init__(self,model = 'resnet18', num_cls = 2):
        super(DG_model, self).__init__()
        self.backbone = Feature_Generator_ResNet(model)
        self.embedder = Feature_Embedder_ResNet(model)
        self.classifier = Classifier(num_cls)

    def forward(self, input, norm_flag = True):
        feature = self.backbone(input)
        feature = self.embedder(feature, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature

class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

class DomainClassifier(nn.Module):
    
    def __init__(self, num_cls = 1):
        super(DomainClassifier, self).__init__()
        self.num_cls = num_cls
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, num_cls),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        y = self.layer(h)
        if self.num_cls == 1:
            return self.sigmoid(y)
        return y
class Discriminator(nn.Module):
    def __init__(self, num_cls = 2):
        super(Discriminator, self).__init__()
        self.num_cls = num_cls
        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, num_cls)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()
        self.sig = nn.Sigmoid()
    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        if self.num_cls == 1:
            return self.sig(adversarial_out)
        return adversarial_out 
if __name__ == '__main__': 
    x = Variable(torch.ones(1, 3, 256, 256))
    model = DG_model()
    y, v = model(x, True)
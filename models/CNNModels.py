import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt

import torch
import torchvision
from torchvision.models import resnet50, resnet152, densenet121, densenet169, vgg16_bn, mobilenet_v2
from torch import nn
from torchsummary import summary



class IndetifyLayer(nn.Module):
    """恒等映射层"""
    def __init__(self):
        super(IndetifyLayer, self).__init__()
    def forward(self, x):
        return x


def changeClasses(model:torchvision.models, classes=2, verbose=True):
    """
    将预训练的模型修改为指定类别的网络，但不修改模型结构
    :param model: 需要修改的结构，推(bi)荐(xu)为预训练的模型
    :param classes: 所需要的类别数
    :param verbose: 是否查看网络结构
    :return: 修改后的网络结构
    """
    try:  # 有时候最后一层就是Linear
        in_features = list(model.children())[-1].in_features
        supplement_layer = nn.Linear(in_features, classes)
        last_layer_name = list(model._modules.keys())[-1]
        setattr(model, last_layer_name, supplement_layer)
        if verbose:
            try:
                summary(model, (3, 224, 224))
            except:
                summary(model.cuda(), (3, 224, 224))
        return model
    except:  # 有时候最后一层是Sequential
        in_features = list(model.children())[-1][-1].in_features
        supplement_layer = nn.Linear(in_features, classes)
        last_layer_name = list(model._modules.keys())[-1]
        target_layer = getattr(model, last_layer_name)
        target_layer[-1] = supplement_layer
        setattr(model, last_layer_name, target_layer)
        if verbose:
            print("Last Layer Is Sequential!")
            print("-"*78)
            try:
                summary(model, (3, 224, 224))
            except:
                summary(model.cuda(), (3, 224, 224))
        return model


def replaceLastLayer(model:torchvision.models, layers=IndetifyLayer(), verbose=True):
    """用指定层替换最后一层"""
    last_layer_name = list(model._modules.keys())[-1]
    setattr(model, last_layer_name, layers)
    if verbose:
        try:
            summary(model, (3, 224, 224))
        except:
            summary(model.cuda(), (3, 224, 224))
    return model


class MyNet(nn.Module):
    def __init__(self, backbone='resnet50', in_features=2049, out_features=2):
        super(MyNet, self).__init__()
        assert backbone in ['resnet50'], "BackBones Only Support ResNet50 Now"
        if backbone == 'resnet50':
            self.backbone = replaceLastLayer(resnet50(pretrained=True), verbose=False)
        self.classifier = nn.Sequential(nn.Linear(in_features, out_features),nn.Sigmoid())


    def forward(self, x):
        feature_maps = self.backbone(x[:,:-1,:,:])
        symmetry = x[:,-1,:,:][:,1,1].unsqueeze(1)
        # print("fm-->", feature_maps.shape)
        # print("sy-->", symmetry.shape)
        x = torch.cat([feature_maps, symmetry], dim=1)
        # print("x-->", x.shape)
        x = self.classifier(x)
        return x

class Mynet_MTL1(nn.Module):
    def __init__(self, backbone='resnet50', out_features=3):
        super(Mynet_MTL1, self).__init__()
        assert backbone in ['resnet50'], "BackBones Only Support ResNet50 Now"
        if backbone == 'resnet50':
            self.backbone = changeClasses(resnet50(pretrained=True), classes=out_features, verbose=False)

    def forward(self, x):
        x = self.backbone(x)
        class_hat = x[:, 0:2]
        # class_hat = torch.nn.functional.softmax(x[:, 0:2], dim=1)
        pred_hat = torch.nn.functional.relu(x[:, 2]).unsqueeze(1)
        # print("--->", class_hat, pred_hat.shape)
        x = torch.cat([class_hat, pred_hat], dim=1)
        # print("--->", x[0].shape, x[1].shape)
        return x

normal_net = changeClasses(resnet50(pretrained=True), classes=opt.classes, verbose=False)






















if __name__ == '__main__':
    # summary(vgg16_bn(), (3, 224, 224))
    # print(mobilenet_v2())
    # changeClasses(mobilenet_v2())
    # replaceLastLayer(resnet50())
    # summary(MyNet(), ((3, 224, 224), (1)))
    inp = torch.rand((5,3,224,224))
    # net = MyNet()
    # net = Mynet_MTL1()
    net = normal_net
    print(net(inp))
    print("OK")
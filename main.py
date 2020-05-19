import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt

import torch
from torch import nn
from data.BuildDataset import CustomBuildDataset, transform_example
import json
from models.CNNModels import *
from models.Losses import *
from interfaces.Train import BasicTrainer
from torch import optim
import argparse
from utils.WriteLog import log_here

parser = argparse.ArgumentParser(description="Let's Do Experiment")
parser.add_argument('-n', '--name', dest='exp_name', default='exp1', help="Experiment Name")
parser.add_argument('-m', '--model', dest='model_name', default=opt.model_name, help="Model Name")
parser.add_argument('-e', '--epoch', dest='epoch', default=opt.epoch, help="Epoches You Want To Train")
parser.add_argument('-d', '--device', dest='device', default=opt.device, help="Device")
args = parser.parse_args()










# 先验知识的json
with open(os.path.join(root, opt.prior_knowledge_josn_path)) as f:
    sym_dict = json.load(f)












# 对于img图像的回调函数处理逻辑
def img_call_back(imgpath, img):
    appendix = sym_dict[imgpath]
    appendix = torch.as_tensor(appendix)
    appendix = appendix.expand([1,img.shape[1], img.shape[2]]).float()
    return torch.cat([img, appendix], dim=0)
    # return sym_dict[imgpath]

def img_call_back1(imgpath, img):
    return img
















# 对于label的处理逻辑
def get_label(imgpath, classid):
    return classid

def get_label1(imgpath, classid):
    return torch.tensor([classid, classid])












# 得到dataset、dataloader、dataset_size一个结构体
custom_data = CustomBuildDataset(dateset_name=opt.dataset_name, classes_dict={'melanoma':0, 'nevus':1}, transform=transform_example, call_back=get_label, img_call_back=img_call_back)
custom_data = custom_data.main()
custom_data1 = CustomBuildDataset(dateset_name=opt.dataset_name, classes_dict={'melanoma':0, 'nevus':1}, transform=transform_example, call_back=get_label1, img_call_back=img_call_back1)
custom_data1 = custom_data1.main()
custom_data_norm = CustomBuildDataset(dateset_name=opt.dataset_name, classes_dict={'melanoma':0, 'nevus':1}, transform=transform_example, call_back=get_label, img_call_back=img_call_back1)
custom_data_norm = custom_data_norm.main()







# 实例化网络
mynet = MyNet()
mynet1 = Mynet_MTL1()

















def get_optimizer(op_name, model=eval(opt.model_name)):
    assert op_name in ['SGD', 'Adam'], "Other Optimizers Still Not Support!"
    if op_name == 'SGD':
        return optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    elif op_name == 'Adam':
        return optim.Adam(model.parameters(), lr=opt.lr)















class Trainer1(BasicTrainer):
    def __init__(self, exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose=True):
        super(Trainer1, self).__init__(exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose)

    def _evaluate(self, opts, labels):
        _, preds = torch.max(opts, 1)
        return torch.sum(preds == labels.data)













class Trainer2(BasicTrainer):
    def __init__(self, exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose=True):
        super(Trainer2, self).__init__(exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose)

    def _evaluate(self, opts, labels):
        class_true, reg_true = labels[:, 0], labels[:, 1]
        class_hat, reg_hat = opts[:, 0:2], opts[:, -1]
        _, preds = torch.max(class_hat, 1)
        res = torch.sum(preds == class_true)
        return res
















class Trainer3(BasicTrainer):
    """
    normal Trainer
    """
    def __init__(self, exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose=True):
        super(Trainer3, self).__init__(exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose)

    def _evaluate(self, opts, labels):
        _, preds = torch.max(opts, 1)
        res = torch.sum(preds == labels)
        return res



















class CreateExperimental(object):
    def __init__(self, exp_name=args.exp_name, model=args.model_name, epoch=args.epoch, device=args.device, verbose=True, criterion=opt.criterion, trainer=eval(opt.trainer), data_res=opt.custom_data):
        self.exp_name = exp_name
        self.model = eval(model)
        self.epoch = epoch
        self.device = device
        self.exp_root = os.path.join(root, opt.experiment_root, self.exp_name)
        self.verbose = verbose
        self.data_res = eval(data_res)
        self.criterion = criterion
        self.trainer = trainer

        if os.path.isdir(self.exp_root):
            self.log_path = os.path.join(self.exp_root, 'log.txt')
            log_here("%s exists! Start train..." % self.exp_root, path=self.log_path, ifPrint=self.verbose)
        else:
            os.mkdir(self.exp_root)
            self.log_path = os.path.join(self.exp_root, 'log.txt')
            log_here("%s created! Start train..." % os.path.isdir(self.exp_root), path=self.log_path, ifPrint=self.verbose)


    def main(self):
        ce = self.trainer(exp_name=self.exp_name, model=self.model, num_epoch=self.epoch, criterion=self.criterion, optimizer=get_optimizer(opt.optimizer_ft, model=self.model), device=self.device, ckpt_name='model.pth')
        ce(self.data_res)



if __name__ == '__main__':
    # ce = CreateExperimental(epoch=1)
    # ce.main()
    # ce = CreateExperimental(exp_name='normal_train', model='normal_net', epoch=50, criterion=torch.nn.CrossEntropyLoss(), trainer=Trainer3, data_res='custom_data_norm')
    # ce.main()
    ce = CreateExperimental(exp_name='feature_map_concat', model='mynet', epoch=50,
                            criterion=torch.nn.CrossEntropyLoss(), trainer=Trainer1, data_res='custom_data')
    ce.main()


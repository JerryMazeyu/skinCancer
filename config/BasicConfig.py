import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)

import torch
import torch.optim as optim
# from models.Losses import CrossEntropyLossWithPenalty

class Config(object):
    def __init__(self):
        # DataLoader参数
        self.classes = 2
        self.dataset_name = 'train-2c'
        self.batch_size = 16
        self.shuffle = True
        self.num_workers = 6
        self.hold_out_rate = 0.9

        # 暂时不需要
        self.log_path = 'misc/log.txt'
        self.call_back = None
        self.loss_storer_location = 'misc/losses.txt'
        self.loss_alpha = -0.1

        # 先验认知
        self.prior_knowledge_josn_path = 'misc/' + self.dataset_name + '.json'


        # 训练参数
        self.experiment_root = 'experiment'
        self.model_name = 'mynet'
        self.epoch = 25
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_ft = 'SGD' # SGD / Adam
        self.lr = 1e-6
        self.custom_data = 'custom_data1'
        self.every_lr_decay = 10
        self.trainer = 'Trainer2'
        self.backbone_pretrain = True




opt = Config()

import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from utils.WriteLog import log_here
import abc
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torchvision
from utils.ShowMeImgs import matplotlib_imshow, select_n_random
from tqdm import tqdm



class BasicTrainer(abc.ABC):
    def __init__(self, exp_name, model, num_epoch, criterion, optimizer, device, ckpt_name, verbose=True):
        """
        训练函数的抽象类，需要实现evaluate方法，然后可以训练
        :param exp_name: exp1
        :param num_epoch: 训练多少个epoch
        :param criterion: 评价指标（loss）
        :param optimizer: 优化策略
        :param device: 训练所用的设备
        :param ckpt_name: 存放模型weights的名称
        :param verbose: 是否需要输出和vis
        """
        self.exp_name = exp_name
        self.exp_root = os.path.join(root, 'experiment', exp_name)
        self.model = model
        self.num_epoch = num_epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.ckpt_name = ckpt_name
        self.verbose = verbose
        self.writer = SummaryWriter(os.path.join(root, 'runs', exp_name))
        torch.cuda.empty_cache()

    @abc.abstractmethod
    def _evaluate(self, opts, labels):
        pass

    def __call__(self, custom_data_res):
        since = time.time()
        log_path = os.path.join(self.exp_root, 'log.txt')
        ckpt_path = os.path.join(self.exp_root, self.ckpt_name)
        try:
            self.model.load_state_dict(torch.load(ckpt_path))
            log_here("load path OK!", 'info', path=log_path, ifPrint=self.verbose)
        except:
            log_here("no model.pth in %s" % ckpt_path, 'debug', path=log_path, ifPrint=self.verbose)

        self.model.to(opt.device)

        best_how_good = 0.0

        epoch_losses = []
        every_losses = []
        how_good_losses = []


        for epoch in range(self.num_epoch):
            epoch_info = 'Epoch {}/{}'.format(epoch, self.num_epoch - 1)
            blank = '-' * 10
            log_here(epoch_info, 'info', path=log_path, ifPrint=self.verbose)
            log_here(blank, 'info', path=log_path, ifPrint=self.verbose)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                how_good = 0


                for inputs, labels in tqdm(custom_data_res.dataloader[phase], ncols=60):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        every_losses.append(loss)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += float(loss.item() * inputs.size(0))
                    how_good += float(self._evaluate(outputs, labels))
                    # break



                if phase=='train':
                    lr_scheduler.StepLR(self.optimizer, step_size=opt.every_lr_decay, gamma=0.1).step()

                epoch_loss = running_loss / custom_data_res.dataset_size[phase]
                epoch_acc = how_good / custom_data_res.dataset_size[phase]

                epoch_losses.append(epoch_loss)
                how_good_losses.append(epoch_acc)

                loss_info = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
                log_here(loss_info, 'info', path=log_path, ifPrint=self.verbose)


                if phase == 'val' and epoch_acc > best_how_good:
                    best_how_good = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    prefix = 'model_' + str(epoch) + '.pth'
                    torch.save(best_model_wts, os.path.join(self.exp_root, prefix))
                    prefix_1 = ckpt_path
                    torch.save(best_model_wts, os.path.join(self.exp_root, prefix_1))
                    log_here("Model has been saved as %s!" % prefix, 'info', path=log_path, ifPrint=self.verbose)

        time_elapsed = time.time() - since
        time_info = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        val_info = 'Best val Acc: {:4f}'.format(best_how_good)
        log_here(time_info, 'info', path=log_path, ifPrint=self.verbose)
        log_here(val_info, 'info', path=log_path, ifPrint=self.verbose)# 将模型、every_loss、epoch_losses、how_good_losses画图

        if self.verbose:
            log_here("=" * 50, 'info', path=log_path, ifPrint=self.verbose)
            log_here("=" * 50, 'info', path=log_path, ifPrint=self.verbose)
            log_here("Starting Preparing Tensorboard....")
            dataiter = iter(custom_data_res.dataloader['train'])
            images, labels = next(dataiter)

            ## 注意！！！这里可能对于不同任务需要重载
            images_ = images[:, :3, :, :]
            # labels = f(labels)

            # create grid of images
            img_grid = torchvision.utils.make_grid(images_)

            # show images
            matplotlib_imshow(img_grid, one_channel=True)

            # write to tensorboard
            self.writer.add_image(self.exp_name, img_grid)
            cpu_model = self.model.cpu()
            self.writer.add_graph(cpu_model, images)
            images, labels = select_n_random(custom_data_res.dataset['train'])
            # 这里也可能需要重载
            print("May be you should find here and reload this.")
            features = images.view(-1, images.shape[1]*images.shape[2]*images.shape[3])
            self.writer.add_embedding(features, metadata=labels)
            for ind, los in enumerate(every_losses):
                self.writer.add_scalar('Every Loss', los, ind)
            self.writer.close()
            for ind, los in enumerate(epoch_losses):
                self.writer.add_scalar('Epoch Loss', los, ind)
            for ind, los in enumerate(how_good_losses):
                self.writer.add_scalar('Epoch Acc', los, ind)















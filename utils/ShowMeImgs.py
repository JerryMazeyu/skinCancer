import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt
from torch.utils.tensorboard import SummaryWriter
from data.BuildDataset import result
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision

"""
在主目录创建runs/train2c文件夹，将示例图像交互到tensorboard中
$ tensorboard --logdir "runs" 开启端口
注意三点神坑：
    1 路径名除了下划线不能有-、空格等特殊字符
    2 新版命令行不是= 而是 ""
    3 要write.close!!
还有，pytorch和tensorboard可能会有冲突，尝试运行./Debug.py即可得到答案
"""


writer = SummaryWriter(os.path.join(root, 'runs', 'train2c'))

dataloader = result.dataloader['train']
dataiter = iter(dataloader)
images, labels = next(dataiter)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))



def select_n_random(dataset, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    images = []
    labels = []
    perm = torch.randperm(len(dataset))
    for i in perm[:n]:
        images.append(dataset[i][0].unsqueeze(0))
        labels.append(dataset[i][1])
    return torch.cat(images, dim=0), labels



img_grid = torchvision.utils.make_grid(images)

matplotlib_imshow(img_grid, one_channel=False)

writer.add_image('example_imgs', img_grid)

writer.close() # 这个不能省略!! 省略了就不行了!!



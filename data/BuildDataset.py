import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt

from PIL import Image
import torch
from torchvision import datasets as td
from torchvision.datasets import ImageFolder
import argparse
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import numpy as np
from torchvision import transforms as T
import os
import cv2
import json

parser = argparse.ArgumentParser(description="Build Dataset And DataLoader")
parser.add_argument('-r', '--root', dest='root', default=opt.dataset_name, help='Dataset Root Name(str)')
parser.add_argument('-l', '--if_local', dest='if_local', default=True, type=bool, help="If Use Local File To Build Dataset")
parser.add_argument('-tv', '--if_train_val', dest='if_train_val', default=True, type=bool, help="If Build Train And Valid Dataset(recommend True)")
args = parser.parse_args()



def __gray2RGB(img):
    img = np.array(img)
    print("shape is:", img.shape)
    if len(img.shape)==1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        return img

transform_example = T.Compose([
    # T.Lambda(lambda img: __gray2RGB(img)),
    T.Resize(224),  # Resize只有一个参数!!!!
    T.ToTensor(),
    # T.Compose()
])

class BuildDataset(object):
    def __init__(self, if_use_local_file=True, dataset_name=opt.dataset_name, if_train_val=True, hold_out_rate=opt.hold_out_rate, transform=transform_example):
        """
        构建一个dataset和dataloader，支持本地构建和从网络上下载两种方式
        :param if_use_local_file: 是否使用本地文件，若是则需要符合image_folder格式
        :param dataset_name: 若是本地文件则传入root名称，若需要下载外部文件名则需要符合torchvision中指定的数据集
        :param if_train_val: 数据集是否是train-val格式，如果没有的话，则可以选择hold-out-rate，分割数据集 !!!若改成False将会让后面训练变的不适合
        |----train----class1-----xxx.jpg
        |           |    |-------xxx.jpg
        |           |    |   ......
        |           |
        |           |-class2-----xxx.jpg
        |
        |----val-------class1----------
        :param hold_out_rate: 如果选择if_train_val且没有按照目录指定，则利用hold_out_rate分割
        :param tranform: 传入一个字典或一个T.compose，如果没有可以指定
        :return:
            class Result:
                def __init__(self, ds, dl, dn, cn):
                    self.dataset = ds  # dict
                    self.dataloader = dl  # dict
                    self.dataset_size = dn  # dict
                    self.class_name = cn  # list
        """
        self.if_use_loacl_file = if_use_local_file
        self.dataset_name = dataset_name
        self.if_train_val = if_train_val
        self.hold_out_rate = hold_out_rate
        self.transform = transform


    def build_datasets(self):
        if self.if_use_loacl_file:
            self.dataset_name = os.path.join(root, 'data', self.dataset_name)
            assert os.path.exists(self.dataset_name), "There is no file named %s" % self.dataset_name
            def is_valid(img_path):
                """验证图像是否符合要求"""
                return (img_path.endswith('jpg') or img_path.endswith('jpeg') or img_path.endswith('png')) and os.path.getsize(img_path) > 10
            if self.if_train_val:
                if os.path.exists(os.path.join(self.dataset_name, 'train')) and os.path.exists(os.path.join(self.dataset_name, 'val')):
                    if not isinstance(self.transform, dict):
                        self.transform = {x: self.transform for x in ['train', 'val']}
                    datasets = {x: ImageFolder(os.path.join(self.dataset_name, x), self.transform[x]) for x in ['train', 'val']}
                else:
                    all_datasets = ImageFolder(self.dataset_name, transform=self.transform, is_valid_file=is_valid)
                    train_size = int(self.hold_out_rate * len(all_datasets))
                    val_size = len(all_datasets) - train_size
                    train_dataset, val_dataset = random_split(all_datasets, [train_size, val_size])
                    datasets = {'train': train_dataset, 'val': val_dataset}
                dataloaders = {x: DataLoader(datasets[x], batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers) for x in ['train', 'val']}
                dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
                try:
                    class_names = datasets['train'].classes
                except:
                    class_names = all_datasets.classes
            else:
                datasets = ImageFolder(self.dataset_name, transform=self.transform, is_valid_file=is_valid)
                dataloaders = DataLoader(datasets, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers)
                dataset_sizes = len(datasets)
                class_names = datasets.classes

        else:
            assert self.dataset_name in ['mnist', 'kmnist', 'cifa10'], "Other datasets are not supported yet, you can load it local."  # 其他的数据集暂时不支持
            if not isinstance(self.transform, dict):
                self.transform = {x: self.transform for x in ['train', 'val']}
            if self.dataset_name == 'mnist':
                train_dataset = td.MNIST(root='.', train=True, transform=self.transform['train'], download=True)
                val_dataset = td.MNIST(root='.', train=False, transform=self.transform['val'], download=True)
            elif self.dataset_name == 'kmnist':
                train_dataset = td.KMNIST(root='.', train=True, transform=self.transform['train'], download=True)
                val_dataset = td.KMNIST(root='.', train=False, transform=self.transform['val'], download=True)
            else:
                train_dataset = td.CIFAR10(root='.', train=True, transform=self.transform['train'], download=True)
                val_dataset = td.CIFAR10(root='.', train=False, transform=self.transform['val'], download=True)
            datasets = {'train':train_dataset, 'val':val_dataset}
            dataloaders = {x: DataLoader(datasets[x], batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers) for x in ['train', 'val']}
            dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
            class_names = datasets['train'].classes
        class Result:
            def __init__(self, ds, dl, dn, cn):
                self.dataset = ds
                self.dataloader = dl
                self.dataset_size = dn
                self.class_name = cn
        res = Result(datasets, dataloaders, dataset_sizes, class_names)
        return res

class CustomBuildDatasetHelper(Dataset):
    def __init__(self, img_path, *args, transform=None, call_back=lambda *args: 1, img_call_back=None):
        """
        从图像中自定义类别信息，返回一个dataset
        :param img_path: imgs的路径
        :param transform: 对图像做的处理
        :param call_back: 传入一个image path，返回一个label
        :param img_call_back: 在指定的情况下，data会返回图像的基础上还加入一个其他的值，(img, img_call_back(imgpath))
        :param args: 在多分类情况下，若需要对每一个类别指定一个固定的label，则可以加参数格式为一个字典, 第二个参数可以是类别或者index
        例如：
            对于数据集目录train9c/.../classA/....jpeg来说
            若我希望返回的是两个值，首先根据其class类别分成1，2，再通过图像得到其不对称度，得到一个tuple
            eg:
            def getlabel(*args):
                imgpath, class_index = args[0], args[1]
                return (imgpath, classdict)
            cbdh = CutomBuildDatasetHelper('/Users/.../train/classA', 1, 'classB', call_back=getlabel)

        """
        def is_valid(imgp):
            """验证图像是否符合要求"""
            return (imgp.endswith('jpg') or imgp.endswith('jpeg') or imgp.endswith('png')) and os.path.getsize(imgp) > 10
        self.imgs = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        self.imgs = list(filter(is_valid, self.imgs))
        self.transform = transform
        self.call_back = call_back
        self.img_call_back = img_call_back
        if args != ():
            self.args = args
        else:
            self.args = None

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.call_back(img_path, *self.args)
        pil_img = Image.open(img_path)
        if self.transform:
            data = self.transform(pil_img)
        else:
            array = np.asarray(pil_img)
            data = torch.from_numpy(array)
        if self.img_call_back != None:
            # data = (data, self.img_call_back(img_path))
            data = self.img_call_back(img_path, data)
        return data, label

    def __len__(self):
        return len(self.imgs)

class CustomBuildDataset(object):
    def __init__(self, dateset_name, classes_dict=None, transform=None, call_back=lambda *args: 1, img_call_back=None, if_train_valid=True, hold_out_rate=opt.hold_out_rate):
        """
        构建一个自己的数据集合
        :param dateset_name: 数据集文件夹名称
        :param classes_dict: 类的hash 例如：{'classA':0, 'classB':1}，key需要和类文件夹对应
        :param transform: 需要的处理（暂时只支持一个callback而不是一个字典）
        :param call_back: 这里call_back指定了从图像得到label的逻辑，传递两个参数，第一个是图像的路径，第二个是从class_dict拿到的对应value，第二个参数为可选，
                          若class_dict为None，则call_back也不应写参数
        :param if_train_valid:
        :param hold_out_rate:
        """
        self.root = os.path.join(root, 'data', dateset_name)
        if os.path.exists(os.path.join(self.root, 'train')) and os.path.exists(os.path.join(self.root, 'val')):
            self.flag = True
            self.classes = next(os.walk(os.path.join(self.root, 'train')))[1]
            self.traintargetroot = {x:os.path.join(self.root, 'train', x) for x in self.classes}  # 防止 train/val 不同
            valclasses = next(os.walk(os.path.join(self.root, 'train')))[1]
            self.valtargetroot = {x:os.path.join(self.root, 'train', x) for x in valclasses}
        else:
            self.flag = False
            self.classes = next(os.walk(self.root))[1]
            self.targetroot = {x:os.path.join(self.root, x) for x in self.classes}

        self.classes_dict = classes_dict
        self.transform = transform
        self.call_back = call_back
        self.img_call_back = img_call_back
        self.if_train_val = if_train_valid
        self.hold_out_rate = hold_out_rate

    def main(self):
        if self.flag:
            train_datasets_list = []
            for key, tar_root in self.traintargetroot.items():
                cbdh = CustomBuildDatasetHelper(tar_root, self.classes_dict[key], transform=self.transform, call_back=self.call_back, img_call_back=self.img_call_back)
                train_datasets_list.append(cbdh)
            val_datasets_list = []
            for key, tar_root in self.valtargetroot.items():
                cbdh = CustomBuildDatasetHelper(tar_root, self.classes_dict[key], transform=self.transform, call_back=self.call_back, img_call_back=self.img_call_back)
                val_datasets_list.append(cbdh)
            datasets = {'train': ConcatDataset(train_datasets_list), 'val':ConcatDataset(val_datasets_list)}
        else:
            datasets_list = []
            for key, tar_root in self.targetroot.items():
                cbdh = CustomBuildDatasetHelper(tar_root, self.classes_dict[key], transform=self.transform, call_back=self.call_back, img_call_back=self.img_call_back)
                datasets_list.append(cbdh)
            datasets_all = ConcatDataset(datasets_list)
            train_size = int(len(datasets_all) * self.hold_out_rate)
            val_size = len(datasets_all) - train_size
            train_datasets, val_datasets = random_split(datasets_all, [train_size, val_size])
            datasets = {'train': train_datasets, 'val': val_datasets}


        dataloaders = {x: DataLoader(datasets[x], batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
        class Result:
            def __init__(self, ds, dl, dn, cn):
                self.dataset = ds
                self.dataloader = dl
                self.dataset_size = dn
                self.class_name = cn
        res = Result(datasets, dataloaders, dataset_sizes, self.classes)
        return res


build = BuildDataset()
result = build.build_datasets()



if __name__ == '__main__':
    # pass
    # build_dataset = BuildDataset(if_use_local_file=args.if_local, dataset_name='tst', if_train_val=True)
    # res = build_dataset.build_datasets()
    # dataloader = res.dataloader['val']
    # for index, (img, label) in enumerate(dataloader):
    #     print(img.shape, label)
    # print("Load Dataset OK!")
    # def getlabel(*args):
    #     imgpath, c = args[0], args[1]
    #     return (1, c)
    #
    # c = CustomBuildDatasetHelper('/Users/mazeyu/newEraFrom2020.5/skinCancer/data/train-2c/melanoma',1, call_back=getlabel)
    #
    # print(c[0][0].shape, c[0][1])

    def getlabel(imgpath, classid):
        return (classid, imgpath)

    from cv.ProcessImage import ProcessImage
    trim = ProcessImage()

    with open(os.path.join(root ,opt.prior_knowledge_josn_path)) as f:
        sym_dict = json.load(f)

    def foo(imgpath, img):
        appendix = sym_dict[imgpath]
        appendix = torch.as_tensor(appendix)
        appendix = appendix.expand([1, img.shape[1], img.shape[2]]).float()
        return torch.cat([img, appendix], dim=0)


    cbd = CustomBuildDataset('train-2c', {'melanoma':0, 'nevus':1}, transform=transform_example, call_back=getlabel, img_call_back=foo)
    res = cbd.main()
    print(res.dataset['train'][0][1])
    print("+"*78)
    print(res.dataset['train'][:100])
    # for (data, label) in res.dataloader['train']:
    #     print(data)
    #     break







#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-02-20 16:22:17
@LastEditTime: 2019-02-20 18:14:57
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-02-20 16:22:17
@LastEditTime: 2019-02-20 16:42:42
'''
# coding=utf-8

"""
cite from https://blog.csdn.net/u012609509/article/details/81264687
"""
"""
torch.utils.data.Dataset 是一个表示数据集的抽象类.
你自己的数据集一般应该继承``Dataset``, 并且重写下面的方法:
    1. __len__ 使用``len(dataset)`` 可以返回数据集的大小
    2. __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第i个样本(0索引)
"""


"""
torch.utils.data中的DataLoader提供为Dataset类对象提供了:
    1.批量读取数据
    2.打乱数据顺序
    3.使用multiprocessing并行加载数据
    
    DataLoader中的一个参数collate_fn：可以使用它来指定如何精确地读取一批样本，
     merges a list of samples to form a mini-batch.
    然而，默认情况下collate_fn在大部分情况下都表现很好
"""


class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        return len(self.img_frame)

    def __getitem__(self, idx):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回第 idx 个图像及相关信息
        :param idx:
        :return:
        """
        img_name = os.path.join(
            self.root_dir, self.img_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.img_frame.iloc[idx, 1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def t_dataset():
    """
    测试 MyDataset 类的使用
    :return: 
    """
    # 实列化 MyDataset 类
    img_dataset = MyDataset(
        csv_file='train_labels.csv', root_dir='train')

    dataloader = DataLoader(img_dataset,
                            batch_size=1, shuffle=True, num_workers=2)

    # 对dataloader对象进行迭代，读取数据
    for i_batch, sample_batched in enumerate(dataloader):
        image_batch, label_batch = sample_batched['image'], sample_batched['label']
        print('i_batch: {}, image_batch.size(): {}, label_batch.size(): {}'.format(
            i_batch, image_batch.size(), label_batch.size()))


"""Transform操作"""


class RandomCrop(object):
    """随机裁剪图片
    Args:
        output_size (tuple or int): 期望输出的尺寸, 如果是int类型, 裁切成正方形.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        # 返回值实际上也是一个sample
        return {'image': image, 'label': label}


class ToTensor(object):
    """
    将 ndarray 的样本转化为 Tensor 的样本
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 交换轴，因为 numpy 图片：H x W x C, torch输入图片要求： C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': torch.FloatTensor([label])}


def t_dataloader():
    transformed_dataset = MyDataset(csv_file='train_labels.csv', root_dir='train',
                                    transform=transforms.Compose([
                                        RandomCrop(25),
                                        ToTensor()]))
    dataloader = DataLoader(transformed_dataset,
                            batch_size=1, shuffle=True, num_workers=2)

    # 对dataloader对象进行迭代，读取数据
    for i_batch, sample_batched in enumerate(dataloader):
        image_batch, label_batch = sample_batched['image'], sample_batched['label']
        print('i_batch: {}, image_batch.size(): {}, label_batch.size(): {}'.format(
            i_batch, image_batch.size(), label_batch.size()))


if __name__ == '__main__':

    # t_dataset()

    t_dataloader()

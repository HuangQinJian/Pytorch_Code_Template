#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-02-20 16:22:17
@LastEditTime: 2019-03-04 18:46:04
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils


"""
cite from https://blog.csdn.net/u012609509/article/details/81264687
cite from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
refer from https://blog.csdn.net/tsq292978891/article/details/78767326?utm_source=blogxgwz2
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
        img_path = os.path.join(
            self.root_dir, self.img_frame.iloc[idx, 0])
        # image = io.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        # image.show()
        label = self.img_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample


def get_train_valid_loader(root_dir, csv_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
    )

    if augment:
        train_transform = transforms.Compose([transforms.RandomRotation(20),
                                              transforms.ToTensor(), normalize])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(), normalize])

    # load the dataset
    train_dataset = MyDataset(csv_file=csv_dir, root_dir=root_dir,
                              transform=train_transform)

    valid_dataset = MyDataset(csv_file=csv_dir, root_dir=root_dir,
                              transform=train_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


if __name__ == '__main__':
    (train_loader, valid_loader) = get_train_valid_loader(root_dir='train',
                                                          csv_dir='train_labels.csv', batch_size=2, augment=False, random_seed=10, valid_size=0.25)
    # 对dataloader对象进行迭代，读取数据
    for i_batch, sample_batched in enumerate(train_loader):
        image_batch, label_batch = sample_batched['image'], sample_batched['label']
        print('i_batch: {}, image_batch.size(): {}, label_batch.size(): {}'.format(
            i_batch, image_batch.size(), label_batch.size()))
        # print(label_batch)
    print('=================================================================================================')
    for i_batch, sample_batched in enumerate(valid_loader):
        image_batch, label_batch = sample_batched['image'], sample_batched['label']
        print('i_batch: {}, image_batch.size(): {}, label_batch.size(): {}'.format(
            i_batch, image_batch.size(), label_batch.size()))
        # print(label_batch)

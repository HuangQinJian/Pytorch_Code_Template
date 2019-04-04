#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-04-04 13:14:01
@LastEditTime: 2019-04-04 16:13:45
'''
# 导入必要模块
import torch
import torch.nn as nn
from torchvision import models

# 读取pytorch自带的resnet-101模型,因为使用了预训练模型，所以会自动下载模型参数
model = models.resnet101(pretrained=True)

# 对于模型的每个权重，使其不进行反向传播，即固定参数
for param in model.parameters():
    param.requires_grad = False
    
# 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
for param in model.fc.parameters():
    param.requires_grad = True

# 修改最后一层
class_num = 200  # 假设要分类数目是200
channel_in = model.fc.in_features  # 获取fc层的输入通道数
# 然后把resnet-101的fc层替换成300类别的fc层
model.fc = nn.Linear(channel_in, class_num)

"""
这个时候是如果按常规训练模型的方法直接使用optimizer的话会出错误的,因为optimizer的输入参数parameters必须都是可以修改、反向传播的，
即requires_grad=True,但是我们刚才已经固定了除了最后一层的所有参数，所以会出错。
"""

# filter()函数过滤掉parameters中requires_grad=Fasle的参数

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),  # 重要的是这一句
    lr=0.1)

for child in model.children():
    print(child)  # 打印网络模型的卷积方式
    for param in child.parameters():  # 打印权重数值
        print(param)

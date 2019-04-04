#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-04-04 12:42:41
@LastEditTime: 2019-04-04 12:53:27
'''

# 首先要引入相关的包
import torch
# 引入torch.nn并指定别名
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

"""
nn.functional，这个包中包含了神经网络中使用的一些常用函数，这些函数的特点是，不具有可学习的参数(如ReLU，pool，DropOut等)，这些函数可以放在构造函数中，
也可以不放，但是这里建议不放。
"""

# 打印一下版本
print(torch.__version__)


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层，输入1350个特征，输出10个特征
        self.fc1 = nn.Linear(1350, 10)  # 这里的1350是如何计算的呢？这就要看后面的forward函数

    # 正向传播
    def forward(self, x):
        # print(x.size())  # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)  # 根据卷积的尺寸计算公式，计算结果是30。
        x = F.relu(x)
        # print(x.size())  # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
        x = F.relu(x)
        # print(x.size())  # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        # print(x.size())  # 这里就是fc1层的的输入1350
        x = self.fc1(x)
        return x


net = Net()
# print(net)

"""
注意:torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch。也就是说，就算我们输入一个样本，也会对样本进行分批，
所以，所有的输入都会增加一个维度，我们对比下刚才的input,nn中定义为3维,但是我们人工创建时多增加了一个维度，变为了4维,最前面的1即为batch-size
"""

input = torch.randn(1, 1, 32, 32)  # 这里的对应前面forward的输入是32
out = net(input)
# print(out.size())

net.zero_grad()  # 在反向传播前，先要将所有参数的梯度清零
out.backward(torch.ones(1, 10))  # 反向传播的实现是PyTorch自动实现的，我们只要调用这个函数即可


y = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
# loss是个scalar，我们可以直接用item获取到它的python类型的数值
# print(loss.item())


out = net(input)  # 这里调用的时候会打印出我们在forword函数中打印的x的大小
criterion = nn.MSELoss()
loss = criterion(out, y)
# 新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()
loss.backward()

# 更新参数
optimizer.step()

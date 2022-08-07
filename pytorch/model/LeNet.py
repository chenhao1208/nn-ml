from torch import nn


# 整个卷积神经网络的开山之作
# 一共有七层，其中两层卷积和两层池化层交替出现，最后输出三层全连接层得到整体的结果
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5, padding=1))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # 自动调整-1所处维度上的元素个数，以保证元素的总数不变
        # 这个-1指的是不知道多少列的情况下
        # 根据原来Tensor内容和Tensor的大小自动分配列数
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x
# LeNet网络层数很浅，而且没有添加激活层

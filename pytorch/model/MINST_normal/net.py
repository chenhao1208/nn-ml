from torch import nn


# 下面是三层全连接神经网络的定义
class simpleNet(nn.Module):
    # 输入的维度、第一层网络的神经元个数、第二层网络神经元的个数、以及第三层网络的
    # 神经元的个数
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        # 注意这里的形式，不能加分号
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# # 一个简单的卷积神经网络
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         layer1 = nn.Sequential()
#         # 下面来讲解这个网络是什么，这是一个卷积模块
#         # 第一个参数表示输入数据体的深度，第二个参数表示输出数据体的深度
#         # 第三个参数是滤波器(卷积核的大小)，第四个参数表示滑动的步长
#         # 第五个参数表示四周进行零填充，零的个数
#         # nn.Conv2d(3,32,3,1,padding=1)
#         layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))
#         # 激活层
#         layer1.add_module('relu1', nn.ReLU(True))
#         # 这个是最大值池化层，2×2的感受野作为池化层
#         layer1.add_module('pool1', nn.MaxPool2d(2, 2))
#         self.layer1 = layer1
#
#         layer2 = nn.Sequential()
#         layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
#         # 与上同理
#         layer2.add_module('relu2', nn.ReLU(True))
#         layer2.add_module('pool2', nn.MaxPool2d(2, 2))
#         self.layer2 = layer2
#
#         layer3 = nn.Sequential()
#         layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
#         # 与上同理
#         layer3.add_module('relu3', nn.ReLU(True))
#         layer3.add_module('pool3', nn.MaxPool2d(2, 2))
#         self.layer3 = layer3
#
#         layer4 = nn.Sequential()
#         # 全连接层
#         layer4.add_module('fc1', nn.Linear(2048, 512))
#         layer4.add_module('fc_relu1', nn.ReLU(True))
#         layer4.add_module('fc2', nn.Linear(512, 64))
#         layer4.add_module('fc_relu2', nn.ReLU(True))
#         layer4.add_module('fc3', nn.Linear(64, 10))
#         self.layer4 = layer4
#
#     def forward(self, x):
#         conv1 = self.layer1(x)
#         conv2 = self.layer2(conv1)
#         conv3 = self.layer3(conv2)
#         fc_input = conv3.view(conv3.size(0), -1)
#         fc_out = self.layer4(fc_input)
#         return fc_out


# 以下是另一种写法
# class simpleNet(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         i, self.layer = 1, nn.Sequential()
#         for h_dim in hidden_dim:
#             self.layer.add_module('layer_{}'.format(i), nn.Linear(in_dim, h_dim))
#             i, in_dim = i + 1, h_dim
#         self.layer.add_module('layer_{}'.format(i), nn.Linear(in_dim, out_dim))
#         self.layerNum = i
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x

# 添加激活函数,只需要在每层网络的输出部分添加激活函数就可以了，用到nn.Sequential()
# 这个函数是将网络的层组合到一起，最后一层输出层不能添加激活函数，因为输出的结果表示实际的得分
class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 加快收敛速度的方法——批标准化,同样使用nn.Sequential()将nn.BatchNormld()，
# 组合到网络中，一般放在全连接层的后面、非线性层的前面
class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

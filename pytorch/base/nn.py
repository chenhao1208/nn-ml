# 在Pytorch中编写神经网络，所有的层结构和损失函数都来自与torch.nn，所有的模型构建
# 都是从这个基类nn.Module继承的,下面就是模板
from torch import nn


# nn.Module详细内容的可以看网址:https://zhuanlan.zhihu.com/p/340453841
# net_name 表示神经网络的名字
class net_name(nn.Module):
    def __init__(self, other_arguments):
        # 继承nn.Module的神经网络模块在实现自己的__init__函数时，一定要先调用super().__init__()
        # 只有这样才能正确地初始化自定义的神经网络模块，否则会缺少上面的代码中的成员变量而导致模块
        # 被调用时出错
        super(net_name, self).__init__()
        # 一般有一个基类来定义接口，通过继承来处理不同维度的input
        # 1.Conv1d 等继承自_ConvNd
        # 2.MaxPool1d继承自_MaxPoolNd

        # 每一个类都有一个对应的nn.functional函数，类定义了所需要的arguments和模块的parameters
        # 在forward函数中将上述两个传给nn.functional的对应函数来实现forward功能

        # 继承nn.Module的模块主要重载init、forward和extra_repr函数
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        # other network layer

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        return x

    # 以上的过程就建立了一个计算图，这个结构可以不断服用，每次调用就相当于用该计算图定义
    # 相同参数做一次前向传播，这得益于PyTorch的自动求导功能，所以我们不需要自己编写反向传播
    # 所有的网络层都是由这个nn这个包得到的，比如线性层nn.Linear等

    # 在定义完模型之后，需要通过nn这个包来定义损失函数，常见的损失函数已经定义在nn中
    # 比如均方误差、多分类的交叉熵、以及二分类的交叉熵等等，调用这些已经定义好的损失函数也很简单

    criterion = nn.CrossEntropyLoss()
    # 这里的损失函数是抽象，没有定义具体的输出和目标
    # loss = criterion(output, target)

# 训练与测试
# nn.Module通过self.training来区分训练和测试两种状态，使得模块可以在训练和测试时有不同的
# 前向传播forward行为(Batch Normalization)
# parameter参数和tensor是一样的，但是写法是不一样的

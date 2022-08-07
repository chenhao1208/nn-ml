# 卷积神经网络包括了卷积层、池化层和全连接层
# 下面一个简单的CNN网络的例子
from torch import nn
from torch.nn import init


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        layer1 = nn.Sequential()
        # 下面来讲解这个网络是什么，这是一个卷积模块
        # 第一个参数表示输入数据体的深度，第二个参数表示输出数据体的深度
        # 第三个参数是滤波器(卷积核的大小)，第四个参数表示滑动的步长
        # 第五个参数表示四周进行零填充，零的个数
        # nn.Conv2d(3,32,3,1,padding=1)
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))
        # 激活层
        layer1.add_module('relu1', nn.ReLU(True))
        # 这个是最大值池化层，2×2的感受野作为池化层
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
        # 与上同理
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        # 与上同理
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        # 全连接层
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out


model = SimpleCNN()
print(model)

# 提取模型的前面两层
new_model = nn.Sequential(*list(model.children())[:2])
print(new_model)

# 提取出模型中所有的卷积层,下面这个代码明明和源代码一样，但是不能运行，查询了相关的
# 资料后，怀疑可能是版本太高的原因
# conv_model = nn.Sequential()
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         conv_model.add_module(name, module)
#
# print(conv_model)

print('-------------------------')
# 输出全部参数的迭代器
for param in model.named_parameters():
    print(param[0])

# 对权重进行初始化，因为权重是一个Variable，所以需要取出其中的data属性，然后对其
# 进行所需要的处理即可
for m in model.modules():
    # 检查m是否为卷积神经网络
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight.data)
        init.xavier_normal(m.weight.data)
        init.kaiming_normal(m.weight.data)
        # 用零填充m的数据
        m.bias.data.fill_(0)
    # 检查m是否为全连接层
    elif isinstance(m, nn.Linear):
        # 用正态分布填充m的数据
        m.weight.data.normal_()

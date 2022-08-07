# 对于一般的线性回归，函数拟合出来的是一条直线，精度不足，我们可以考虑多项式回归
# 原理和之前的线性回归是一样的，但是这里使用的是高次多项式而不是简单的一次线性多项式

# 我们首先拟合一个一元三次方式，设置参数方程 y = b + w1*x + w2*x^2 + w3*x^3
# 首先要明白多维线性回归的过程，x有d个属性，试图学得最优的函数f(x)
# 并且使得最小二乘最小，同样的可以利用最小二乘法对w和b进行估计，为了方便计算，可以将w
# 和b写进同一个矩阵，将数据集D表示成一个m*(d+1)的矩阵X，每行前面d个元素表示d个属性值
# 最后一个元素设为1，那么可以得到，需要求的参数有d+1个，可以把其看成为(d+1)×1的列向量

# 预处理数据，使得数据变成一个矩阵的形式
import torch
from torch.autograd import Variable
from torch import nn, optim


# 注意矩阵的形式是一个 m × (d + 1) 的矩阵


def make_features(x):
    # 本身这个数据就是有n个，所以可以扩展为n × (3+1)的矩阵
    """创建行矩阵[x, x^2, x^3]"""
    # 理解这个unsqueeze(1)，表示在哪个地方增加一个维度，原来是含有三个元素的数组
    # 现在的维度是[3,1],这个矩阵的每个元素都是一个x
    x = x.unsqueeze(1)
    # 理解torch.cat，在维度1上进行合并，也就是说，保持行数不发生改变
    # 通过这样的方式得到3×3的矩阵
    return torch.cat([x ** i for i in range(1, 4)], 1)


# 然后定义好真实的数据，想要拟合的方程，希望每一个参数都能学习到和真实参数很接近的结果
W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    """拟合方程"""
    # 这个表示做矩阵乘法，f函数就是每次输入一个x得到一个y的真实函数

    return x.mm(W_target) + b_target[0]


# 这里的权重已经定义好了，这里不是相同的权重
# unsqueeze(1)是将原来的tensor大小由3变成(3,1),x.mm(W_target)表示做矩阵乘法
# f(x)就是每次输入一个x得到一个y的真实函数

# 进行训练的时候需要采集一些点，每次取得这么多的数据点，将其转换为矩阵的形式
# 再把这个值通过函数之后的结果也返回作为真实的目标
def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)
    # 决定是否使用Variable


# 下面定义多项式函数
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)
        # 同样调用了里面的Linear线性函数包

    def forward(self, x):
        out = self.poly(x)
        return out


# 模型输入是3维，输出是一维
if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
# 使用均方误差来衡量模型的好坏，用随机梯度下降来优化模型

epoch = 0
while True:
    # 得到数据
    batch_x, batch_y = get_batch()
    # 前向传播
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.item()
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    optimizer.step()
    epoch += 1

    if print_loss < 1e-4:
        break
    elif (epoch + 1) % 20 == 0:
        # 为什么最后一次的loss.item()的值反而增加了呢
        print('epoch', epoch, 'loss', loss.item())
    # 希望能够不断地优化，知道实现设立的条件，取出的点的均方误差能够小于0.001
# 可以看到参数已经足够接近目标参数了
for name, parameters in model.named_parameters():
    print(name, ':', parameters.data)

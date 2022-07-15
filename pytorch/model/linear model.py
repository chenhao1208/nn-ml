# 先从机器学习最简单的线性模型入手，看pytorch如何解决这个问题
# 简单的、特殊的、类似的、一般的
# 给定很多个数据点，希望能找到一个函数来拟合这些数据点，使得误差最小
# 给出一系列的点，找一条直线，使得直线尽可能与这些点接近，也就是这些点到直线的距离之和尽可能小
# f(x) = w1*x1 + w2*x2 + ... +wd*xd +b
# 一般可以通过向量的内积来进行表示，这里w和b都是需要学习的参数

# 先从最简单的一维线性回归入手
# 给定数据集 D = {(x1,y1),(x2,y2),(x3,y3),...,(xm,ym)},线性回归希望优化出一个好的
# 函数，使得函数与yi尽可能的接近
# 可以使用它们之间的距离差异来衡量误差，取平方是因为距离有正有负，希望能将其全部变为正的
# 这就是均方误差，希望找到w*和y*使得均方误差最小，这个方法也被称为最小二乘法
# 这里的w 和 b 都是可以被解出来的

# 多维线性回归，有d个属性，试图学得最优的函数f(x),情况是类似的

# 一维线性回归的代码实现,先随便给出一些点

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 把这些点画出来
# plt.plot(x_train, y_train)
# plt.show()

# 先将numpy.array转换为Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# 接着开始建立模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出都是一维的

    # 下面的前向传播就是一维的，具体模型已经上面已经揭示了，这里直接调用包就解决这个问题了
    def forward(self, x):
        out = self.linear(x)
        return out

    # 下面开始计算损失函数


if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# 这就是一个超级简单的模型，y = wx + b，输入参数是一维，输出参数也是一维
# 然后开始定义损失函数和优化函数，这里使用均方误差作为优化函数，使用梯度下降进行优化
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 然后就可以开始训练我们的模型了
num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)
    # 前向传播
    out = model(inputs)
    loss = criterion(out, target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('epoch', epoch, 'loss', loss.item())

# 定义好要跑的epoch个数，然后将数据变成Variable放入计算图中，然后通过out = model(inputs)得到
# 网络前向传播得到的结果，通过loss = criterion(out,target)得到损失函数，然后归零梯度，做反向
# 传播和更新参数，每次做反向传播之前都要归零梯度，optimizer.zero_grad()。在训练的过程中，隔一段时间
# 就将损失函数的值打印出来看看，确保我们的模型误差越来越小

# 做完训练之后可以预测一下结果
model.eval()
model.cpu()
predict = model(Variable(x_train))
# 这里要加上一个(),书上是加了的，但是你还是忽略了这个问题，导致了这个错误的产生
predict = predict.data.numpy()
# 这里的plot的用法和matlab中有异曲同工之妙
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()

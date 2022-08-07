# 在机器学习中，监督学习主要分为回归问题和分类问题
# 回归问题中，希望预测的结果是连续的，则分类问题中所预测的结果就是离散的类别
# 输入变量可以是离散的，也可以是连续的，从数据中学习一个分类模型或者分类决策函数，称为分类器
# 先从classify data.txt文件中读取数据，使用非常简单的读取方法
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.autograd import Variable

# r模式是默认的
with open('classify data.txt', 'r') as f:
    data_list = f.readlines()
    # 先分成一行一行的数组
    data_list = [i.split('\n')[0] for i in data_list]
    # 每一行的数组又分为0，1，2的下标
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
# 首先把第i行的数据存储，然后第i行中的每个属性进行存储分类

# 下面表示最后一个属性是0或者1，表示分为两种类别,查看数据
x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))
plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]
plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]
plt.plot(plot_x0_0, plot_x0_1, 'ro', label='label_0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label='label_1')
plt.legend(loc='best')
# 分为红蓝两个部分，定义二分类的损失函数和优化方法
plt.show()

# 获得训练数据
with open('classify data.txt') as f:
    data = f.read().split('\n')
    data = [row.split(',') for row in data][:-1]
    label0 = np.array([(float(row[0]), float(row[1])) for row in data if row[2] == '0'])
    label1 = np.array([(float(row[0]), float(row[1])) for row in data if row[2] == '1'])
x = np.concatenate((label0, label1), axis=0)
x_data = torch.from_numpy(x).float()

y = [[0] for i in range(label0.shape[0])]
y += [[1] for i in range(label1.shape[0])]
y_data = torch.FloatTensor(y)


# 下面是定义模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        # 这个参数表示输入和输出，就是说，输入两个参数，输出两个参数
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        # 这里就相当于将多项式的结果再输入到逻辑回归函数中
        x = self.sm(x)
        return x


# 创建一个模型实例
logistic_model = LogisticRegression()
if torch.cuda.is_available():
    logistic_model.cuda()
# 同样地定义损失函数和优化函数，采取随机梯度下降优化函数，然后训练模型
# 并且带有动量0.9
criterion = nn.BCELoss()
# 这里的损失函数是二分类损失函数
optimizer = optim.SGD(logistic_model.parameters(), lr=1e-3,
                      momentum=0.9)
# 开始训练
for epoch in range(50000):
    if torch.cuda.is_available():
        x = Variable(x_data).cuda()
        y = Variable(y_data).cuda()
    else:
        x = Variable(x_data)
        y = Variable(y_data)
    # forward前向计算
    out = logistic_model(x)
    loss = criterion(out, y)

    # 计算准确率
    print_loss = loss.item()
    mask = out.ge(0.5).float()

    correct = (mask == y).sum()
    acc = correct.data.item() / x.size(0)

    # BP回溯
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))
        print('correct rate is {:.4f}'.format(acc))

w0, w1 = logistic_model.lr.weight[0]
w0 = w0.item()
w1 = w1.data.item()
b = logistic_model.lr.bias.item()
plt.plot(plot_x0_0, plot_x0_1, 'ro', label='label_0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label='label_1')
plt.legend(loc='best')
plot_x = np.arange(30, 100, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
plt.show()
# 不知道为什么有时候能够运行，有时候不能运行

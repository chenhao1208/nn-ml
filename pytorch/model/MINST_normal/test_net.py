# from torch import nn, optim
import csv

import numpy as np
import torch
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# # 这个import net就是之前定义网路的Python文件
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable

import net

#
# # 定义超参数(Hyper parameters)
# batch_size = 64
# learning_rate = 1e-2
# num_epoches = 60
#
# # 数据预处理，将数据标准化
# # 这里就是将各种预处理组合到一起
# # 因为图片是灰度图，所以只有一个通道，如果是彩色的图片，有三通道，那么就需要使用
# # transforms.Normalize([a,b,c],[d,e,f]来表示每个通道对应的均值和方差
# data_tf = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#
# # 下载训练集MINST手写数字训练集
# # 通过Pytorch的内置函数torchvision.datasets.MNIST导入数据集，传入数据预处理
# # 接着使用torch.utils.data.DataLoader建立一个数据迭代器，传入数据集和batch_size
# # 通过shuffle = True来表示每次迭代数据的时候是否将数据打乱
# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=data_tf, download=True)
#
# test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # 接着导入网络，把结果分为十类
# model = net.Batch_Net(28 * 28, 300, 100, 10).cuda()
# model.load_state_dict(torch.load('Batch_save.pt'))
# model.eval()
# if torch.cuda.is_available():
#     model = model.cuda()
#
# # 下面是损失函数和优化器，损失函数定义为最常见的损失函数交叉熵
# # 使用随机梯度下降来优化损失函数
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#
#
# eval_loss = 0
# eval_acc = 0
# for data in test_loader:
#     img, label = data
#     img = img.view(img.size(0), -1)
#     if torch.cuda.is_available():
#         # 这个表示前向传播时不会保留缓存，因为对于测试集，不需要进行反向传播
#         # 所以可以在前向传播的时候释放掉内存，节约内存空间
#         with torch.no_grad():
#             img = Variable(img).cuda()
#             label = Variable(label).cuda()
#     else:
#         with torch.no_grad():
#             img = Variable(img)
#             label = Variable(label)
#
#     out = model(img)
#     loss = criterion(out, label)
#     # 获得测试数据
#     eval_loss += loss.item() * label.size(0)
#     _, pred = torch.max(out, 1)
#     num_correct = (pred == label).sum()
#     eval_acc += num_correct.item()
# print('Test Loss: {:.6f},Acc: {:.6f}'.format(
#     eval_loss / (len(test_dataset)),
#     eval_acc / (len(test_dataset))))
epoch_size = 700
learning_rate = 1e-2
num_epoches = 60
# 重写测试网络,现在测试的是训练好的Batch_Net
model = net.Batch_Net(28 * 28, 300, 100, 10).cuda()
model.load_state_dict(torch.load('Batch_save.pt'))
model.eval()

# 引入测试数据,测试数据没有给标签

with open('./data/train.csv') as f:
    lines = csv.reader(f)
    label, attr = [], []
    for line in lines:
        if lines.line_num == 1:
            continue
        label.append(int(line[0]))
        attr.append([float(j) for j in line[1:]])
print(len(label), len(attr[1]))
# 将数据分为40(epoches) * 700(rows)的数据集
epoches = []

for i in range(0, len(label), epoch_size):
    torch_attr = torch.FloatTensor(attr[i:i + epoch_size])
    torch_label = torch.LongTensor(label[i:i + epoch_size])
    epoches.append((torch_attr, torch_label))

# 损失函数,交叉熵
criterion = nn.CrossEntropyLoss()
# 优化函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# 测试过程
def test():
    epoch_num, loss_sum, cort_num_sum = 0, 0.0, 0
    for epoch in epoches:
        epoch_num += 1
        if torch.cuda.is_available():
            inputs = Variable(epoch[0]).cuda()
            target = Variable(epoch[1]).cuda()
        else:
            inputs = Variable(epoch[0])
            target = Variable(epoch[1])
        output = model(inputs)
        loss = criterion(output, target)
        loss_sum += loss.item()
        _, pred = torch.max(output.data, 1)
        t = epoch[1]
        t = np.array(t).astype(int)
        t = torch.from_numpy(t)
        t = t.cuda()
        num_correct = torch.eq(pred, t).sum()
        cort_num_sum += num_correct

    loss_avg = loss_sum / epoch_num
    cort_num_avg = cort_num_sum / epoch_num / epoch_size
    return loss_avg, cort_num_avg


# 对所有的测试数据跑300遍模型
loss, correct = [], []
testing_time = 300
for i in range(1, testing_time + 1):
    loss_avg, correct_num_avg = test()
    loss.append(loss_avg)
    if i < 20 or i % 20 == 0:
        print('--- test time{} ---'.format(i))
        print('average loss = {:.4f}'.format(loss_avg))
        print('average correct number = {:.4f}'.format(correct_num_avg))
    correct.append(correct_num_avg)

# 画测试过程中的损失值图像
lx = np.array(range(len(loss)))
ly = np.array(loss)
plt.title('loss of training')
plt.plot(lx, ly)
plt.show()

# 画测试过程中的正确率
cx = np.array(range(len(correct)))
cy = np.array(torch.tensor(correct, device='cpu'))
plt.title('correct rate of training')
plt.plot(cx, cy)
plt.show()

# 由于这个test.csv没有给标签，所以只能在原本的训练集进行测试，实际上测试的结果是不会改变的


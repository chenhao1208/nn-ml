import csv
import net
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 确定超参数
import torch

epoch_size = 700
learning_rate = 1e-2
num_epoches = 60
# 获得训练数据-train.csv
with open('./data/train.csv') as f:
    lines = csv.reader(f)
    label, attr = [], []
    for line in lines:
        if lines.line_num == 1:
            continue
        label.append(int(line[0]))
        attr.append([float(j) for j in line[1:]])
# print(len(label), len(attr[1]))

# 将数据分为60(epoches) * 700(rows)的数据集
epoches = []

for i in range(0, len(label), epoch_size):
    torch_attr = torch.FloatTensor(attr[i:i + epoch_size])
    torch_label = torch.LongTensor(label[i:i + epoch_size])
    epoches.append((torch_attr, torch_label))

# 模型实例化
if torch.cuda.is_available():
    # 通过改变下面的内容来改变网络的组成
    net = net.Batch_Net(28 * 28, 300, 100, 10).cuda()
else:
    net = net.Batch_Net(28 * 28, 300, 100, 10)

# 损失函数,交叉熵
criterion = nn.CrossEntropyLoss()
# 优化函数
optimizer = optim.SGD(net.parameters(), lr=learning_rate)


# 训练过程
def train():
    epoch_num, loss_sum, cort_num_sum = 0, 0.0, 0
    for epoch in epoches:
        epoch_num += 1
        if torch.cuda.is_available():
            inputs = Variable(epoch[0]).cuda()
            target = Variable(epoch[1]).cuda()
        else:
            inputs = Variable(epoch[0])
            target = Variable(epoch[1])
        output = net(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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


# 对所有的数据跑300遍模型
loss, correct = [], []
training_time = 300
for i in range(1, training_time + 1):
    loss_avg, correct_num_avg = train()
    loss.append(loss_avg)
    if i < 20 or i % 20 == 0:
        print('--- train time{} ---'.format(i))
        print('average loss = {:.4f}'.format(loss_avg))
        print('average correct number = {:.4f}'.format(correct_num_avg))
    correct.append(correct_num_avg)

# 画图输出训练过程情况


# 画训练过程中的损失值图像
lx = np.array(range(len(loss)))
ly = np.array(loss)
plt.title('loss of training')
plt.plot(lx, ly)
plt.show()

# 画训练过程中正确率变化
cx = np.array(range(len(correct)))
cy = np.array(torch.tensor(correct, device='cpu'))
plt.title('correct rate of training')
plt.plot(cx, cy)
plt.show()

# 引入测试数据

with open('./data/test.csv') as f:
    lines = csv.reader(f)
    test = []
    for line in lines:
        if lines.line_num == 1:
            continue
        test.append([float(i) for i in line])
test = torch.FloatTensor(test)
test = test.cuda()
net.eval()
# volatile = True 表示前向传播不保留缓存,预测最大值
with torch.no_grad():
    predict = net(Variable(test))
    _, predict = torch.max(predict, 1)
    predict = predict.data.cpu().numpy()

with open('./data/predict.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ImageId', 'Label'])
    for i in range(predict.shape[0]):
        result = [i + 1, predict[i]]
        writer.writerow(result)
    print('write done.')
# simpleNet的表现过于糟糕
# Score: 0.96257
# Activation_Net预测正确率为96.257%
# Score: 0.97517
# Batch_Net预测正确率为97.517%

# 保存成功
# torch.save(net.state_dict(), 'Batch_save.pt')
# 下面读取保存成功的文件
# model = net.Batch_Net(28 * 28, 300, 100, 10).cuda()
# model.load_state_dict(torch.load('Batch_save.pt'))
# model.eval()

# 在机器学习和深度学习中，需要通过修改参数使得损失函数最小化，优化算法就是一种调整参数更新的策略
# 但是在上一步的nn模块中，并没有详细的说清楚损失函数的表示，只是抽象地进行了表示

# 在有了数值优化算法的基础之后，我们再来看这个torch.optim算法
# 一阶优化算法：最速下降法
# 参数更新公式，分为学习率和函数的梯度，这是最常用的优化方法
# 二阶优化算法，使用了二阶导数，来最小化或最大化损失函数，主要基于牛顿法，但是由于二阶导数
# 计算成本很高，所以这种方法没有广泛使用

# 大多数常见的算法能够直接通过这个包来进行调用，在调用的时候将需要优化的参数传入，这些参数都必须是
# Variable，然后传入一些基本的设定，比如学习率和动量
import torch
# 这个没有具体的模型，所以就随便引入一个包的model模型
from torchaudio.models.wav2vec2 import model

# 学习率是0.01，动量是0.9的随机梯度下降，在优化之前需要先将梯度归零，即optimizer.zeros()
# 然后通过loss.backward()反向传播，自动求导得到每个参数的梯度，最后只需要optimizer.step()就可以通过梯度做一步参数更新

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

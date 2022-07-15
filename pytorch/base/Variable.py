import torch.cuda
from torch.autograd import Variable

# 下面介绍一个在numpy中没有，在神经网络计算图中特有的一个概念，Variable提供了自动求导的功能
# Variable和张量没有本质上的区别，但是Variable会被放在一个计算图中，然后进行前向传播和反向传播，并且自动求导
# 并且将一个张量a变为Variable只需写Variable(a)即可
# Variable的属性有data、grad和grad_fn，grad_fn得到的是这个variable的操作，grad得到的是这个变量的反向传播梯度

# 下面通过具体的例子进行说明,注意需要导入的包
# Variable是在torch中的autograd中的，注意其导入形式

# 创建变量，但是要导入相应的包，在创建相应的变量的时候，需要传入一个参数requires_grad = True
# 这个参数表示是否对这个变量求梯度，默认的是false，也就是不对这个变量求梯度
# 我们希望得到这些变量的梯度，所以传入这个参数
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# 构建一个计算图，这三者都是变量
y = w * x + b

# 开始计算梯度
y.backward()  # 与y.backward(torch.FloatTensor([1]))相同，这一行代码就是自动求导的意思
# 输出各个变量的梯度
print(x.grad)  # 就是y对x求偏导，得到的就是变量w，此时变量w的值为2，由此可得，x的梯度为2
print(w.grad)  # 同理为1
print(b.grad)  # 同理为1
print('------------------------------------')
# 上述过程是对于标量求导的过程，同时也可以对矩阵进行求导
# x = torch.randn(3)
x = torch.Tensor([2, 4, 8])
x = Variable(x, requires_grad=True)

y = x * 2
print(y)
# 相当于给出了一个三维向量进行运算，得到的结果y就是一个向量，这里对这个向量求导一定要有后面的内容
# 传入参数声明，这样得到的梯度就是原本的梯度分别乘上1，0.1，0.01
y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)  # 注意这里不是y = x^2，而是 y = 2*x ,所以求导得到结果为2，在每一个分量上都为2
# 对于这个问题我犯了错误，一定要注意

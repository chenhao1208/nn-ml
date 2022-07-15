import numpy as np
import torch.cuda

# 首先先学习张量Tensor
# 32位浮点型 torch.FloatTensor
# 64位浮点型 torch.DoubleTensor
# 16位整型 torch.ShortTensor
# 32位整型 torch.IntTensor
# 64位整型 torch.LongTensor

a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
# 这是一个三行两列的张量，其实就是一个向量
print('a is: {}'.format(a))
# 默认的是torch.FloatTensor数据类型
print('a size is {}'.format(a.size()))

# 当然也可以改变定义为我们想要的数据类型，如下
b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print('b is : {}'.format(b))

# 可以创建一个全是0或者取值为正态分布作为随机初始值
c = torch.zeros((3, 2))  # 表示三行两列
print('zero tensor: {}'.format(c))

d = torch.randn((3, 2))
print('normal randon is :{}'.format(d))

# 同样的可以通过数组索引的方式取得其中的元素或者改变其中的元素
a[0, 1] = 100

print('changed a is: {}'.format(a))
# print(torch.cuda.is_available())，这句话可以检查是否可以使用GPU

# 张量和numpy.ndarray之间相互转换
numpy_b = b.numpy()
print('converse to numpy is \n {}'.format(numpy_b))  # 这是张量向numpy数组的转换

e = np.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)  # 这是numpy数组向张量的转换，转换的形式位int类型
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()  # 把这个张量的数据形式转变为float张量的形式
print('change data type to float tensor: {}'.format(f_torch_e))

# 下面就是把张量a放到GPU上
if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)

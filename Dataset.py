# 要首先引入代表一类数据的抽象类，如下，引入之后，可以自己定义自己的数据
# 继承和重写这个抽象类，只需要定义_len_和_getitem_这两个函数

from turtle import pd

from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


# 在处理任何机器学习问题之前，都要完成对数据的读取，下面我们对这个问题进行一定的学习
# torch.utils.data.Dataset是这一数据的抽象类，可以定义自己的数据类继承和重写这个抽象类
# 这里要先学习一下python中关于类的知识




class Dog:
    """一次模拟小狗的简单尝试"""

    def __init__(self, name, age):  # 每次调用这个方法来创建狗的实例时都会自动调用这个方法
        """初始化属性name和age"""
        self.name = name
        self.age = age

    def sit(self):
        """模拟小狗蹲下"""
        print(f"{self.name} is now sitting.")

    def roll_over(self):
        print(f"{self.name} rolled over!")


my_dog = Dog('Willie', 6)

print(f"My dog's name is {my_dog.name}.")
print(f"My dog is {my_dog.age} years old.")
my_dog.sit()
my_dog.roll_over()


# 现在开始进行数据集内容的学习
class myDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        # 这是一个读文件的操作
        # python读取文件有几种方式，read_csv从文件,url，文件型对象中加载带分隔符的数据
        # 默认分隔符为逗号
        # 也可以写成：self.csv_data = pd.read_csv(csv_file, nrows = 10)
        # 上面注释表示只读取前十行
        self.csv_data = pd.read_csv(csv_file)

        # r参数表示以只读方式打开文件，文件的指针会放在文件的开头，这是默认模式
        # 其他模式可以参考这个博客：https://blog.csdn.net/yuyingji/article/details/118362628
        with open(txt_file, 'r') as f:
            # f.readlines()表示返回文件中行内容的列表，size参数可选
            data_list = f.readlines()
            # txt_data存放行内容的列表，root_dir存放地址
            self.txt_data = data_list
            self.root_dir = root_dir

    def __len__(self):
        # 这里应该是得到csv_file文件的行数
        return len(self.csv_data)

    def _getitem_(self, idx):
        # 返回的是两个指标，一个是x，一个是y
        data = (self.csv_data[idx], self.txt_data[idx])
        return data


"""
CSV文件的规则：

1、开头是不留空，以行为单位。

2、可含或不含列名，含列名则居文件第一行。

3、一行数据不跨行，无空行。

4、以半角逗号（即，）作分隔符，列为空也要表达其存在。

5、列内容如存在半角引号（即"），替换成半角双引号（""）转义，即用半角引号（即""）将该字段值包含起来。
"""

# 以上的方式可以定义我们需要的数据类，可以通过迭代的方式来取得每一个数据，但是这样很难实现
# 取batch，shuffle或者多线程的取数据，pytorch中使用torch.utils.data.DataLoader来定义一个新的迭代器
# 其中batch是批量取得数据，shuffle是随机采样

# 生成一个迭代器
# batch_size 32个一组，随机采样
# 以下为dataloader取数据的流程
"""
在dataloader按照batch进行取数据的时候, 是取出大小等同于batch size的index列表;
 然后将列表列表中的index输入到dataset的getitem()函数中,取出该index对应的数据; 
 最后, 对每个index对应的数据进行堆叠, 就形成了一个batch的数据.
"""
# 而在最后一步进行堆叠的时候可能出现问题，如果一条数据中所含有的每个数据元的长度不同，那么将无法进行堆叠
# 如：multi—hot类型的数据，序列数据
# 通常使用这些数据的时候，要先进行长度上的补齐，再进行堆叠，以现在的流程，不能加入该操作的
# 此外，某些优化方法是要对一个batch的数据进行操作
# collate—fn函数就是手动将抽取出来的样本堆叠起来的函数
# loader = Dataloader(dataset, batch_size, shuffle, collate_fn, ...)用法如下
# https://mp.weixin.qq.com/s/Uc2LYM6tIOY8KyxB7aQrOw 这个视频详尽地展示了DataLoader的工作原理
dataiter = DataLoader(myDataset, batch_size=32, shuffle=True,
                      collate_fn=default_collate)

# 可以看到，假设的myDataset返回两个数据项x和y，那么传入collate_fn的参数定义为data
# 则其shape为(batch_size,2,...).
# 下面就是定义collate_fn函数
"""
def collate_fn(data):
	for unit in data:
		unit_x.append(unit[0])
		unit_y.append(unit[1])
		...
	return {x: torch.tensor(unit_x),  y: torch.tensor(unit_y)}
"""
# 在使用collate_fn的使用过程中，只输入data十分不方便，需要额外的参数来传递其他变量
# 1、使用lambda函数
"""
info = args.info	# info是已经定义过的
loader = Dataloader(collate_fn=lambda x: collate_fn(x, info))
"""
# 相当于利用这个函数再定义了一个匿名函数
# 2、创建可被调用的类
"""
class collate():
	def __init__(self, *params):
		self. params = params
	
	def __call__(self, data):
		'''在这里重写collate_fn函数'''
		
collate_fn = collate(*params)
loader = Dataloader(collate_fn=collate_fn)
"""
# 总结
# collate_fn的用处：自定义数据堆叠的过程，自定义batch数据的输出形式
# 定义一个以data为输入的函数，输入输出分别与getitem函数和loader函数相对应

# 同时，在torchvision包中还有一个更高级的有关计算机视觉的数据读取类:ImageFolder,主要就是处理图片
# 且要求图片是下面的形式
# root/dog/xxx.png
# root/dog/xxy,png
# root/cat/123.png
# root.cat/asd.png
# 这就是说明，首先是路径，然后是两个文件夹把两个类给区分开，然后每个图片不同命名
# 每一个文件夹都表示一个类别

dset = ImageFolder(root='root_path', transform=None, loader=default_loader)
# transform和target_transform是图片增强，loader是图片读取的办法，读取图片的名字
# 然后通过loader将图片转换为我们需要的图片类型进入神经网络

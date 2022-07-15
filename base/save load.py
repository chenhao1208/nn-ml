# 这里主要是讲述如何保存和加载模型的结构和参数，有两种方式
# 1、保存整个模型的结构信息和参数信息，保存的对象是模型model
# 2、保存模型的参数，保存的对象是模型的状态，model.state_dict()

# 保存的方式如下：
# torch.save(model, './model.pth')
# torch.save(model.state_dict(), './model_state.pth')
# 加载模型有两种方式对应于保存模型的方式：1、加载完整的模型结构和参数信息,使用load_model =
# torch.load('model.pth')，在网络较大的时候，加载时间较长，同时存储空间也较大
# 2、加载模型参数信息需要先导入模型的结构，
# 然后通过model.load_state_dic(torch.load('model_state.pth'))

import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
batch_size = 10

features = torch.tensor(np.random.normal(0, 1, (num_examples,num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,size=labels.size()), dtype=torch.float)


# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)  # 使用print可以打印出网络的结构
# for param in net.parameters():
#     print(param)      # 使用print可以打印出网络的参数


# init.normal_(net.weights, mean=0.0, std=0.01)   #初始化为均值为0 ，标准差为0.01.的代码
init.constant_(net[0].bias, val=0.0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
# loss = nn.MSELoss()
#
# optimizer = optim.SGD(net.parameters(), lr=0.03)
# print('optimizer:',optimizer)
#
# num_epochs = 10
# for epoch in range(0, num_epochs ):
#     for X, y in data_iter:
#         output = net(X)
#         l = loss(output, y.view(-1, 1))
#         optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
#         l.backward()
#         optimizer.step()
#     print('epoch %d, loss: %f' % (epoch, l.item()))
#
# dense = net[0]
# print(true_w, dense.weight.data)
# print(true_b, dense.bias.data)
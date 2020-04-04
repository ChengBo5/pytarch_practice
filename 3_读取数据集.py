import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import d2lzh_pytorch as d2l
import sys

#加载数据
mnist_train = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST',
                                                train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST',
                                              train=False, download=False, transform=transforms.ToTensor())

# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))
# 查看单个样本大小
# feature, label = mnist_train[0]
# print(feature.shape, label) # Channel x Height X Width

# 本函数已保存在d2lzh包中⽅方便便以后使⽤用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress','coat','sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这⾥里里的_表示我们忽略略（不不使⽤用）的变量量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        plt.show()

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
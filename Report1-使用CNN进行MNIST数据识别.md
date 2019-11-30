# 使用CNN进行MNIST识别
*2019/11/30 刘睿平*
## 一、原理
MNIST是手写数字数据集，其中包括训练数据和测试数据两部分。每个手写数字由大小为28*28的点阵构成，每个点阵使用一个0-255的数字表示其灰度值。使用灰度图片进行图像识别时，每个数据的频道（channel）只有一个，若识别彩色图像，则频道有三个（RGB）。

卷积神经网络（CNN）是神经网络中应用广泛的一种，适合处理二维图像数据，通过对图像数据分块卷积提取信息，在神经网络中应用卷积层处理图像是非常有效的方法。CNN的特征为在神经网络中使用一层或多层卷积层。多层感知器（MLP）是人工神经网络中最为常见的网络层结构，MLP的层间节点使用全连接，前一层网络的数据加权后传递给后一层。

## 二、实验内容

根据现有资料搭建一个PyTorch卷积神经网络，完成对手写数字识别的训练和预测。主要内容有：

- 神经网络类的定义
- 训练过程的熟悉和训练函数的定义
- PyTorch下张量数据的处理
- 结果计算和调参优化

## 三、实验结果

### 3.1 CNN的定义

基于网络上的教程和示例，搭建了一个卷积神经网络模型，模型的定义源码如下：

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x
```

### 3.2 训练函数的实现

#### 确定起始模型

如果该训练是基于之前训练的结果继续的，则需要提取原来模型，否则建立一个新的模型（即建立模型类的一个对象）。

#### 导入训练数据

在神经网络中，数据是分批次输入网络的，如果数据单个输入则其批次个数为1，需要使用view()函数为其增加一个维度。一般来说，选择一个合适的批次数（batch size）可以提高模型的准确率、稳定性和运行速度，具体的选取应该在2-32之间。（参考[怎么选取训练神经网络时的Batch size?](https://www.zhihu.com/question/61607442/answer/440944387)）

#### 设置损失函数和优化器

损失函数是机器学习中用于优化的目标函数，每个学习模型的训练目标都是最小化损失函数（有的为最大化收益函数）。Pytorch中自带很多常见的损失函数，MNIST识别属于分类问题，我们使用分类问题最有效的交叉熵损失函数（Cross Entropy Loss），该函数是计算分类正确和错误的交叉熵，具体原理略，可见网络或CS5228课程内容。如果PyTorch库中没有合适的损失函数，则需要自己定义一个类似于compute_loss()函数，可见CS5446的第三次作业。

优化器是定义优化方法进行参数优化的类，有效的优化器可以使模型拟合更快更准确。在本次实验中使用Adam优化器，Adam是经典的一阶导数优化器，其中的原理与AdaBoost算法相同，比梯度下降法更有效，具体方法可以看[论文]( https://arxiv.org/abs/1412.6980 )或CS5228课程内容。

#### 学习循环过程

神经网络的学习和机器学习一样，都是循环进行、不断优化的。神经网络的学习一般对同一个数据学习不止一次，重复学习的次数是一个超参数定义为Epoch。循环中每一次进行的主要工作有：将一个批次的数据传入神经网络中；并行计算批次中每个数据的输出值（forward方法）；通过输出值和标记值比较；计算损失函数；代入优化器进行网络参数的优化。每个批次的数据所用的网络参数相同。

#### 代码如下

```python
def train(model_class, train_loader,model_path=False):
    if model_path != False:
        model = get_model(model_path)
    else:
        model = model_class()
    print(model)
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print(model.parameters())
    print(opt)
    loss_count = []
    for epoch in range(MAX_EPOCHES):
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            batch_x = x.to(device)  # torch.Size([128, 1, 28, 28])
            batch_y = y.to(device)  # torch.Size([128])
            out = model.forward(batch_x)  # torch.Size([128,10])
            loss = loss_func(out, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(out.data.argmax(1), batch_y, end=' ')
            # exit()
            correct += (out.data.argmax(1) == batch_y).sum()
            if i % 20 == 0:
                loss_count.append(loss)
                print('Episode: {}\tLoss: {:.6f}\tAccurancy: {:.3f}%'\
                      .format(i, loss.data, float(correct*100)/float((i+1) * BATCH_SIZE)))
                torch.save(model, './model.pt')
```

### 3.3 调参和结果

设置适当超参数后，结果为

| Max_Epoches | Learning_Rate | Batch_Size | Accurancy |
| :---------: | :-----------: | :--------: | :-------: |
|      2      |     0.001     |     32     |  97.557%  |
|      3      |     0.001     |     32     |  98.128%  |
|             |    0.0005     |     32     |  97.827%  |
|             |    0.0005     |     24     |  98.273%  |
|      4      |     0.001     |     32     |  97.256%  |
|      4      |    0.0005     |     32     |  97.763   |

### 3.4 结果分析

1. Epoch > 3时在训练时结果提升显著，但是在测试集中的结果反而下降，说明产生了过拟合现象。
2. Learning Rate过大和过小都不宜，取得合适的值即可，在训练后期，学习率应该减小使得模型慢慢逼近最优解。
3. Batch Size对训练结果也是敏感的，使用较小的数值会使训练速度明显减慢。且较小的数值代表有更多次优化模型的机会，因此小的Batch Size应该配合小的Learning Rate使用。

## 四、优化展望

从本次实验的结果可以看出，无论怎么调整超参数，模型的准确率都只能在98%，要想提升模型的准确性，单纯调整超参数是远远不够的。以下列出几种改进准确性的方法，供后续实验

- 多次训练神经网络，每次使用不同的超参数
- 参考优化的模型和文献，调整神经网络的结构，隐含层中节点的数量
- 在学习后期设置两个Buffer存储分类正确和错误的学习案例，在学习时同时从两个案例中对等取出，避免因为后期错误案例较少使得一个批次中没有一个分类错误，随即优化器任意改变参数，导致模型参数不稳定
- 参考资料设置其他改进方法。
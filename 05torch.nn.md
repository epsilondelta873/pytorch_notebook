# Torch.nn

[torch.nn官方文档](https://pytorch.org/docs/stable/nn.html)

## nn.Module

nn.Module是Torch.nn中的一个类，包含了神经网络的基本骨架(初始化和前向传播)，建立自己的神经网络基本都需要继承这个类然后根据自己的需要修改方法/参数

注意在初始化时需要继承父类的初始化

```python
super(xx,self).__init__()
```

一个简单的示范

```python
import torch.nn as nn
import torch

class ep(nn.Module):
    def __init__(self):
        super(ep,self).__init__() #继承父类 必须要有
    def forward(self,x):
        out_put = x+1
        return out_put

model = ep()
#输入需要是tensor类型
input = torch.tensor(1.0)
print(model(input))
```

## nn.Conv1d/nn.Conv2d 卷积

> 参数：in_channels, out_channels, kernel_size, **padding, stride, dilation**

[卷积操作及参数可视化padding/stride/dilation](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

1.将tensor输入卷积层的时候需要注意tensor的维度,要求的维度有[**batch_size,channel,height,width**],如果没达到需要通过torch.reshape转换(类比于numpy.reshape)

2.卷积会改变tensor的尺寸,已知输出输出tensor的尺寸可以通过公式计算出卷积的参数

$$
H_{out} =\left\lfloor\ \frac{H_{in} + 2 \times padding[0] - dilation[0] \times (kernel\_size[0] - 1) - 1}{stride[0]} + 1\right\rfloor
$$

$$
W_{out} = \left\lfloor\frac{W_{in} + 2 \times padding[1] - dilation[1] \times (kernel\_size[1] - 1) - 1}{stride[1]} + 1\right\rfloor
$$

3.卷积核矩阵不需要自行设定,设定卷积核的size即可,它相当于线性层的权重矩阵,反向传播会自动对其优化(个人理解)

4.out_channels是输出通道数,就是卷积核的个数,实际上就是使用两个卷积核进行卷积操作,再把结果合并

2个简单的例子,第一个例子展示了padding和stride参数,第二个例子是在数据集上对卷积进行了演示

```python
import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]])

#定义卷积核
kernel = torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]])

print(input.shape)
print(kernel.shape)

#如果要进行卷积需要对tensor的尺寸进行变换
#batch_size,channel,height,width
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

print(input.shape)
print(kernel.shape)

#进行卷积操作
#stride是步长
output = F.conv2d(input,kernel)
print(output)

output2 = F.conv2d(input,kernel,stride=2)
print(output2)

#padding是填充
#padding = 1,是上下左右都填充1,5*5变成7*7
output3 = F.conv2d(input,kernel,stride=1,padding=1)
print(output3)
```

```python
#kernel_size是卷积核的大小,不需要手动写kernel矩阵,训练过程中会对卷积核不断调优
#out_channels是输出通道数,就是卷积核的个数
#实际上就是使用两个卷积核进行卷积操作,再把结果合并

#提数据 先提成dataset再转换成dataLoader
dataset = torchvision.datasets.CIFAR10(root='./data2',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

#定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #定义卷积层
        #为什么输入通道是3,因为图片是RGB三通道
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        return x

cnn = CNN()

writer = SummaryWriter('cnn')
#把数据填进网络,并在tensorboard中查看结果
#30 = 32 - 3 + 1
step = 0
for data in dataloader:
    img,target = data
    # [64, 3, 32, 32] -> [64, 6, 30, 30]
    output = cnn(img)
    #要在tensorboard中查看结果需要channel不大于3
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images('cnn',output,step)
    step += 1
writer.close()
```

## nn.MaxPool2d 池化

>参数:kernel_size, stride, padding, dilation, ceil_mode=False

池化和卷积很类似,但是最大池化是求选中的矩阵区域的最大值，平均池化是求选中区域的均值

ceil_mode参数控制了最后一个kernel如果无法完全覆盖矩阵，应当保留还是舍弃的问题,默认为False舍弃那么结果的矩阵就会变小(与True相比)

最大池化的作用：保留数据的特征并且减小数据量 1080p -> 720p

下面是一个简单的示例可以通过tensorboard查看最大池化对图片的压缩效果

```python
#直观查看maxpooling的效果

#提数据 先提成dataset再转换成dataLoader
dataset = torchvision.datasets.CIFAR10(root='./data2',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

writer = SummaryWriter('max_pooling')
#把数据填进网络,并在tensorboard中查看结果
#30 = 32 - 3 + 1
step = 0
model = max_pooling()
for data in dataloader:
    img,target = data
    # [64, 3, 32, 32] -> [64, 6, 30, 30]
    output = model(img)
    #要在tensorboard中查看结果需要channel不大于3
    writer.add_images('max_pooling',output,step)
    step += 1
writer.close()
```

## 非线性激活层 nn.functional.sigmoid()

```python
input = torch.tensor([[1,-0.5],[-1,3]])

class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid,self).__init__()
    def forward(self,x):
        return nn.functional.sigmoid(x)

dataset = torchvision.datasets.CIFAR10(root='./data2',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

writer = SummaryWriter('sigmoid')
#把数据填进网络,并在tensorboard中查看结果
step = 0
model = sigmoid()
for data in dataloader:
    img,target = data
    writer.add_images('input',img,step)
    output = model(img)
    #要在tensorboard中查看结果需要channel不大于3
    writer.add_images('output',output,step)
    step += 1
writer.close()
```

## 线性层

> 参数:in_features, out_features, bias=True

线性层需要将所有数据展开成为二维张量(batch_size, channels *height * width)输入，因此in_features为channel*height*width，输出结果也是一维向量

输入张量可通过torch.flatten()展平，如果有batch_size记得把start_dim设置为1

```python
dataset = torchvision.datasets.CIFAR10(root='./data2',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

writer = SummaryWriter('linear')

class linear(nn.Module):
    def __init__(self):
        super(linear,self).__init__()
        #channel*height*width
        self.linear = nn.Linear(3*32*32,10)
    def forward(self,x):
        return self.linear(x)

model = linear()
step = 0
for data in dataloader:
    img,target = data
    print(img.shape)
    img = torch.flatten(img,start_dim=1)
    print(img.shape)
    output = model(img)
    print(output.shape)
    step += 1
    break
```

## nn.Sequential 简化模型

在初始化中使用nn.Sequential定义model可以简化代码在前向传播中的写法，具体见下方实例代码

```python
class ep(nn.Module):
    def __init__(self):
        super(ep,self).__init__() #调用父类的构造函数
        self.model1 = nn.Sequential(
            #conv中的padding通过计算得出
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #linear的输入输出都是一维
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        #前向传播中就可以直接调用写好的model即可
        x = self.model1(x)
        return x

model = ep()
input = torch.ones((64,3,32,32))
print(model(input).shape)

writer = SummaryWriter('logs_seq')
#计算图
writer.add_graph(model,input)
writer.close()
```

## nn.Loss 损失函数

损失函数并不写在nn.Module中，而是与nn.Module平级写在训练过程中，并且需要先初始化损失函数的类（与nn.Module同理）

在调用nn.Loss时也需要关注输入tensor与目标tensor要求的形状，一般来说都是(batch_size,*)

```python
import torch
from torch import nn

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

#为什么要reshape,因为输入的数据是[3],而loss函数需要的是[1,3] 1是batch_size
#主要关注input和target的shape
input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target,(1,1,1,3))

loss = nn.L1Loss()
print(loss(input,target))
```

损失函数+反向传播示例

```python
import torchvision
from torch.utils.data import DataLoader
dataset = torchvision.datasets.CIFAR10(root='./data2',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=1)
class ep(nn.Module):
    def __init__(self):
        super(ep,self).__init__() #调用父类的构造函数
        self.model1 = nn.Sequential(
            #conv中的padding通过计算得出
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #linear的输入输出都是一维
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x
#nn.Module与nn.Loss都要先进行初始化
model = ep()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    img,target = data
    output = model(img)
    #计算损失
    loss_value = loss(output,target)
    #反向传播写在这里
    loss_value.backward()
    print(loss_value)
    break
```

## 反向传播与优化器

训练一个神经网络的大致代码架构为：

```text
提取数据:dataset提取外部数据,dataloader将dataset中的数据随机分批

定义神经网络模型(nn.Module)
    初始化
    前向传播

初始化模型
初始化损失函数
初始化优化器

分epoch分batch遍历数据(dataloader)
    把数据喂给模型
    计算损失(nn.Loss)
    之前的梯度清0(optim.zero_grad)
    反向传播求梯度(loss.backward)
    更新权重(optim.step)
```

初始化优化器时需要设定使用的方法(SGD/Adam)，模型参数(模型名称.parameters())，以及学习率lr

提供一个简单的示例

```python
dataset = torchvision.datasets.CIFAR10(root='./data2',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=1)
class ep(nn.Module):
    def __init__(self):
        super(ep,self).__init__() #调用父类的构造函数
        self.model1 = nn.Sequential(
            #conv中的padding通过计算得出
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #linear的输入输出都是一维
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x
#先进行初始化 模型/损失函数/优化器
model = ep()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(5):
    running_loss = 0.0
    for data in dataloader:
        img,target = data
        output = model(img)
        #计算损失
        loss_value = loss(output,target)
        #梯度清零
        optim.zero_grad()
        #反向传播求梯度
        loss_value.backward()
        #更新参数
        optim.step()
        running_loss += loss_value.item()
    print(running_loss)
```

# Dataloader

从dataset提取数据到神经网络，打包成batch的形式传入

设置多个epoch可以使得不同的样本合成一个批次(shuffle = True)

同时要注意使用tensorbord查看数据的时候SummaryWriter('dataloader')可以不用写绝对路径和提前创建，会自动在工作路径创建，但在terminal执行时最好引用绝对路径不然系统找不到(或者使用cd切换)

```conda
cd F:\RUC\pytorch
tensorboard --logdir='dataloader'
```

```python
#从torchvision中加载数据集
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

test_loader = DataLoader(test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
#batch_size:每次读取并合并的数据量,然后将所有数据按照该方式划分成n//4批

#查看原始数据集的第一张图片及标签
img,target = test_set[0]
print(img.shape)
print(target)

#查看dataloader中的数据
writer = SummaryWriter('dataloader')
step = 0
#epoch是指遍历整个数据集的次数
for epoch in range(2):
    for data in test_loader:
        img,target = data
        #print(img.shape)
        #print(target)
        writer.add_images('test_drop',img,step)
        step += 1
writer.close()
```

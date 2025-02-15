# Transforms

## Tensor介绍——基于transforms.ToTensor()

**transforms.ToTensor()**:将PIL.Image或者numpy.ndarray转换为tensor

```python
from torchvision import transforms
from PIL import Image

#读取一个图片
img_path = r'F:\RUC\pytorch\数据集\练手数据集\train\ants_image\541630764_dbd285d63c.jpg'
img = Image.open(img_path)

#从transforms中提取ToTensor工具用于转换图片
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

#查看tensor
tensor_img
```

**为什么需要转换为tensor**：tensor数据类型包含了神经网络模型训练时需要的backward,grad,grad_fn等一系列参数

## 常见的transforms

### compose:将多个transforms方法组合起来使用

```python
#打开一张图片
from PIL import Image
from torchvision import transforms
img = Image.open(r'F:\RUC\pytorch\数据集\练手数据集\train\ants_image\2288481644_83ff7e4572.jpg')
                 
#compose
#transforms.Compose将多个transforms方法组合起来使用
#比如将图片先resize到256*256，然后随机裁剪到224*224，最后转换为tensor
```

### totensor:将PIL Image或者ndarray转换为tensor

```python
#将PIL Image或者 ndarray 转换为tensor，并且归一化到[0-1.0]之间
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
```

### normalize:将每个信道的数据标准化到设定的均值和标准差

```python
#将每个信道的数据标准化到设定的均值和标准差
#标准化前
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
#标准化后
print(img_norm[0][0][0])
```

### resize:调整图片大小

```python
#resize
#调整图片大小
print(img.size)
trans_resize = transforms.Resize((256,256))
img_resize = trans_resize(img)
print(img_resize.size)
#如果要在tensorboard中显示，需要转换成tensor  
```

### randomcrop:随机裁剪图片

```python
#randomcrop
#随机裁剪图片
trans_random = transforms.RandomCrop(256)
trans_compose = transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image('randomcrop',img_crop,i)
#在tensorboard中查看各步骤结果
writer.close()
```

## 如何将transform与数据集结合(进行批量操作)

```python
#下载torchvision中的数据集且不进行transform操作
import torchvision
train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=True)

#查看原始数据集
img,target = train_set[0]
img.show()
#查看标签
print(train_set.classes[target])
```

```python
#下载torchvision中的数据集且进行transform操作
#用compose定义transform
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
```

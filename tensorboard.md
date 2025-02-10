# Tensorboard的使用

## Tensorboard安装与打开

需要先写入内容再打开Tensorboard

```conda
#in conda terminal
#安装
pip install tensorboard

#打开tensorboard logs文件夹需要本地创建,且最好使用logs的绝对地址
tensorboard --logdir=logs #ctrl+c退出

#修改端口地址
tensorboard --logdir=logs --port=6007
```

## Tensorboard内容的写入

### SummaryWriter类

SummaryWriter类是 PyTorch 中 torch.utils.tensorboard 模块提供的一个重要工具，主要用于将训练过程中的各种数据（如损失值、准确率、图像等）写入 TensorBoard 可以读取的日志文件，方便用户通过 TensorBoard 可视化工具直观地观察和分析模型的训练过程和性能。

#### add_scalar()的使用

add_scalar()用于记录标量数据（如损失值、准确率、学习率等）随训练步数或训练轮数的变化情况 在 TensorBoard 中会以折线图的形式展示

```python
#导入类
from torch.utils.tensorboard import SummaryWriter
#创建类
writer = SummaryWriter('logs') #logs为存放日志的文件夹,需自行创建

"""
add_scalar参数介绍:
tag:标题
scalar_value:y轴的值
global_step:x轴的值
"""
for i in range(100):
    writer.add_scalar('y = 2x',i,2*i)

#关闭类
writer.close()
```

#### add_image()的使用

add_image()用于记录图像数据，例如输入图像、模型生成的图像等

注意:add_image读取的图片类型需为**tensor或者numpy.array**

```python
from PIL import Image
import numpy as np

#打开一张图片然后转成numpy类型
image_path = 'F:\RUC\pytorch\数据集\练手数据集\train\ants_image\0013035.jpg'
image_PIL = Image.open(image_path)
image_array = np.array(image_PIL)
#查看array的通道数
print(image_array.shape)

#创建类
writer = SummaryWriter('logs')
#转换成numpy数组后需要修改dataformats,因为和默认的不一样
writer.add_image('test',image_array,1,dataformats='HWC')
writer.close()

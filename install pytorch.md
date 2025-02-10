# 创建虚拟环境 下载pytorch
```conda
# 以下命令都在anaconda prompt中完成
# 在conda中创建新的虚拟环境
conda create -n pytorch python=3.9

# 激活环境
conda activate pytorch

#下载cuda toolkit和cuda dnn 网上有详细的教程

# 下载pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pytorch安装完成后
python
import torch
# 检查pytorch是否可以使用GPU,输出应为True
torch.cuda.is_available()
```

# 在vscode的jupyter中配置pytorch
```conda
# 首先在pytorch虚拟环境中安装jupyter notebook
pip install jupyer notebook
```
然后按照该文章进行配置

[解决windows11在vscode中powershell终端无法调用anaconda/miniconda的base虚拟环境](https://zhuanlan.zhihu.com/p/639866697)

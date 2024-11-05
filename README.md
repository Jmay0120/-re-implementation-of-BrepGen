# BrepGen: A B-rep Generative Diffusion Model with Structured Latent Geometry (SIGGRAPH 2024)
## 原文链接：https://github.com/samxuxiang/BrepGen

# Environment
    - Linux
    - Python 3.9
    - CUDA 11.8
    - PyTorch 2.2
    - Diffusers 0.27

# Dependencies
## 安装相关依赖与chamferdist包
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip install chamferdist
```

chamferdist若下载失败，使用官方提供的chamferdist源下载
```
python setup.py install
```

测试chamferdist是否安装成功
```python
import torch
from chamferdist import ChamferDistance

source_cloud = torch.randn(1, 100, 3).cuda()
target_cloud = torch.randn(1, 50, 3).cuda()

chamferDist = ChamferDistance()

dist_forward = chamferDist(source_cloud, target_cloud)
print(dist_forward.detach().cpu().item())
```
无报错则安装成功！

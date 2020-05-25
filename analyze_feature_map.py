# @Time    : 2020/5/25 10:23
# @Author  : Libuda
# @FileName: analyze_feature_map.py
# @Software: PyCharm
from model import CNN
import torchvision.transforms  as transform
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


trams = transform.Compose([
    transform.ToTensor()]
)

model_path = "./model/0.pth"
cnn = CNN(60,160,4)
cnn.load_state_dict(torch.load(model_path))
print(cnn)

img = Image.open("./dataset/test/0CFV_1590132234.png")
img = trams(img)
# 扩充维度 chw  nchw  dim
img = torch.unsqueeze(img,dim=0)

out = cnn(img)
for feature_map in out:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    plt.figure()
    # 展示前多少层特征图 可从网络结构中看到有多少通道
    for i in range(16):
        ax = plt.subplot(4,4,i+1)
        # 不指定gray 就会用蓝绿代替黑白
        plt.imshow(im[:,:,i])
    plt.show()

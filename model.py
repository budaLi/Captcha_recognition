# @Time    : 2020/5/21 17:17
# @Author  : Libuda
# @FileName: model.py
# @Software: PyCharm

# 为什么这么设计网络

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,width,height,captcha_len):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # layer1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # layer2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2),
            # layer3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            # 全连接层和验证码的宽高有关系 64为上一层输出
            nn.Linear((width//8)*(height//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            # 10+26+26 验证码字符种类
            nn.Linear(1024, captcha_len * (10 + 26 + 26)))

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    # nchw
    captcha = torch.rand(1,3,60,160)
    cnn = CNN(60,160,4)
    print(cnn)
    ca_res = cnn(captcha)
    print(ca_res.shape)
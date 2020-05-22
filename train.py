# @Time    : 2020/5/21 17:42
# @Author  : Libuda
# @FileName: train.py
# @Software: PyCharm
import torch
from torch.utils.data import DataLoader
from model import CNN
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transform
from dataset import Captcha_DataSets

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160
CAPTCHA_LEN= 4
LEARNING_RAGE = 0.001
NUM_EPOCH = 10
TRAIN_IMAGE_FILE = "./dataset/train"
TEST_IMAGE_FILE = "./dataset/test"


def train():
    # 加载网络
    cnn = CNN(IMAGE_WIDTH,IMAGE_HEIGHT,CAPTCHA_LEN)
    print(cnn)
    # 定义损失函数
    criterion = nn.MultiLabelSoftMarginLoss()
    # 优化器
    optimzier = torch.optim.Adam(
        cnn.parameters(),
        lr=LEARNING_RAGE
    )

    # 加载训练集验证集测试集
    trams = transform.Compose([

        # Grayscale() 会将模型通道数改为1 导致报错
        # transform.Grayscale(),
        transform.ToTensor()]
    )

    # 训练集
    train_dataset = Captcha_DataSets(TRAIN_IMAGE_FILE,trams)
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=0)
    # 测试集
    test_dataset = Captcha_DataSets(TEST_IMAGE_FILE,trams)
    test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True,num_workers=0)

    for epoch in range(NUM_EPOCH):
        # 指定为train
        cnn.train(True)
        for iter_num,(images,labels) in enumerate(train_dataloader):
            # 自动灰度化
            images = Variable(images)
            labels = Variable(labels.float())
            # 评测模型
            predict_labels = cnn(images)
            loss = criterion(predict_labels,labels)
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()

            if (iter_num) % 10 == 0:
                cnn.eval()
                test_loss = 0
                with torch.no_grad():
                    for i, (images, labels) in enumerate(test_dataloader):
                        images = Variable(images)
                        labels = Variable(labels.float())
                        # 评测模型
                        predict_labels = cnn(images)
                        test_loss += criterion(predict_labels, labels)

                print("epoch:{},iter:{},train loss:{:.4f},val loss:{:.4f}".format(epoch,iter_num,loss.item(),test_loss))

        # 一个epoch保存一个模型
        torch.save(cnn.state_dict(), "./model/{}.pth".format(epoch))
        print("save epoch :{} model".format(epoch))



if __name__ == '__main__':
    train()
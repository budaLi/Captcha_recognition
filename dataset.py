# @Time    : 2020/5/21 17:44
# @Author  : Libuda
# @FileName: dataset.py
# @Software: PyCharm
# 数据集
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import torchvision.transforms as transform

# 自定义dataloader
class Captcha_DataSets(Dataset):
    def __init__(self,image_file,transform=None):
        # os.listdir本就是乱序的
        self.images = [os.path.join(image_file,image)  for image in os.listdir(image_file)]
        self.transform  = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        # 验证码标签
        label = self.images[idx].split(os.path.sep)[-1].split(".")[0]
        if self.transform:
            image = self.transform(image)

        return image,label


if __name__ == '__main__':
    image_file = "./dataset"
    trams = transform.Compose([
        transform.Grayscale(),
        transform.ToTensor()]
    )
    dataset = Captcha_DataSets(image_file,trams)
    data = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)

    for epoch_data in data:
        for s in epoch_data:
            print("image:{},label:{}".format(s[0],s[0]))

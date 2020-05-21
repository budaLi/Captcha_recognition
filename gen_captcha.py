# @Time    : 2020/5/21 16:33
# @Author  : Libuda
# @FileName: gen_captcha.py
# @Software: PyCharm

# 生成验证码数据
from captcha.image import ImageCaptcha
from  PIL import Image
import random
import time
import os

# 验证码长度
MAX_CAPTCHA = 4
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160


def random_captcha(captcha_lenght,fliepath):
    """
    随机生成数字 可对应数字及字母的ascii码  再将其转换为对应字符
    :param captcha_lenght:生成的验证码长度
    :param fliepath:生成的文件路径
    :return:
    """
    res_captcha = ""
    # 数字
    number_range=list(map(str,range(0,10)))
    # 小写数字
    small_letter_range = list(map(chr,range(65,91)))
    # 大写数字
    max_letter_range = list(map(chr,range(97,123)))
    captcha_lis = number_range+small_letter_range+max_letter_range
    for _ in range(captcha_lenght):
        random_index  = random.randint(0,len(captcha_lis)-1)
        ca = captcha_lis[random_index]
        res_captcha+=ca

    image = ImageCaptcha()
    captcha_image = Image.open(image.generate(res_captcha))
    captcha_image.save(fliepath+res_captcha+".png")

    return res_captcha


if __name__ == '__main__':
    while 1:
        res = random_captcha(4,"./dataset/")
        print(res)









# @Time    : 2020/5/22 9:21
# @Author  : Libuda
# @FileName: one_hot_encode.py
# @Software: PyCharm

# -*- coding: UTF-8 -*-
import numpy as np
# import captcha_setting

# 如何将n位数的验证码编码 输入神经网络  得到forwar结果后可解码为原结果
#）用0,1编码 每63个编码一个字符，因为数字加字符共62  再加一个非验证码类 _
ALL_CHAR_SET_LEN = 10+26*2
MAX_CAPTCHA = 4

def encode(text):
    vector = np.zeros(ALL_CHAR_SET_LEN * MAX_CAPTCHA, dtype=float)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k
    for i, c in enumerate(text):
        idx = i * ALL_CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1.0
    return vector

def decode(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % ALL_CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

if __name__ == '__main__':
    e = encode("BK7H")
    print(e)
    print(decode(e))



# -*- coding:UTF-8 -*-
import numpy as np


def softmax(x):  # 给定一个矩阵, 计算每行减去该行最大值后的softmax输出
    #print("x=",x)
    c=np.nanmax(x, axis=1, keepdims=True)
    x_bar = x - c  # 减去该行最大值: 防止数值溢出
    z = np.nansum(np.exp(x_bar), axis=1, keepdims=True)  # 计算自然底数幂和
    s=np.exp(x_bar)/z
    #print("s=",s)
    return s  # 返回softmax输出


class SoftmaxCE(object):  # 利用softmax转换计算交叉熵损失值的抽象类
    def __init__(self):  # 构造函数
        pass

    @staticmethod
    def __call__(x, y,):  # 调用方法: x为n_samples×n_features的矩阵, y是标签0~9
        sf = softmax(x)  # 计算x的softmax输出
        n = x.shape[0]  # 获取样本数
        eps = 1e-7
        sf_log = -np.sum(y * np.log(sf + eps))/n  # 对应y的位置的概率即为预测准确的概率
        #print("sf_log=",sf_log)
        loss = np.mean(sf_log)  # 用这部分概率来计算交叉熵值
        g = sf.copy()

        g -= y  # softmax交叉熵求导
        g /= n
        #print("g=",g)
        return loss, g, sf  # 返回交叉熵损失值, 梯度及softmax层的输出结果


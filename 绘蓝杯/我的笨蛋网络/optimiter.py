# -*- coding:UTF-8 -*-
import abc
import numpy as np
from tqdm import tqdm


class Optimizer(object):  # 抽象优化器类, 为神经网络优化的一次算法(SGD算一次,Newton算二次): params_groups是神经网络模型中所有的参数构成的列表, configs是优化超参数
    def __init__(self, param_groups):  # 构造函数
        self.param_groups = param_groups  # 参数组应该是一个列表, 一般包含w, b两个字典, 再下一层是param与grad两个字段

    @abc.abstractmethod
    def step(self, isNestrov):  # 步长问题
        pass


class SGD(Optimizer):  # 随机梯度下降类
    def __init__(self, param_groups,
                 lr=1e-2,
                 ):  # 构造函数
        super(SGD, self).__init__(param_groups)
        self.configs = dict(  # 配置字典中包含学习率与权重衰减等超参数
            lr=lr,
        )

    def step(self):  # 步长问题
        lr = self.configs["lr"]  # 获取学习率
        count = 0
        for group in self.param_groups:  # 遍历每个参数
            count += 1
            for k, p in group.items():  # 获取每个参数的信息
                grad = p["grad"]  # 获取参数梯度值
                #print("grad=",grad.shape)
                #print("p=",p["param"].shape)
                v =- grad
                p["param"] += lr * v


class Adam:
    def __init__(self, param_groups, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.param_groups=param_groups

    def update(self, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for group in self.param_groups:  # 遍历每个参数
                for key, val in group.items():  # 获取每个参数的信息
                    self.m[key] = np.zeros_like(val)
                    self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in range(6):
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            self.param_groups[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

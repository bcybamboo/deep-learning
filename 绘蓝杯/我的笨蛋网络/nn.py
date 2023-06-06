# -*- coding:UTF-8 -*-
import math
import random
import time
import numpy as np
from loss import SoftmaxCE, softmax
from layers import Conv, ELU, MaxPool, Linear,BN


class CNN(object):  # 卷积神经网络架构类: 3层（卷积+激活）+ 池化 + 3层（线性+激活+线性）+softmax
    def __init__(self,
                 image_size=(3,224,224),
                 channels=3,
                 conv1_kernel=3,
                 conv2_kernel=3,
                 conv3_kernel=3,
                 pool_kernel=2,
                 hidden_units1=512,
                 hidden_units2=512,
                 n_classes=10,
                 ):  # 构造函数: 初始化神经网络,定义网络层
        """ 类构造参数 """
        self.image_size = image_size  # 图片形状3×H×W
        self.channels = channels  # 卷积层的轨道数
        self.conv1_kernel = conv1_kernel  # 卷积层核维度
        self.conv2_kernel = conv2_kernel  # 卷积层核维度
        self.conv3_kernel = conv3_kernel  # 卷积层核维度
        self.pool_kernel = pool_kernel  # 池化层核维度
        self.hidden_units1 = hidden_units1  # 线性传播中隐层单元数量
        self.hidden_units2 = hidden_units2  # 线性传播中隐层单元数量
        self.n_classes = n_classes  # 分类总数
        """ 类常用参数 """
        channel, height, width = self.image_size  # 这三个变量将记录卷积部分的输入轨道与维度
        self.conv_stride = 1  # 卷积核移动步长
        self.conv_padding = 1  # 卷积层对输入padding数量
        self.pool_stride = 2  # 池化层窗口移动步长
        self.pool_padding = 0  # 池化层对输入padding数量
        self.conv1 = Conv(  # 卷积层: 3×224×224-->3×(224-3+2+1)×(224-3+2+1)-->3×224×224
            height=self.conv1_kernel,
            width=self.conv1_kernel,
            in_channels=self.image_size[0],
            out_channels=self.channels,
            stride=self.conv_stride,
            padding=self.conv_padding,
            init_scale=1e-2,
        )

        """ 经过卷积层后轨道与维度的变化 """
        channel = self.channels
        height += (2 * self.conv_padding - self.conv1_kernel)
        height /= self.conv_stride
        height = int(height) + 1
        width += (2 * self.conv_padding - self.conv1_kernel)
        width /= self.conv_stride
        width = int(width) + 1
        self.relu1 = ELU()  # 激活层 A: 3×224×224-->3×224×224

        """ 类常用参数 """
        self.conv2 = Conv(  # 卷积层: 3×224×224-->3×(224-3+2+1)×(224-5+1)-->3×224×224
            height=self.conv2_kernel,
            width=self.conv2_kernel,
            in_channels=self.channels,
            out_channels=self.channels,
            stride=self.conv_stride,
            padding=self.conv_padding,
            init_scale=1e-2,
        )
        """ 经过卷积层后轨道与维度的变化 """
        channel = channel
        height += (2 * self.conv_padding - self.conv2_kernel)
        height /= self.conv_stride
        height = int(height) + 1
        width += (2 * self.conv_padding - self.conv2_kernel)
        width /= self.conv_stride
        width = int(width) + 1
        self.relu2 = ELU()  # 激活层 A: 3×224×224-->3×224×224
        self.conv3 = Conv(  # 卷积层: 3×224×224-->3×(224-3+2+1)×(224-3+2+1)-->3×224×224
            height=self.conv3_kernel,
            width=self.conv3_kernel,
            in_channels=self.channels,
            out_channels=self.channels,
            stride=self.conv_stride,
            padding=self.conv_padding,
            init_scale=1e-2,
        )
        """ 经过卷积层后轨道与维度的变化 """
        channel = channel
        height += (2 * self.conv_padding - self.conv3_kernel)
        height /= self.conv_stride
        height = int(height) + 1
        width += (2 * self.conv_padding - self.conv3_kernel)
        width /= self.conv_stride
        width = int(width) + 1
        self.relu3 = ELU()  # 激活层 A: 3×224×224-->3×224×224
        self.pool = MaxPool(  # 池化层: 3×224×224-->3×((224-2)/2+1)×((224-2)/2+1)-->3×112×112
            kernel_size=self.pool_kernel,
            stride=self.pool_stride,
            padding=self.pool_padding,
        )
        """ 经过池化层后轨道与维度的变化 """
        channel = channel
        height += (2 * self.pool_padding - self.pool_kernel)
        height /= self.pool_stride
        height = int(height) + 1
        width += (2 * self.pool_padding - self.pool_kernel)
        width /= self.pool_stride
        width = int(width) + 1

        #print("height={}.width={}".format(height,width))
        self.linear1 = Linear(  # 线性层 A: 3×112×112-->37632-->512
            in_features=channel * height * width,
            out_features=self.hidden_units1,
            init_scale=1e-2,
        )
        self.relu4 = ELU()  # 激活层 B: 512-->512
        self.linear2 = Linear(  # 线性层 B: 512-->512
            in_features=self.hidden_units1,
            out_features=hidden_units2,
            init_scale=1e-2,
        )
        self.relu5 = ELU()  # 激活层 B: 512-->512
        self.linear3 = Linear(  # 线性层 B: 512-->10
            in_features=self.hidden_units2,
            out_features=10,  # 最后一层应该是输出对每个字段的预测概率
            init_scale=1e-2,
        )
        """ 类初始化 """
        self.softmaxce = SoftmaxCE()
        self.param_groups = [  # 卷积层与线性层有参数(3+3)
            {
                "w": {
                    "param": self.conv1.params["w"]["param"],
                    "grad": self.conv1.params["w"]["grad"],
                    "pregrad": np.zeros_like(self.conv1.params["w"]["grad"])
                },
                "b": {
                    "param": self.conv1.params["b"]["param"],
                    "grad": self.conv1.params["b"]["grad"],
                    "pregrad": np.zeros_like(self.conv1.params["b"]["grad"])
                },
            },
            {
                "w": {
                    "param": self.conv2.params["w"]["param"],
                    "grad": self.conv2.params["w"]["grad"],
                    "pregrad": np.zeros_like(self.conv2.params["w"]["grad"])
                },
                "b": {
                    "param": self.conv2.params["b"]["param"],
                    "grad": self.conv2.params["b"]["grad"],
                    "pregrad": np.zeros_like(self.conv2.params["b"]["grad"])
                },
            },
            {
                "w": {
                    "param": self.conv3.params["w"]["param"],
                    "grad": self.conv3.params["w"]["grad"],
                    "pregrad": np.zeros_like(self.conv3.params["w"]["grad"])
                },
                "b": {
                    "param": self.conv3.params["b"]["param"],
                    "grad": self.conv3.params["b"]["grad"],
                    "pregrad": np.zeros_like(self.conv3.params["b"]["grad"])
                },
            },
            {
                "w": {
                    "param": self.linear1.params["w"]["param"],
                    "grad": self.linear1.params["w"]["grad"],
                    "pregrad": np.zeros_like(self.linear1.params["w"]["grad"])
                },
                "b": {
                    "param": self.linear1.params["b"]["param"],
                    "grad": self.linear1.params["b"]["grad"],
                    "pregrad": np.zeros_like(self.linear1.params["b"]["grad"])
                },
            },
            {
                "w": {
                    "param": self.linear2.params["w"]["param"],
                    "grad": self.linear2.params["w"]["grad"],
                    "pregrad": np.zeros_like(self.linear2.params["w"]["grad"])
                },
                "b": {
                    "param": self.linear2.params["b"]["param"],
                    "grad": self.linear2.params["b"]["grad"],
                    "pregrad": np.zeros_like(self.linear2.params["b"]["grad"])

                },
            },
            {
                "w": {
                    "param": self.linear3.params["w"]["param"],
                    "grad": self.linear3.params["w"]["grad"],
                    "pregrad": np.zeros_like(self.linear3.params["w"]["grad"])
                },
                "b": {
                    "param": self.linear3.params["b"]["param"],
                    "grad": self.linear3.params["b"]["grad"],
                    "pregrad": np.zeros_like(self.linear3.params["b"]["grad"])
                },
            },
        ]

    def oracle(self, x, y):  # 计算损失函数值,输出得分,损失函数梯度: x为一个N_samples×N_channels×Height×Width的张量,y为类别标签
        """ 前向传播 """
        """ 卷积层 """
        conv1_out = self.conv1.forward(x)
        #print("conv1=",conv1_out)

        """ 激活层 """
        relu1_out = self.relu1.forward(conv1_out)

        """ 卷积层 """
        conv2_out = self.conv2.forward(relu1_out)
        #print("conv2=",conv2_out)

        """ 激活层 """
        relu2_out = self.relu2.forward(conv2_out)

        """ 卷积层 """
        conv3_out = self.conv3.forward(relu2_out)
        #print("conv3=",conv3_out)

        """ 激活层 """
        relu3_out = self.relu3.forward(conv3_out)

        """ 池化层 """
        pool_out = self.pool.forward(relu3_out)


        """ 线性层 """
        linear1_out = self.linear1.forward(pool_out)


        """ 激活层 """
        relu4_out = self.relu4.forward(linear1_out)

        """ 线性层 """
        linear2_out = self.linear2.forward(relu4_out)

        """ 激活层 """
        relu5_out = self.relu5.forward(linear2_out)

        """ 线性层 """
        linear3_out = self.linear3.forward(relu5_out)

        """ 软大交叉熵 """
        fx, g, s = self.softmaxce(linear3_out, y)  # 损失函数值&梯度(是算在最后一层上面的梯度)&得分
        """ 反向传播 """
        linear3_back = self.linear3.backward(g, relu5_out)

        relu5_back = self.relu5.backward(linear3_back, linear2_out)

        linear2_back = self.linear2.backward(relu5_back, relu4_out)
        #print(self.param_groups)
        #print(self.param_groups[0]["w"]["pregrad"])
        #print("self.param_groups[0] =",self.param_groups[0])

        relu4_back = self.relu4.backward(linear2_back, linear1_out)

        linear1_back = self.linear1.backward(relu4_back, pool_out)

        pool_back = self.pool.backward(linear1_back, relu3_out)

        relu3_back = self.relu3.backward(pool_back, conv3_out)


        conv3_back = self.conv3.backward(relu3_back, relu2_out)

        relu2_back = self.relu2.backward(conv3_back, conv2_out)


        conv2_back = self.conv2.backward(relu2_back, relu1_out)

        relu1_back = self.relu1.backward(conv2_back, conv1_out)



        conv1_back = self.conv1.backward(relu1_back, x)
        self.update_param()
        #print("l2 w_grad shape=",self.param_groups[3]["w"]["grad"].shape)
        return fx, s,conv1_back



    def score(self, x):  # 预测的得分,除了oracle函数外还需要一个另外的得分函数,这在检查精度时是有用的: x为输入特征
        conv1_out = self.conv1(x)
        relu1_out = self.relu1(conv1_out)
        conv2_out = self.conv2(relu1_out)
        relu2_out = self.relu2(conv2_out)
        conv3_out = self.conv3(relu2_out)
        relu3_out = self.relu3(conv3_out)
        pool_out = self.pool(relu3_out)
        linear1_out = self.linear1(pool_out)
        relu4_out = self.relu4(linear1_out)
        linear2_out = self.linear2(relu4_out)
        relu5_out = self.relu5(linear2_out)
        linear3_out = self.linear3(relu5_out)
        #print(linear2_out)
        s = softmax(linear3_out)
        return s

    def update_param(self, ):  # 更新参数及梯度
        self.param_groups[0] = self.conv1.params
        self.param_groups[1] = self.conv2.params
        self.param_groups[2] = self.conv3.params
        self.param_groups[3] = self.linear1.params
        self.param_groups[4] = self.linear2.params
        self.param_groups[5] = self.linear3.params

if __name__ == "__main__":
    cnn = CNN()
    '''a=np.random.randn(3, 3, 3)
    print(a)
    print(cnn.param_groups)
    print(cnn.param_groups[0]["w"]["param"])
    cnn.param_groups[0]["w"]["param"] = a
    print(cnn.param_groups[0]["w"]["param"])'''
    # print("c1=",cnn.conv1.params["w"]["param"])
    # cnn.param_groups[0]["w"]["param"]=cnn.conv1.params["w"]["param"]
    # print("成功！")
    print(cnn.param_groups[0]["w"]["pregrad"])

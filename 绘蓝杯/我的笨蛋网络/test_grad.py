# -*- coding:UTF-8 -*-
import os
import sys

import numpy as np
from nn import CNN
from check_gradient import *
from layers import Conv, MaxPool, Linear, ELU


def rel_error(x, y):  # 计算相对误差
    return np.nanmax(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))



check_conv_forward = True
# check_conv_forward = False
check_conv_backward = True
# check_conv_backward = False
check_linear_forward = True
# check_linear_forward = False
check_linear_backward = True
# check_linear_backward = False
check_pool_forward = True
# check_pool_forward = False
check_pool_backward = True
# check_pool_backward = False

if check_conv_forward:  # 检查卷积层前向传播是否正确
    x_shape = (2, 3, 4, 4)  # 2个样本, 3个轨道, 4×4像素
    w_shape = (3, 3, 4, 4)  # 3个输入轨道, 3个输出轨道, 4×4卷积核
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)  # 2×3×4×4的样本
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)  # 3×3×4×4的卷积过滤器
    b = np.linspace(-0.1, 0.2, num=3)  # 3个偏差
    conv = Conv(4, 4, 3, 3, 2, 1)  # 卷积层: 4×4卷积核, 3个输入轨道, 3个输出轨道, 步长2, padding值为1
    conv.params["w"]["param"] = w  # 设置卷积过滤器
    conv.params["b"]["param"] = b  # 设置偏差值
    out = conv(x)  # Layer类是可以被调用的, 返回前向传播的结果
    correct_out = np.array([  # 事先计算好的正确输出维度为: 2×3×2×2
        [
            [[-0.08759809, -0.10987781], [-0.18387192, -0.2109216]],
            [[0.21027089, 0.21661097], [0.22847626, 0.23004637]],
            [[0.50813986, 0.54309974], [0.64082444, 0.67101435]]
        ],
        [
            [[-0.98053589, -1.03143541], [-1.19128892, -1.24695841]],
            [[0.69108355, 0.66880383], [0.59480972, 0.56776003]],
            [[2.36270298, 2.36904306], [2.38090835, 2.38247847]]
        ]
    ])
    print("Testing convolutional forward...")
    print("difference: {}".format(rel_error(out, correct_out)))

if check_conv_backward:  # 检查卷积层反向传播是否正确
    np.random.seed(231)  # 初始化随机种子
    x = np.random.randn(2, 3, 16, 16)  # 2个样本, 3个轨道, 16×16像素
    w = np.random.randn(3, 3, 3, 3)  # 3个输入轨道, 3个输出轨道, 3×3的卷积核
    b = np.random.randn(3, )  # 3个偏差
    dout = np.random.randn(2, 3, 14, 14)  # 随机给定一个输出dout: 这个应该是被假设为下一层反向传回当层的输入梯度值
    conv = Conv(3, 3, 3, 3, 1, 0)  # 初始化一个3个输入轨道, 3个输出轨道, 3×3的卷积核, 步长为1且不padding
    conv.params["w"]["param"] = w  # 设置卷积过滤器
    conv.params["b"]["param"] = b  # 设置偏差
    out = conv(x)  # 计算卷积层在输入x后的输出结果
    dx = conv.backward(dout, x)  # 计算对应dout输入的反向传播输出
    dx_num = eval_numerical_gradient_array(conv, x, dout)  # dx_num维度与x,dx完全相同
    params = conv.params  # 获取卷积层的参数


    def fw(v):  # 计算用v过滤器后的输出结果
        tmp = params["w"]["param"]
        params["w"]["param"] = v
        f_w = conv(x)
        params["w"]["param"] = tmp
        return f_w


    def fb(v):  # 计算用v偏差后的输出结果
        tmp = params["b"]["param"]
        params["b"]["param"] = v
        f_b = conv(x)
        params["b"]["param"] = tmp
        return f_b


    dw = params["w"]["grad"]  # 卷积过滤器的梯度
    dw_num = eval_numerical_gradient_array(fw, w, dout)  # dw_num维度与w,dw完全相同
    db = params["b"]["grad"]  # db_num维度与b,db完全相同
    db_num = eval_numerical_gradient_array(fb, b, dout)

    print("Testing convolutional backward")
    print("dx error: {}".format(rel_error(dx_num, dx)))
    print("dw error: {}".format(rel_error(dw_num, dw)))
    print("db error: {}".format(rel_error(db_num, db)))

if check_linear_forward:  # 检查线性层前向传播是否正确
    x_shape = (2, 3, 4, 4)  # 2个样本, 3个轨道, 4×4像素
    w_shape = (3 * 4 * 4, 64)  # 从48维映射到64维
    b_shape = (1, 64)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=64).reshape(b_shape)  # 64个偏差
    linear = Linear(3 * 4 * 4, 64)
    linear.params["w"]["param"] = w  # 设置卷积过滤器
    linear.params["b"]["param"] = b  # 设置偏差值
    out = linear(x)  # Layer类是可以被调用的, 返回前向传播的结果

    correct_out = np.dot(x.reshape(2, 48), w) + b  # 单纯的全连接层
    print("Testing linear forward...")
    print("difference: {}".format(rel_error(out, correct_out)))

if check_linear_backward:  # 检查线性层反向传播是否正确
    np.random.seed(231)  # 初始化随机种子
    x = np.random.randn(2, 3, 4, 4)  # 2个样本, 3个轨道, 16×16像素
    w = np.random.randn(3 * 4 * 4, 64)  # 3个输入轨道, 3个输出轨道, 3×3的卷积核
    b = np.random.randn(1, 64)  # 3个偏差
    dout = np.random.randn(2, 64)  # 随机给定一个输出dout: 这个应该是被假设为下一层反向传回当层的输入梯度值
    linear = Linear(3 * 4 * 4, 64)  # 初始化一个3个输入轨道, 3个输出轨道, 3×3的卷积核, 步长为1且不padding
    linear.params["w"]["param"] = w  # 设置卷积过滤器
    linear.params["b"]["param"] = b  # 设置偏差
    out = linear(x)  # 计算卷积层在输入x后的输出结果
    dx = linear.backward(dout, x)  # 计算对应dout输入的反向传播输出
    dx_num = eval_numerical_gradient_array(linear, x, dout)  # dx_num维度与x,dx完全相同
    dx_num = dx_num.reshape(dx_num.shape[0], -1)  # 调整dx_num维度为二维张量
    params = linear.params  # 获取卷积层的参数


    def fw(v):  # 计算用v过滤器后的输出结果
        tmp = params["w"]["param"]
        params["w"]["param"] = v
        f_w = linear(x)
        params["w"]["param"] = tmp
        return f_w


    def fb(v):  # 计算用v偏差后的输出结果
        tmp = params["b"]["param"]
        params["b"]["param"] = v
        f_b = linear(x)
        params["b"]["param"] = tmp
        return f_b


    dw = params["w"]["grad"]  # 卷积过滤器的梯度
    dw_num = eval_numerical_gradient_array(fw, w, dout)  # dw_num维度与w,dw完全相同
    db = params["b"]["grad"]  # db_num维度与b,db完全相同
    db_num = eval_numerical_gradient_array(fb, b, dout)

    print("Testing linear backward")
    print("dx error: {}".format(rel_error(dx_num, dx)))
    print("dw error: {}".format(rel_error(dw_num, dw)))
    print("db error: {}".format(rel_error(db_num, db)))

if check_pool_forward:  # 检查池化层前向传播是否正确
    x_shape = (2, 3, 4, 4)  # 2个样本, 3个轨道, 4×4像素
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)  # 2×3×4×4的样本
    pool = MaxPool(kernel_size=2, stride=2, padding=0)  # 2×2的池化层, 步长2且不padding
    out = pool(x)
    out_shape = (2, 3, 2, 2)
    correct_out = np.zeros(out_shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            for k in range(out_shape[2]):
                for l in range(out_shape[3]):
                    correct_out[i, j, k, l] = x[i, j, 2 * k + 1, 2 * l + 1]  # 因为是按顺序排列的, 每个窗口的右下角恰好为最大值
    print("Testing pooling forward...")
    print("difference: {}".format(rel_error(out, correct_out)))

if check_pool_backward:  # 检查池化层反向传播是否正确
    np.random.seed(231)  # 初始化随机种子
    x = np.random.randn(3, 2, 8, 8)  # 随机给定一个输入x
    dout = np.random.randn(3, 2, 4, 4)  # 随机给定一个输出dout: 这个应该是被假设为下一层反向传回当层的输入梯度值
    pool = MaxPool(kernel_size=2, stride=2, padding=0)  # 初始化一个2×2的池化核, 不padding且步长为2, 其输出恰好为8×8-->4×4
    out = pool(x)  # 得出一个对应x的前向输出
    dx = pool.backward(dout, x)  # 得出一个对应dout的反向输出的梯度
    dx_num = eval_numerical_gradient_array(pool, x, dout)  # 调用手动计算梯度的函数: dx_num的维度与x的维度完全相同(3,2,8,8)
    print("Testing pooling backward:")
    print("dx error: ", rel_error(dx, dx_num))  # 你的误差应该在1e-12左右


# -*- coding:UTF-8 -*-
import sys
import numpy as np
from random import randrange

if sys.version_info >= (3, 0):  #xrange生成器
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))


def eval_numerical_gradient(f, x,
                            verbose=True,
                            h=1e-5
                            ):  # 求函数f在点x处的梯度: f为只有一个参数的函数, x为需要求梯度的点或者是数组
    fx = f(x)  # 求函数在给定点的函数值
    grad = np.zeros_like(x)  # 预设梯度为与x形状相同的玩意儿
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])  # 把x做成numpy生成器
    while not it.finished:  # 遍历x中的所有元素, 一个个求偏导
        ix = it.multi_index  # 获取生成器index
        oldval = x[ix]  # 获取对应index的数值
        x[ix] = oldval + h  # 右移一段
        fxph = f(x)  # 计算右函数值
        x[ix] = oldval - h  # 左移一段
        fxmh = f(x)  # 计算左函数值
        x[ix] = oldval  # 还原该index上原先的值
        grad[ix] = (fxph - fxmh) / (2 * h)  # 计算偏导数
        if verbose: print(ix, grad[ix])  # 输出梯度
        it.iternext()  # 步入下一次
    return grad


def eval_numerical_gradient_array(f, x, df,
                                  h=1e-5
                                  ):  # 对于一个接收了一个数组并返回数组的函数来计算数值的梯度
    grad = np.zeros_like(x)  # 这里的输入x的维度为N_samples×n_Channels×Height×Width
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])  # 把x做成一个迭代生成器
    while not it.finished:
        ix = it.multi_index  # 获取生成器index
        oldval = x[ix]  # 获取对应index的数值
        x[ix] = oldval + h  # 右移一段
        pos = f(x).copy()  # 计算右函数值
        x[ix] = oldval - h  # 左移一段
        neg = f(x).copy()  # 计算左函数值
        x[ix] = oldval  # 还原该index上原先的值
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)  # 这个乘以df
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output,
                                  h=1e-5
                                  ):  # 对一个操作输入与输出斑点的函数计算数值梯度: f为函数(f接收几个输入斑点作为参数,然后跟随一个斑点用于写入输出==>y=f(x,w,out),x与w为输入斑点,f的结果将被写入out),inputs为输入的斑点,output为输出斑点,h为步长
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(
            input_blob.vals,
            flags=["multi_index"],
            op_flags=["readwrite"]
        )
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]
            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig
            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)
            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output,
                                h=1e-5
                                ):
    result = eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )
    return result


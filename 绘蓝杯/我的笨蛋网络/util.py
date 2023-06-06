# -*- coding:UTF-8 -*-
import numpy as np


def accuracy(score, y):  # 计算分类精度, 返回一个0~1的标量: score是n×n类别得分矩阵, y为n×1的标签
    #print(y)
    b = np.argwhere(y > 0)
    #print(b)
    b=np.delete(b,0,axis = 1)
    #print(b)
    #print('Pre',score.argmax(axis=1))
    #print('GT',b)
    acc = np.sum(score.argmax(axis=1) == b.T)/len(b)
    return acc

if __name__=="__main__":
    score = np.random.randn(2,2)
    y=([1,0],[0,1])
    print("score=",score)
    acc=accuracy(score,y)
    print("acc=",acc)
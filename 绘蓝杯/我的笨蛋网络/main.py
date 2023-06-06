# -*- coding:UTF-8 -*-
import numpy as np
import os
from nn import CNN
from optimiter import SGD,Adam
from util import accuracy
from matplotlib import pyplot as plt
from dataset import train_loader,eval_loader
from tqdm import tqdm


def save(parameters, save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i]
    np.savez(save_as, **dic)


def load(parameters, file):
    params = np.load(file,allow_pickle=True)
    for i in range(len(parameters)):
        parameters[i] = params[str(i)]

def train(model, batch_size, n_epochs,
          lr=0.001,
          lr_decay=0.2,
          verbose=True,
          print_level=1,
          ):  # 模型训练
    print("开始训练...")
    n_train =1400  # 获取训练集样本数量
    iterations_per_epoch = max(n_train // batch_size, 1)  # 每个epoch的迭代次数
    n_iterations = n_epochs * iterations_per_epoch  # 总的迭代次数
    loss_hist = []  # 储存每个epoch的损失函数值
    acc_hist = []
    opt_params = {"lr": lr}  # 设置最优参数, 预设了学习率
    print("训练样本数：{}".format(n_train))
    print("每阶段迭代次数：{}".format(iterations_per_epoch))
    print("阶段数：{}".format(n_epochs))
    print("总迭代次数：{}".format(n_iterations))
    count = 0
    for epoch in tqdm(range(n_epochs)):  # 遍历所有epoch
        for t, (img, y) in tqdm(enumerate(train_loader)):  # 遍历总的迭代次数
            count += 1
            loss, score,grads = model.oracle(img, y)  # 评估函数值与梯度
            running_loss = np.sum(loss)
            np.save(f'loss/loss_epoch_{epoch}', running_loss)  # 保存loss数据
            loss_hist.append(loss)
            sgd = SGD(model.param_groups, **opt_params)  # 定义优化器
            sgd.step()  # 进行随机梯度下降
            #adm=Adam(model.param_groups)
            #adm.update(grads)
            if verbose and t % print_level == 0:  # 输出训练损失
                train_acc = accuracy(score, y)  # 计算精度
                acc_hist.append(train_acc)
                np.save('acc/acc_epoch_{}'.format(epoch), train_acc)  # 保存acc数据
                print("(Iteration {}/{},epoch {})loss:{},accu:{}".format(
                    count, n_iterations, epoch+1, loss_hist[-1], train_acc))
        print("epoch {}.loss:{}.accu:{}".format(
             epoch + 1, np.sum(loss_hist)/iterations_per_epoch,np.sum(acc_hist)/iterations_per_epoch ))
        acc_hist = []
        loss_hist = []

    """ 实验绘图 """
    plt.close()
    plt.figure()
    plt.plot(loss_hist,label="training loss")
    plt.legend(loc="best")
    plt.show()

def eval(val_loader):
    predict_labels = []
    labels = []
    for index, data in tqdm(enumerate(eval_loader)):
        x, y = data
        logits = model.score(x)
        #print("logit=",logits)
        pred = np.argmax(logits, axis=1)
        #print("pred=",pred)
        predict_labels.append(pred)
        y = np.argmax(y, axis=1)
        labels.append(y)
        #print(y)
    pred = np.array(predict_labels)
    labels = np.array(labels)
    acc = 0
    for i in range(len(pred)):
        acc += np.sum(pred[i] == labels[i])
    #print("acc=", acc)
    acc = acc / 200
    print("val dataset accuracy:", acc)
    return acc



def plot_acc(n):# n为文件数量
    plt.figure(figsize=(8, 7))  # 窗口大小
    y1 = []
    for i in range(0, n):
        enc = np.load('acc\\acc_epoch_{}.npy'.format(i))  # 文件返回数组
        tempy = enc.tolist()
        tempy = float(tempy)
        y1.append(tempy * 100)
    x1 = list(range(0, len(y1)))
    plt.plot(x1, y1, '.-', label='accuracy')  # label对于的是legend显示的信息名
    plt.grid()  # 显示网格
    plt_title = 'BATCH_SIZE = 8'
    plt.title(plt_title)  # 标题名
    plt.ylabel('acc')  # 纵坐标名
    plt.legend()  # 显示曲线信息
    plt.savefig("train_acc.jpg")  # 当前路径下保存图片名字
    plt.show()

def plot_loss(n):# n为文件数量
    plt.figure(figsize=(8, 7))  # 窗口大小
    y1 = []
    for i in range(0, n):
        enc = np.load('loss\\loss_epoch_{}.npy'.format(i))  # 文件返回数组
        tempy = enc.tolist()
        tempy = float(tempy)
        y1.append(tempy * 100)
    x1 = list(range(0, len(y1)))
    plt.plot(x1, y1, '.-', label='loss')  # label对于的是legend显示的信息名
    plt.grid()  # 显示网格
    plt_title = 'BATCH_SIZE = 8'
    plt.title(plt_title)  # 标题名
    plt.ylabel('loss')  # 纵坐标名
    plt.legend()  # 显示曲线信息
    plt.savefig("train_loss.jpg")  # 当前路径下保存图片名字
    plt.show()


if __name__ == "__main__":
    model = CNN()  # 初始化卷积神经网络
    num_train = 1400  # 训练集数量
    n_epochs = 1
    batch_size = 8
    load_file = 'bbb.npz'
    if os.path.isfile(load_file):
        load(model.param_groups, load_file)
        print("模型加载成功！")
    train(model,batch_size,n_epochs)
    save(model.param_groups, load_file)
    print("模型保存成功！")
    eval(eval_loader)
    plot_loss(n_epochs)

import numpy as np
import os

from main import load
'''from nn import CNN
model=CNN()
load_file = 'conect.npz'
if os.path.isfile(load_file):
    #load(model.param_groups, 'aaa.npz')
    print(model.param_groups[1]["w"]["param"])
    print("成功！")

'''
'''from loss import softmax
x=[[-0.01139895,-0.01300407,0.00725653,0.00553836,0.0045907,0.032295,
  -0.02366544, -0.00316791,  0.02977014,  0.01490272]]
y=softmax(x)'''

'''from loss import SoftmaxCE

for epoch in range(1,3):
    logits=np.random.randn(10,8)
    labels = np.random.randn(10,1)
    running_loss = 0.0
    loss_function = SoftmaxCE()
    loss=loss_function(logits, labels)
    running_loss =np.sum(loss)
    Train_Loss = str(running_loss)
    Loss0 = np.array(Train_Loss)
    np.save('loss/loss_epoch_{}'.format(epoch), Loss0)

    acc_train = 0.0
    train_num=10
    train_accurate = acc_train / train_num
    acc = str(train_accurate)
    acc0 = np.array(acc)
    np.save('acc/acc_epoch_{}'.format(epoch), acc0)



import matplotlib.pyplot as plt
import numpy as np


def plot_loss(n):
    plt.figure(figsize=(8, 7))  # 窗口大小可以自己设置
    y1 = []
    for i in range(1, n):
        enc = np.load('acc\\acc_epoch_{}.npy'.format(i))  # 文件返回数组
        tempy = enc.tolist()
        tempy = float(tempy)
        y1.append(tempy * 100)
    x1 = list(range(0, len(y1)))
    plt.plot(x1, y1, '.-', label='accuracy')  # label对于的是legend显示的信息名
    plt.grid()  # 显示网格
    plt_title = 'BATCH_SIZE = 8; LEARNING_RATE:0.001'
    plt.title(plt_title)  # 标题名
    plt.xlabel('per 400 times')  # 横坐标名
    plt.ylabel('LOSS')  # 纵坐标名
    plt.legend()  # 显示曲线信息
    plt.savefig("train_loss_v3.jpg")  # 当前路径下保存图片名字
    plt.show()


if __name__ == "__main__":
    plot_loss(2)  # 文件数量
'''

'''self.pool1 = MaxPool(  # 池化层: 3×224×224-->3×((224-2)/2+1)×((224-2)/2+1)-->3×112×112
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
width = int(width) + 1'''

'''index=7
label = np.zeros((1,9),dtype="int64")
print(label)
label=np.insert(label,index,1)
print(label)
print(label.shape)'''
'''g=np.random.randn(8,10)
#print("g1=", g.shape)
n=8
y=np.random.randn(8,10).astype("int64")
print(g)
for i in range(n):
    g -= y  # softmax交叉熵求导
g /= n
print("g=", g)'''
'''from nn import CNN
from dataset import train_loader
from tqdm import tqdm
from optimiter import SGD
from util import accuracy
from PIL import Image
model=CNN()
batch_size=1
n_epochs=10

def train(model, batch_size, n_epochs,
          lr=1e-4,
          lr_decay=0.1,
          momentum=0.0,
          wd=0.0,
          verbose=True,
          print_level=1,
          ):  # 模型训练
    print("开始训练...")
    n_train =1  # 获取训练集样本数量
    iterations_per_epoch = max(n_train // batch_size, 1)  # 每个epoch的迭代次数
    n_iterations = n_epochs * iterations_per_epoch  # 总的迭代次数
    loss_hist = []  # 储存每个epoch的损失函数值
    acc_hist = []
    opt_params = {"lr": lr, "weight_decay": wd, "momentum": momentum}  # 设置最优参数, 预设了学习率
    print("训练样本数：{}".format(n_train))
    print("每阶段迭代次数：{}".format(iterations_per_epoch))
    print("阶段数：{}".format(n_epochs))
    print("总迭代次数：{}".format(n_iterations))
    count = 0
    for epoch in tqdm(range(n_epochs)):  # 遍历所有epoch
        img_path ="D:\dataset\car\\train\\bus\\0aa8c6bece0ab6ceda43e920634e4fa2.jpg"
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img).transpose(2, 0, 1).astype("float")
        img = img / 255.0
        img = img[np.newaxis,:]
        label = 0
        labels = np.zeros((1, 9), dtype="int64")
        labels = np.insert(labels, label, 1)
        # print(label)
        count += 1
        loss, score = model.oracle(img, labels)  # 评估函数值与梯度
        loss_hist.append(loss)
        sgd = SGD(model.param_groups, **opt_params)  # 定义优化器
        sgd.step()  # 进行随机梯度下降

        train_acc = accuracy(score,labels)  # 计算精度
        acc_hist.append(train_acc)
        print("(Iteration {}/{},epoch {})loss:{},accu:{}".format(
                count, n_iterations, epoch+1, loss_hist[-1], train_acc))
        if count == iterations_per_epoch - 1: opt_params["lr"] *= lr_decay
    print("epoch {}.loss:{}.accu:{}".format(
         count + 1, np.sum(loss_hist)/iterations_per_epoch,np.sum(acc_hist)/iterations_per_epoch ))

#train(model,batch_size,n_epochs)

from dataset import eval_loader
def eval(val_loader):
    predict_labels = []
    labels = []
    for index, data in tqdm(enumerate(eval_loader)):
        x, y = data
        logits = model.score(x)
        accs=accuracy(logits,y)
    print("val dataset accuracy:", accs)
    return accs
eval(eval_loader)'''

####################################全连接####################################

from loss import SoftmaxCE, softmax
from layers import Conv, ELU, MaxPool, Linear,BN
import numpy as np
from optimiter import SGD
from util import accuracy
from matplotlib import pyplot as plt
from dataset import train_loader,eval_loader
from tqdm import tqdm
from main import load,save
class CNN(object):  # 卷积神经网络架构类: 层（卷积+激活+池化）+3层（线性+激活+线性）+softmax
    def __init__(self,
                 image_size=(3,224,224),
                 channels=3,
                 hidden_units1=512,
                 hidden_units2=512,
                 n_classes=10,
                 ):  # 构造函数: 初始化神经网络,定义网络层
        """ 类构造参数 """
        self.image_size = image_size  # 图片形状3×H×W
        self.channels = channels  # 卷积层的轨道数
        self.hidden_units1 = hidden_units1  # 线性传播中隐层单元数量
        self.hidden_units2 = hidden_units2  # 线性传播中隐层单元数量
        self.n_classes = n_classes  # 分类总数
        """ 类常用参数 """
        channel, height, width = self.image_size  # 这三个变量将记录卷积部分的输入轨道与维度

        #print("height={}.width={}".format(height,width))
        self.linear1 = Linear(  # 线性层 A: 3×56×56-->9408-->128
            in_features=channel * height * width,
            out_features=self.hidden_units1,
            init_scale=1e-2,
        )
        self.BN4 = BN()
        self.relu4 = ELU()  # 激活层 B: 128-->128
        self.linear2 = Linear(  # 线性层 B: 128-->10
            in_features=self.hidden_units1,
            out_features=hidden_units2,  # 最后一层应该是输出对每个字段的预测概率
            init_scale=1e-2,
        )

        self.relu5 = ELU()  # 激活层 B: 128-->128
        self.linear3 = Linear(  # 线性层 B: 128-->10
            in_features=self.hidden_units2,
            out_features=10,  # 最后一层应该是输出对每个字段的预测概率
            init_scale=1e-2,
        )

        """ 类初始化 """
        self.softmaxce = SoftmaxCE()
        self.param_groups = [  # 卷积层与线性层有参数(5+3)
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

    def oracle_full(self, x, y):  # 计算损失函数值,输出得分,损失函数梯度: x为一个N_samples×N_channels×Height×Width的张量,y为类别标签
        """ 前向传播 """

        """ 线性层 """
        linear1_out = self.linear1.forward(x)


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
        # print(self.param_groups)
        # print(self.param_groups[0]["w"]["pregrad"])
        # print("self.param_groups[0] =",self.param_groups[0])

        relu4_back = self.relu4.backward(linear2_back, linear1_out)

        linear1_back = self.linear1.backward(relu4_back, x)

        self.update_param()
        # print("l2 w_grad shape=",self.param_groups[3]["w"]["grad"].shape)
        return fx, s

    def score(self, x):  # 预测的得分,除了oracle函数外还需要一个另外的得分函数,这在检查精度时是有用的: x为输入特征
        linear1_out = self.linear1(x)
        relu4_out = self.relu4(linear1_out)
        linear2_out = self.linear2(relu4_out)
        relu5_out = self.relu5(linear2_out)
        linear3_out = self.linear3(relu5_out)
        #print(linear2_out)
        s = softmax(linear3_out)
        return s


    def update_param(self, ):  # 更新参数及梯度
        self.param_groups[0] = self.linear1.params
        self.param_groups[1] = self.linear2.params
        self.param_groups[2] = self.linear3.params

def train(model, batch_size, n_epochs,
          lr=0.0001,
          lr_decay=0.99,
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
            loss, score = model.oracle_full(img, y)  # 评估函数值与梯度
            running_loss = np.sum(loss)
            np.save(f'loss/loss_epoch_{epoch}', running_loss)  # 保存loss数据
            loss_hist.append(loss)
            sgd = SGD(model.param_groups, **opt_params)  # 定义优化器
            sgd.step()  # 进行随机梯度下降
            if verbose and t % print_level == 0:  # 输出训练损失
                train_acc = accuracy(score, y)  # 计算精度
                acc_hist.append(train_acc)
                np.save('acc/acc_epoch_{}'.format(epoch), train_acc)  # 保存acc数据
                print("(Iteration {}/{},epoch {})loss:{},accu:{}".format(
                    count, n_iterations, epoch+1, loss_hist[-1], train_acc))
            if t == iterations_per_epoch - 1: opt_params["lr"] *= lr_decay
        print("epoch {}.loss:{}.accu:{}".format(
             epoch + 1, np.sum(loss_hist)/iterations_per_epoch,np.sum(acc_hist)/iterations_per_epoch ))
        acc_hist=[]
        loss_hist=[]

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
    pred = np.array(predict_labels,dtype=object)
    labels = np.array(labels,dtype=object)
    print("pred=", pred)
    print("labels=",labels)
    acc=0
    for i in range(len(pred)):
        acc += np.sum(pred[i] == labels[i])
    print("acc=",acc)
    acc = acc /200
    print("val dataset accuracy:", acc)
    return acc

model=CNN()
load_file = 'ccc.npz'
if os.path.isfile(load_file):
    load(model.param_groups, load_file)
    print("模型加载成功！")
train(model,8,10)
save(model.param_groups, load_file)
print("模型保存成功！")
eval(eval_loader)



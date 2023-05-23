[啊](https://blog.csdn.net/JMU_Ma/article/details/97935299)

![image-20230314222316862](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20230314222316862.png)



### UNet

来源：

作用：

网络结构：

创新点：



### Swin-Unet

它是一个类unet的纯Transformer，用于医学图像分割。



### nnformer

基于transformer的3D医学图像分割模型。

transformer一般用于自然语言处理（NLP)，但是也可以用于3D医学图像分割，在非卷积上效果比较明显。





### nnUNet  //no new net

结构修改的越多，越容易过拟合

应该关注于能提高模型性能和泛化性的其他方面

把重心放在：预处理（resamping和normalization）、训练（loss、optimizer设置、数据增广）、推理（patch-based策略、test-time-augmentations集成和模型集成等）、后处理（如增强单连通域等）。

在医疗影像中，预处理和后处理的作用有时候大于网络本身的改造。

#### 网络结构

 作者提出了3个网络，分别是2D U-Net、3D U-Net和级联Unet。2D和3D U-Net可以生成全分辨率的结果，级联网络的第一级产生一个低分辨率结果，第二级对它进行优化。

- 改动：使用Leaky ReLU(neg.slope 1e-2)取代ReLU，使用Instance normalization取代Batchnormalization([解释](https://link.zhihu.com/?target=https%3A//blog.csdn.net/z13653662052/article/details/84503024))
- **2D U-Net**: 既然使用了3D U-Net，为啥还有用2D的。因为有[证据](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1707.00587.pdf)（这证据也是作者发现的）表明**当数据是各向异性**的时候，传统的3D分割方法就会很差。

![img](https://img-blog.csdnimg.cn/20191117100643616.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTQyNDkyNg==,size_16,color_FFFFFF,t_70#pic_center)

左侧压缩路径，右侧扩展路径

下采样就是压缩路径，将特征压缩成有特征组成的特征图。

上采样就是扩展路径，将提取的特征图解码为和原来图像尺寸一致的分割后的预测图像。

copy and crop的操作：将压缩的特征图叠加在扩展路径上同尺寸大小的特征图上再进行卷积核上采样操作，以此来整合更多信息进行图像分割。

输入572*572，输出388**388因为用了“Overlap-tile”，比输入尺寸大一点，如果超出边缘部分，就使用镜像方式padding,保留轮廓完整性。



- **3D U-Net**: 3D网络固然好，就是太占用GPU显存。那我们可以使用**小一点的图像块**去训练，但这对于那些较大的图像例如肝脏，这种基于块的方法就会阻碍训练。这是因为受限于感受野的大小，网络结构不能收集足够的上下文信息去正确的识别肝脏和其他器官。

![img](https://img-blog.csdnimg.cn/img_convert/1ebe07f2663a5efc46ee43cf415f8904.png) 

2D和3D区别：通道数翻倍的时刻和反卷积操作不同

（1）在2D U-net中，通道数翻倍的时刻是在下采样后的第一次卷积时；在3D U-net中，通道数翻倍的时刻是在下采样或上采样前的卷积中。

（2）对于反卷积操作，2D U-net中通道数减半，而3D U-net中通道数不变。

（3）另外，**3D U-net还使用batch normalization来加快收敛和避免网络结构的瓶颈。**



- **级联3D U-Net**：为了解决3D U-Net在**大图像尺寸**数据集上的缺陷，本文提出了级联模型。首先第一级3D U-Net在下采样的图像上进行训练，然后将结果上采样到原始的体素spacing。将上采样的结果作为一个额外的输入通道（one-hot编码）送入第二级3D U-Net，并使用基于图像块的策略在全分辨率的图像上进行训练。





### V-Net：用于三维医学图像分割的全卷积神经网络

背景：cnn被用于图像分割，早期是通过斑块式图像分类来获得图像或体积中的解剖划线。第一，因为是只考虑局部情况，所以很容易失败；第二，效率低，不必要的计算太多。

来源：因为cnn只能处理2D图像，而医学影像大多数都是3D的，所以提出了一种基于体积、全卷积、神经网络的三维图像分割方法。

问题1：处理前景和背景体素数量严重不平衡

办法1：提出了一种新的基于Dice系数的目标函数，并在训练过程中对其进行了优化

问题2：处理有限数量的可用于训练的标签图

办法2：使用**随机非线性转换**和**直方图匹配**来增加数据

较浅的层捕获局部信息，而较深的层使用卷积核，其接受域更广，因此捕获全局信息



自动描述感兴趣的器官和结构通常是执行诸如视觉增强[10]，计算机辅助诊断[12]，干预等任务所必需的

[1]图像定量指标的提取。

使用体积卷积代替，避免输入体积切片；基于Dice系数最大化在训练过程中进行优化。



**创新点**：1、引入残差（真实值-预估值=残差），在每个stage中，Vnet采用了ResNet（残差神经网络）的短路连接方式（水平方向的残差链接使用element-wise）；2、卷积层代替上采样和下采样的池化层。



为什么卷积代替池化？

[Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)简单说明卷积层去掉池化几乎没有影响，而将池化层换成卷积层的话准确度还会有提升。

从最初的alexnet在每个卷积之后使用一个池化，到后来的两三个卷积后使用一个池化，再到后来的一个block(多个卷积层)后使用一个池化。性能好的网络在使用越来越少的池化层，道理其实很明显，池化层损失的信息更多，使用卷积代替池化效果更好。

但是，尽管如此，还是要保留池化层的，毕竟卷积层是有参数的，而池化层却没有，完全使用卷积层代替池化层，可想参数量是多么大。

![image-20230314222937415](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20230314222937415.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------------

![img](https://img-blog.csdnimg.cn/20191117134856525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTQyNDkyNg==,size_16,color_FFFFFF,t_70)



### [ResNet残差神经网络](https://zhuanlan.zhihu.com/p/463935188)

产生原因：也许网络层数过多，会出现过拟合问题，达到负优化-->如果多出来的层数是恒等映射，甚至再好一点，更接近最优函数，就起到了正则化的作用。



#### [网络构成](https://blog.csdn.net/m0_53874746/article/details/121328429)

网络大体上由以下两种block构成：

![img](https://img-blog.csdnimg.cn/img_convert/2868ebcab043b959fdfcad6b443c325c.png)
Basicblock（用于resnet18，resnet34)

```
class Basicblock(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(Basicblock, self).__init__()
        self.stride = stride
        self.conv0 = Conv2D(in_channel, out_channel, 3, stride = stride, padding = 1)
        self.conv1 = Conv2D(out_channel, out_channel, 3, stride=1, padding = 1)
        self.conv2 = Conv2D(in_channel, out_channel, 1, stride = stride)
        self.bn0 = BatchNorm2D(out_channel)
        self.bn1 = BatchNorm2D(out_channel)
        self.bn2 = BatchNorm2D(out_channel)

    def forward(self, inputs):
        y = inputs
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.stride == 2:
            y = self.conv2(y)
            y = self.bn2(y)
        z = F.relu(x+y)
        return z

```

![img](https://img-blog.csdnimg.cn/img_convert/be25fad6a3ba8fd9b799b251deb5d04d.png)
Bottleneckblock（用于resnet50，resnet101，resnet152）

```
class Bottleneckblock(paddle.nn.Layer):
    def __init__(self, inplane, in_channel, out_channel, stride = 1, start = False):
        super(Bottleneckblock, self).__init__()
        self.stride = stride
        self.start = start
        self.conv0 = Conv2D(in_channel, inplane, 1, stride = stride)
        self.conv1 = Conv2D(inplane, inplane, 3, stride=1, padding=1)
        self.conv2 = Conv2D(inplane, out_channel, 1, stride=1)
        self.conv3 = Conv2D(in_channel, out_channel, 1, stride = stride)
        self.bn0 = BatchNorm2D(inplane)
        self.bn1 = BatchNorm2D(inplane)
        self.bn2 = BatchNorm2D(out_channel)
        self.bn3 = BatchNorm2D(out_channel)

    def forward(self, inputs):
        y = inputs
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.start:
            y = self.conv3(y)
            y = self.bn3(y)
        z = F.relu(x+y)
        return z
```

残差函数：(H(x)为最优函数，F(x)为残差函数)
$$
F(x)=H(x)-x
$$

#####  残差块单元

​	一个残差块有2条路径 F（x） 和 x，F（x） 路径拟合残差，不妨称之为残差路径；x路径为`identity mapping`恒等映射，称之为`shortcut`。图中的⊕为`element-wise addition`，要求参与运算的 F(x) 和 x 的尺寸要相同。



![img](https://production-media.paperswithcode.com/methods/resnet-e1548261477164_2_mD02h5A.png)

`shortcut` 路径大致可以分成 2 种，取决于残差路径是否改变了`feature map`数量和尺寸。

- 一种是将输入`x`原封不动地输出。
- 另一种则需要经过 1×1 卷积来升维或者降采样，主要作用是将输出与 F(x) 路径的输出保持`shape`一致，对网络性能的提升并不明显。

![image-20230312163340800](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20230312163340800.png)

为什么我们的深度残差网络可以很容易地从深度的大幅增加中获得精度收益，产生的结果比以前的网络要好得多？



<img src="C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20230310171823881.png" alt="image-20230310171823881"  />

**ResNet训练CIFAR10数据集的[pytorch](https://zhuanlan.zhihu.com/p/225597229)实现**

```
import os
import datetime
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image
import time
import argparse

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
 
batch_size = 128
path = './'
train_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),# 随机剪切成227*227
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

traindir = os.path.join(path, 'train')
valdir = os.path.join(path, 'val')
 
train_set = torchvision.datasets.CIFAR10(
    traindir, train=True, transform=train_transform, download=True)
valid_set = torchvision.datasets.CIFAR10(
    valdir, train=False, transform=val_transform, download=True)
 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
 
dataloaders = {
    'train': train_loader,
    'valid': valid_loader,
#     'test': dataloader_test
    }
 
dataset_sizes = {
    'train': len(train_set),
    'valid': len(valid_set),
#     'test': len(test_set)
    }
print(dataset_sizes)

def train(model, criterion, optimizer, scheduler, device, num_epochs, dataloaders,dataset_sizes):
    model = model.to(device)
    print('training on ', device)
    since = time.time()
 
    best_model_wts = []
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        # 训练模型
        s = time.time()
        model,train_epoch_acc,train_epoch_loss = train_model(
            model, criterion, optimizer, dataloaders['train'], dataset_sizes['train'], device)
        print('Epoch {}/{} - train Loss: {:.4f}  Acc: {:.4f}  Time:{:.1f}s'
            .format(epoch+1, num_epochs, train_epoch_loss, train_epoch_acc,time.time()-s))
        # 验证模型
        s = time.time()
        val_epoch_acc,val_epoch_loss = val_model(
            model, criterion, dataloaders['valid'], dataset_sizes['valid'], device)
        print('Epoch {}/{} - valid Loss: {:.4f}  Acc: {:.4f}  Time:{:.1f}s'
            .format(epoch+1, num_epochs, val_epoch_loss, val_epoch_acc,time.time()-s))
        # 每轮都记录最好的参数.
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()
        # 优化器
#         if scheduler not in None:
#             scheduler.step()
        # 保存画图参数
        train_losses.append(train_epoch_loss.to('cpu'))
        train_acc.append(train_epoch_acc.to('cpu'))
        val_losses.append(val_epoch_loss.to('cpu'))
        val_acc.append(val_epoch_acc.to('cpu'))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)
    return model
 
def train_model(model, criterion, optimizer, dataloader, dataset_size,device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs,labels in dataloader:
        optimizer.zero_grad()
        # 输入的属性
        inputs = Variable(inputs.to(device))
        # 标签
        labels = Variable(labels.to(device))
        # 预测
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        # 计算损失
        loss = criterion(outputs,labels)
        #梯度下降
        loss.backward()
        optimizer.step()
 
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)
 
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    
    return model,epoch_acc,epoch_loss
 
def val_model(model, criterion, dataloader, dataset_size, device):
    model.eval（)
    running_loss = 0.0
    running_corrects = 0
    for (inputs,labels) in dataloader:
        # 输入的属性
        inputs = Variable(inputs.to(device))
        # 标签
        labels = Variable(labels.to(device))
        # 预测
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        # 计算损失
        loss = criterion(outputs,labels)
        
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)
 
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    
    return epoch_acc,epoch_loss
class ResNet(nn.Module):
 
    def __init__(self):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fn = nn.Flatten()
        self.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        out = self.b1(x)
        
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        
        out = self.avgpool(out)
        out = self.fn(out)
        out = self.fc(out)
        return out
    
# 种类型的网络： 一种是当use_1x1conv=False时，应用ReLU非线性函数之前，将输入添加到输出。 
# 另一种是当use_1x1conv=True时，添加通过卷积调整通道和分辨率
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        out += x
        self.relu(out)
        return out

# 残差块
def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
X = torch.randn(1, 3, 224, 224)
net = ResNet()
net = nn.Sequential(net.b1, net.b2, net.b3, net.b4, net.b5,
                    net.avgpool,net.fn, net.fc)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
val_losses,val_acc = [],[]
train_losses,train_acc = [],[]
 
lr,num_epochs = 0.01,30
model = ResNet()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

model = train(model, criterion, optimizer, None ,
              try_gpu(), num_epochs, dataloaders, dataset_sizes)

lr,num_epochs = 0.001,10
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
model = train(model, criterion, optimizer, None ,
              try_gpu(), num_epochs, dataloaders, dataset_sizes)

plt.plot(range(1, len(train_losses)+1),train_losses, 'b', label='training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='val loss')
plt.legend()

plt.plot(range(1,len(train_acc)+1),train_acc,'b--',label = 'train accuracy')
plt.plot(range(1,len(val_acc)+1),val_acc,'r--',label = 'val accuracy')
plt.legend()
```





我的理解：

​		一般来说，增加网络深度效果会比增加网络广度好，所以不断的增加层数。但是过多的增加层数效果会变差，网络太深就很难训练，因为反向传播的误差会越来越大，很有可能会梯度消失或者梯度爆炸，导致精度下降。

​		所以残差网络就是当前面的准确率已经接近饱和的时候，像是复制他的效果（恒等映射层），后期目标就是将残差结果逼近0，这样误差就不会增加，起码精度不会下降。

​		而且，虽然网络层数变深了，但是因为对输入结果更加敏感（因为从0.1到0的变化比从5.1到5的变化明显，所以说更加敏感），所以收敛速度也不会变得很慢。

​		残差主要突出的是微小的变化，这样对权重的调整效果更大。而且我们不知道什么时候的深度达到最佳，但起码准确率不会下降。





### [医学图像分割基础](https://www.zhihu.com/question/427767524)

一般都是CT图像和MRI图像（核磁共振）。

DICOM的常用Tag



[为什么](https://zhuanlan.zhihu.com/p/104613999)

ResNet（Residual Neural Network）由微软研究院的Kaiming He等四名华人提出，通过使用ResNet Unit成功训练出了152层的神经网络，并在ILSVRC2015比赛中取得冠军，在top5上的错误率为3.57%，同时参数量比VGGNet低，效果非常突出。ResNet的结构可以极快的加速神经网络的训练，模型的准确率也有比较大的提升。同时ResNet的推广性非常好，甚至可以直接用到InceptionNet网络中。ResNet的主要思想是在网络中增加了直连通道，即Highway Network的思想。

而无需修改求解器。

ResNet的主要思想是在网络中增加了直连通道，即Highway Network的思想。此前的网络结构是性能输入做一个非线性变换，而Highway Network则允许保留之前网络层的一定比例的输出。ResNet的思想和Highway Network的思想也非常类似，允许原始输入信息直接传到后面的层中，如下图所示。


![../../_images/model_develop_flow.png](https://www.paddlepaddle.org.cn/documentation/docs/zh/_images/model_develop_flow.png)

### tensor张量

支持运行在 CPU 上，还支持运行在 GPU 及各种 AI 芯片上，以实现计算加速

#### 创建方式

##### 1.指定数据创建

与 Numpy 创建数组方式类似，通过给定 Python 序列（如列表 list、元组 tuple），可使用 [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/to_tensor_cn.html) 创建任意维度的 Tensor。

###### 一维（类似向量）

```
import paddle
Tensor=paddle.to_tensor([2.0,3.0,4.0])
paddle.to_tensor(2)
paddle.to_tensor([2])
```

###### 二维（类似矩阵）

```
ndim_2_Tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
```

###### 三维

```
ndim_3_Tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
```

<img src="https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/guides/01_paddle2.0_introduction/basic_concept/images/Tensor_2.0.png?raw=true" alt="img" style="zoom:150%;" />

##### 2.指定形状创建

```
paddle.zeros([m, n])             # 创建数据全为 0，形状为 [m, n] 的 Tensor
paddle.ones([m, n])              # 创建数据全为 1，形状为 [m, n] 的 Tensor
paddle.full([m, n], 10)          # 创建数据全为 10，形状为 [m, n] 的 Tensor
```

##### 3.指定区间创建

```
paddle.arange(start, end, step)  # 创建以步长 step 均匀分隔区间[start, end)的 Tensor
paddle.linspace(start, stop, num) # 创建以元素个数 num 均匀分隔区间[start, stop)的 Tensor
```

- **创建一个空 Tensor**，即根据 shape 和 dtype 创建尚未初始化元素值的 Tensor，可通过 [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/empty_cn.html) 实现。
- **创建一个与其他 Tensor 具有相同 shape 与 dtype 的 Tensor**，可通过 [paddle.ones_like](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/ones_like_cn.html) 、 [paddle.zeros_like](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/zeros_like_cn.html) 、 [paddle.full_like](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/full_like_cn.html) 、[paddle.empty_like](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/empty_like_cn.html) 实现。
- **拷贝并创建一个与其他 Tensor 完全相同的 Tensor**，可通过 [paddle.clone](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/clone_cn.html) 实现。
- **创建一个满足特定分布的 Tensor**，如 [paddle.rand](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/rand_cn.html), [paddle.randn](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/randn_cn.html) , [paddle.randint](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/randint_cn.html) 等。
- **通过设置随机种子创建 Tensor**，可每次生成相同元素值的随机数 Tensor，可通过 [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/seed_cn.html) 和 [paddle.rand](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/rand_cn.html) 组合实现。

##### 4.指定图像、文本数据创建

- 对于图像场景，可使用 [paddle.vision.transforms.ToTensor](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/vision/transforms/ToTensor_cn.html) 直接将 PIL.Image 格式的数据转为 Tensor，使用 [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/to_tensor_cn.html) 将图像的标签（Label，通常是 Python 或 Numpy 格式的数据）转为 Tensor。
- 对于文本场景，需将文本数据解码为数字后，再通过 [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/to_tensor_cn.html) 转为 Tensor。不同文本任务标签形式不一样，有的任务标签也是文本，有的则是数字，均需最终通过 paddle.to_tensor 转为 Tensor。

##### 5.自动创建

有一些 API 封装了 Tensor 创建的操作，从而无需用户手动创建 Tensor

 [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/io/DataLoader_cn.html) 能够基于原始 Dataset，返回读取 Dataset 数据的迭代器，迭代器返回的数据中的每个元素都是一个 Tensor

 [paddle.Model.fit](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/Model_cn.html) 、[paddle.Model.predict](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/Model_cn.html) ，如果传入的数据不是 Tensor，会自动转为 Tensor 再进行模型训练或推理

##### 6.tensor和numpy数组相互转换

将 Tensor 转换为 Numpy 数组，可通过 [Tensor.numpy](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/Tensor_cn.html#numpy) 方法实现。

#### 属性

##### 形状（shape）

- `Tensor.shape`查看形状

- `paddle.reshape`重置形状

- `paddle.squeeze`，可实现 Tensor 的降维操作，即把 Tensor 中尺寸为 1 的维度删除。
- `paddle.unsqueeze`，可实现 Tensor 的升维操作，即向 Tensor 中某个位置插入尺寸为 1 的维度。
- `paddle.flatten`，将 Tensor 的数据在指定的连续维度上展平。
- `paddle.transpose`，对 Tensor 的数据进行重排。

##### 数据类型（dtype)

- `Tensor.dtype`查看数据类型
- `paddle.cast`改变数据类型

```
float32_Tensor = paddle.to_tensor(1.0)

float64_Tensor = paddle.cast(float32_Tensor, dtype='float64')
```

##### 设备位置（place）

- `Tensor.place`指定分配位置
- `paddle.device.set_device`设置全局默认设备位置

##### 名称（name）

在每个 Tensor 创建时，会自定义一个独一无二的名称。

##### stop_gradient属性

stop_gradient 表示是否停止计算梯度，默认值为 True，表示停止计算梯度，梯度不再回传。在设计网络时，如不需要对某些参数进行训练更新，可以将参数的 stop_gradient 设置为 True

```
eg = paddle.to_tensor(1)
eg.stop_gradient
```

#### 操作

##### 索引和切片

索引或切片的第一个值对应第 0 维，第二个值对应第 1 维，依次类推，如果某个维度上未指定索引，则默认为 `:` 

注意：通过索引或切片修改 Tensor，该操作会**原地**修改该 Tensor 的数值，且原值不会被保存。如果被修改的 Tensor 参与梯度计算，仅会使用修改后的数值，这可能会给梯度计算引入风险。飞桨框架会自动检测不当的原位（inplace）使用并报错。

##### 数学运算

```
x.abs()                       #逐元素取绝对值
x.ceil()                      #逐元素向上取整
x.floor()                     #逐元素向下取整
x.round()                     #逐元素四舍五入
x.exp()                       #逐元素计算自然常数为底的指数
x.log()                       #逐元素计算 x 的自然对数
x.reciprocal()                #逐元素求倒数
x.square()                    #逐元素计算平方
x.sqrt()                      #逐元素计算平方根
x.sin()                       #逐元素计算正弦
x.cos()                       #逐元素计算余弦
x.add(y)                      #逐元素相加
x.subtract(y)                 #逐元素相减
x.multiply(y)                 #逐元素相乘
x.divide(y)                   #逐元素相除
x.mod(y)                      #逐元素相除并取余
x.pow(y)                      #逐元素幂运算
x.max()                       #指定维度上元素最大值，默认为全部维度
x.min()                       #指定维度上元素最小值，默认为全部维度
x.prod()                      #指定维度上元素累乘，默认为全部维度
x.sum()                       #指定维度上元素的和，默认为全部维度

```

关于python数学运算重写

```
x + y  -> x.add(y)            #逐元素相加
x - y  -> x.subtract(y)       #逐元素相减
x * y  -> x.multiply(y)       #逐元素相乘
x / y  -> x.divide(y)         #逐元素相除
x % y  -> x.mod(y)            #逐元素相除并取余
x ** y -> x.pow(y)            #逐元素幂运算

```

##### 逻辑运算

```
x == y  -> x.equal(y)         #判断两个 Tensor 的每个元素是否相等
x != y  -> x.not_equal(y)     #判断两个 Tensor 的每个元素是否不相等
x < y   -> x.less_than(y)     #判断 Tensor x 的元素是否小于 Tensor y 的对应元素
x <= y  -> x.less_equal(y)    #判断 Tensor x 的元素是否小于或等于 Tensor y 的对应元素
x > y   -> x.greater_than(y)  #判断 Tensor x 的元素是否大于 Tensor y 的对应元素
x >= y  -> x.greater_equal(y) #判断 Tensor x 的元素是否大于或等于 Tensor y 的对应元素

```

##### 线性代数

```
x.t()                         #矩阵转置
x.transpose([1, 0])           #交换第 0 维与第 1 维的顺序
x.norm('fro')                 #矩阵的弗罗贝尼乌斯范数
x.dist(y, p=2)                #矩阵（x-y）的 2 范数
x.matmul(y)                   #矩阵乘法

```



### 数据集定义和加载

![../../_images/data_pipeline.png](https://www.paddlepaddle.org.cn/documentation/docs/zh/_images/data_pipeline.png)

#### 定义数据集

将磁盘中保存的原始图片、文字等样本和对应的标签映射到 Dataset，方便后续通过索引（index）读取数据，在 Dataset 中还可以进行一些数据变换、数据增广等预处理操作。在飞桨框架中推荐使用 [paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html#dataset) 自定义数据集，另外在 [paddle.vision.datasets](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html#api) 和 [paddle.text](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/text/Overview_cn.html#api) 目录下飞桨内置了一些经典数据集方便直接调用

##### 内置数据集

```
计算机视觉（CV）相关数据集： ['DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
自然语言处理（NLP）相关数据集： ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16', 'ViterbiDecoder', 'viterbi_decode']
```

##### 定义数据集Dataset

可构建一个子类继承自 `paddle.io.Dataset` ，并且实现下面的三个函数：

1. `__init__`：完成数据集初始化操作，将磁盘中的样本文件路径和对应标签映射到一个列表中。
2. `__getitem__`：定义指定索引（index）时如何获取样本数据，最终返回对应 index 的单条数据（样本数据、对应的标签）。
3. `__len__`：返回数据集的样本总数。

在 `__init__` 函数和 `__getitem__` 函数中还可实现一些数据预处理操作

##### 定义数据读取器dataloader

使用paddle.io.DataLoader对数据集进行多进程的读取，并且可自动完成划分 batch 的工作

```
# 定义并初始化数据读取器
train_loader = paddle.io.DataLoader(train_custom_dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

# 调用 DataLoader 迭代读取数据
for batch_id, data in enumerate(train_loader()):
    images, labels = data
    print("batch_id: {}, 训练数据shape: {}, 标签数据shape: {}".format(batch_id, images.shape, labels.shape))
    break

```

##### 自定义采样器

飞桨框架在 [paddle.io](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Overview_cn.html) 目录下提供了多种采样器，如批采样器 [BatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/BatchSampler_cn.html)、分布式批采样器 [DistributedBatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DistributedBatchSampler_cn.html)、顺序采样器 [SequenceSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/SequenceSampler_cn.html)、随机采样器 [RandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/RandomSampler_cn.html) 等

```
from paddle.io import BatchSampler

# 定义一个批采样器，并设置采样的数据集源、采样批大小、是否乱序等
bs = BatchSampler(train_custom_dataset, batch_size=8, shuffle=True, drop_last=True)
# 在 DataLoader 中使用 BatchSampler 获取采样数据   
train_loader = paddle.io.DataLoader(train_custom_dataset, batch_sampler=bs, num_workers=1)
```



#### 迭代读取数据集

自动将数据集的样本进行分批（batch）、乱序（shuffle）等操作，方便训练时迭代读取，同时还支持多进程异步读取功能可加快数据读取速度。在飞桨框架中可使用 [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader) 迭代读取数据集

### 数据预处理

![../../_images/data_preprocessing.png](https://www.paddlepaddle.org.cn/documentation/docs/zh/_images/data_preprocessing.png)

#### paddle.vision.transforms

在 [paddle.vision.transforms](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html#about-transforms) 下内置了数十种图像数据处理方法

图像数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomAffine', 'RandomRotation', 'RandomPerspective', 'Grayscale', 'ToTensor', 'RandomErasing', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'affine', 'rotate', 'perspective', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize', 'erase']

##### 单个使用

```
from paddle.vision.transforms import Resize

# 定义一个待使用的数据处理方法，这里定义了一个调整图像大小的方法
transform = Resize(size=28)
```

##### 多个组合使用

先定义好每个数据处理方法，然后用`Compose` 进行组合

```
from paddle.vision.transforms import Compose, RandomRotation

# 定义待使用的数据处理方法，这里包括随机旋转、改变图片大小两个组合处理
transform = Compose([RandomRotation(10), Resize(size=32)])
```

#### 在数据集中应用数据预处理操作

##### 1.框架内置数据集

前面定义好transform，加载时传递给`tranform`字段即可

```
# 通过 transform 字段传递定义好的数据处理方法，即可完成对框架内置数据集的增强
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

```



##### 2.自定义数据集

对于自定义的数据集，可以在数据集中将定义好的数据处理方法传入 `__init__` 函数，将其定义为自定义数据集类的一个属性，然后在 `__getitem__` 中将其应用到图像上

#### 数据预处理的几种方法介绍

##### CenterCrop

对输入图像进行裁剪，保持图片中心点不变。

```
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import CenterCrop

transform = CenterCrop(224)

image = cv2.imread('flower_demo.png')

image_after_transform = transform(image)
plt.subplot(1,2,1)
plt.title('origin image')
plt.imshow(image[:,:,::-1])
plt.subplot(1,2,2)
plt.title('CenterCrop image')
plt.imshow(image_after_transform[:,:,::-1])

```

##### RandomHorizontalFlip

基于概率来执行图片的水平翻转。

```
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import RandomHorizontalFlip

transform = RandomHorizontalFlip(0.5)

image = cv2.imread('flower_demo.png')

image_after_transform = transform(image)
plt.subplot(1,2,1)
plt.title('origin image')
plt.imshow(image[:,:,::-1])
plt.subplot(1,2,2)
plt.title('RandomHorizontalFlip image')
plt.imshow(image_after_transform[:,:,::-1])

```

##### ColorJitter

随机调整图像的亮度、对比度、饱和度和色调

```
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import ColorJitter

transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

image = cv2.imread('flower_demo.png')

image_after_transform = transform(image)
plt.subplot(1,2,1)
plt.title('origin image')
plt.imshow(image[:,:,::-1])
plt.subplot(1,2,2)
plt.title('ColorJitter image')
plt.imshow(image_after_transform[:,:,::-1])

```



### 模型组网



#### 直接使用内置模型

飞桨框架内置模型： ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext50_64x4d', 'resnext101_32x4d', 'resnext101_64x4d', 'resnext152_32x4d', 'resnext152_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'MobileNetV3Small', 'MobileNetV3Large', 'mobilenet_v3_small', 'mobilenet_v3_large', 'LeNet', 'DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenet264', 'AlexNet', 'alexnet', 'InceptionV3', 'inception_v3', 'SqueezeNet', 'squeezenet1_0', 'squeezenet1_1', 'GoogLeNet', 'googlenet', 'ShuffleNetV2', 'shufflenet_v2_x0_25', 'shufflenet_v2_x0_33', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenet_v2_swish']

```
# 模型组网并初始化网络
lenet = paddle.vision.models.LeNet(num_classes=10)

# 可视化模型组网结构和参数
paddle.summary(lenet,(1, 1, 28, 28))

```

通过 [paddle.summary](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/summary_cn.html#summary) 可清晰地查看神经网络层次结构、每一层的输入数据和输出数据的形状（Shape）、模型的参数量（Params）等信息，方便可视化地了解模型结构、分析数据计算和传递过程。

#### 使用 paddle.nn.Sequential 组网

构建顺序的线性网络结构（如 LeNet、AlexNet 和 VGG）时，可以选择该方式。相比于 Layer 方式 ，Sequential 方式可以用更少的代码完成线性网络的构建。

#### 使用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#layer) 组网

构建一些比较复杂的网络结构时，可以选择该方式。相比于 Sequential 方式，Layer 方式可以更灵活地组建各种网络结构。Sequential 方式搭建的网络也可以作为子网加入 Layer 方式的组网中

### 模型训练与评估

#### 模型训练

#### 模型评估

### 模型推理

#### 模型保存

#### 模型加载并执行推理


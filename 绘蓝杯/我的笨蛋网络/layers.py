# -*- coding:UTF-8 -*-
import abc
import numpy as np



class Layer(object):  # 神经网络层的基类,该类包含抽象方法,因此不可以被实例化
    def __init__(self):  # 抽象类构造函数
        self.params = dict()  # 参数字典: 其中键为参数名称的字符串,键为字典,键字典里面包含"param"与"grad"两个字段,分别记录参数值与梯度

    @abc.abstractmethod
    def forward(self, x):  # 评估输入特征与返回输出: x为输入特征, 返回值f(x)为输出特征
        pass

    @abc.abstractmethod#实例化
    def backward(self, grad_in,
                 x):  # 计算梯度并将梯度反向传播, 更新过的梯度被储存在self.params的对应区域中: grad_in为从反向传播得到的梯度, x为输入特征, 返回值grad_x为反向传播到下一层的梯度(分别为w.r.t x)
        pass

    def __call__(self, *args, **kwargs):  # 使得Layer类型的变量可调用
        return self.forward(*args, **kwargs)


class Conv(Layer):  # 卷积层类, 参数w为卷积过滤器, 参数b为偏差
    def __init__(self, height, width,
                 in_channels=3,
                 out_channels=3,
                 stride=1,
                 padding=1,
                 init_scale=1e-2,
                 ):  # 构造函数
        super(Conv, self).__init__()
        """ 类构造参数 """
        self.height = height  # 卷积核的高度
        self.width = width  # 卷积核的宽度
        self.in_channels = in_channels  # 输入轨道数
        self.out_channels = out_channels  # 输出轨道数
        self.stride = stride  # 卷积核移动步长
        self.padding = padding  # 是否需要在周围补充0
        self.init_scale = init_scale  # 初始规模
        """ 父类参数: 不在类初始化时设定grad参数, 因为如果重复调用同一对象的backward方法可能会导致梯度重复更新而错误 """
        self.params["w"] = {# 输出轨道数应当与卷积过滤器的数量相一致
            "param": np.random.randn(self.out_channels,self.in_channels, self.height, self.width)/np.sqrt(self.in_channels*self.width*self.height),# conv1:3×8×8
            "grad": None,
            "pregrad": None,
        }
        self.params["b"] = {  # 偏差值
            "param": np.zeros(self.out_channels,),# 3
            "grad": None,
            "pregrad": None,
        }

    def forward(self, x):  # 前向传播: x为一只四维张量, N_samples×n_Channels×Height×Width, 返回值out为卷积核的输出
        nSamples, nChannels, height, width = x.shape  # 获取输入张量x的四个维度值
        assert nChannels == self.in_channels  # 断言
        outshape = (  # 计算输出的形状
            nSamples, self.out_channels,  # 样本数, 输出轨道数
            int((2 * self.padding + height - self.height) // self.stride) + 1,  # 输出高度
            int((2 * self.padding + width - self.width) // self.stride) + 1  # 输出宽度
        )
        out = np.zeros(outshape)  # 初始化输出
        if self.padding:  # 如果需要padding
            x_ = np.zeros((
                nSamples, nChannels,
                height + 2 * self.padding,
                width + 2 * self.padding
            ))
            x_[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = x
        else:
            x_ = x.copy()
        for i in range(outshape[0]):  # 遍历样本
            for j in range(outshape[1]):  # 遍历输出轨道
                for k in range(outshape[2]):
                    for l in range(outshape[3]):
                        x1, y1 = k * self.stride, l * self.stride
                        x2, y2 = x1 + self.height, y1 + self.width
                        total = 0
                        for m in range(nChannels):
                            t1 = x_[i, m, x1:x2, y1:y2]  # 输入对应区域
                            t2 = self.params["w"]["param"][j, m, :, :]  # 卷积过滤器对应区域
                            total += np.nansum(t1 * t2)
                        out[i, j, k, l] = total + self.params["b"]["param"][j]  # 加上偏差值
        return out


    def backward(self, grad_in, x):  # 卷积层的反向传播: grad_in的维度与卷积层forward中输出的维度相同
        self.params["w"]["grad"] = np.zeros((
            self.out_channels, self.in_channels, self.height, self.width))  # 选择在反向传播时再重定义grad参数
        self.params["b"]["grad"] = np.zeros((self.out_channels,))  # 选择在反向传播时再重定义grad参数
        self.params["w"]["pregrad"] = np.zeros((
            self.out_channels, self.in_channels, self.height, self.width))  # 选择在反向传播时再重定义pregrad参数
        self.params["b"]["pregrad"] = np.zeros((self.out_channels,))  # 选择在反向传播时再重定义pregrad参数
        nSamples, nChannels, height, width = x.shape
        outshape = (  # 计算输出的形状: 也是grad_in的形状
            nSamples, self.out_channels,  # 样本数, 输出轨道数
            int((2 * self.padding + height - self.height) / self.stride) + 1,  # 输出高度
            int((2 * self.padding + width - self.width) / self.stride) + 1  # 输出宽度
        )
        assert outshape == grad_in.shape  # 断言
        x_ = np.zeros((  # 复现padding后的输入
            nSamples, nChannels,
            height + 2 * self.padding,
            width + 2 * self.padding
        ))
        if self.padding:  # 如果需要padding
            x_[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = x
        else:
            x_ = x.copy()
        grad_x = np.zeros_like(x_)  # 先设法对padding后的x求梯度, 然后只需取grad_x中间部分即可
        for i in range(outshape[0]):  # 遍历样本
            for j in range(outshape[1]):  # 遍历输出轨道
                self.params["b"]["grad"][j] += np.nansum(grad_in[i, j, :, :])
                for k in range(outshape[2]):  # 遍历像素点
                    for l in range(outshape[3]):  # grad_in的维度必然为outshape, 通过遍历前向传播中outshape中每个位置上元素表达式来反向求导
                        x1, y1 = k * self.stride, l * self.stride  #滑窗大小
                        x2, y2 = x1 + self.height, y1 + self.width
                        for m in range(nChannels):
                            grad_x[i, m, x1:x2, y1:y2] += grad_in[i, j,
                                                                  k, l] * self.params["w"]["param"][j, m, :, :]
                            self.params["w"]["grad"][j, m, :, :] += grad_in[
                                                                        i, j, k, l] * x_[i, m, x1:x2, y1:y2]
        grad_x = grad_x[:, :, self.padding:-self.padding,
                 self.padding:-self.padding] if self.padding else grad_x  # 取出grad_x中不是padding的部分作为将要传递下去的梯度
        return grad_x


class Linear(Layer):  # 线性层类, 用于对特征应用线性变换: w为n_in×n_out的矩阵, b为1×n_out的向量
    def __init__(self, in_features, out_features,
                 init_scale=1e-4):  # 构造函数
        super(Linear, self).__init__()
        """ 类构造参数 """
        self.in_features = in_features  # 输入结点
        self.out_features = out_features  # 输出结点
        self.init_scale = 1e-2
        """ 父类参数 """
        self.params["w"] = {  # 线性变换矩阵
            "param":np.random.randn(
                self.in_features, self.out_features)/np.sqrt(self.in_features),
            "grad": None,  # 在backward方法中初始化
            "pregrad":None,
        }
        self.params["b"] = {  # 常数项偏差
            "param":init_scale*np.random.randn(1, self.out_features),
            "grad": None,
            "pregrad": None,
        }

    def forward(self, x):  # 前向传播
        w = self.params["w"]["param"]
        b = self.params["b"]["param"]
        #print("w.shape=",w.shape)
        #print("b.shape=", b.shape)
        #print("x.shape=", x.shape)
        x_ = x.reshape(x.shape[0], -1)  # flatten层
        #print("x_.shape=", x_.shape)
        out = np.dot(x_, w) + b  # 这里两个维度分别为n_Sample×n_out与n_out×1, 但是它们还是可以相加的, 结果为每个sample加上b
        return out

    def backward(self, grad_in, x):  # 线性层的反向传播: 这里的grad_in维度与forward中out维度相同
        """
            out = np.dot(x,w) + b;
            x.shape = (n_Sample,in_features);
            w.shape = (in_features,out_features);
            out.shape = grad_in.shape = (n_Sample,out_features);
            b.shape = (1,out_features)
            b的梯度应该为(n_Sample,out_features)
        """
        x_ = x.reshape(x.shape[0], -1)
        self.params["w"]["grad"] = np.dot(x_.T, grad_in)
        self.params["b"]["grad"] = np.nansum(grad_in, axis=0)  # 每个样本都被b加了一次
        self.params["w"]["pregrad"] = np.zeros_like(self.params["w"]["grad"])
        self.params["b"]["pregrad"] = np.zeros_like(self.params["b"]["grad"])
        grad_x = np.dot(grad_in, self.params["w"]["param"].T)
        return grad_x


class ELU(Layer):  # 激活层类
    def __init__(self):  # 构造函数
        self.gamma=1/5.5
        super(ELU, self).__init__()

    def forward(self, x):  # 前向传播
        X = np.where(x > 0, x, self.gamma * (np.exp(x) - 1))
        return X

    def backward(self, grad_in, x):  # 反向传播
        X = np.where(x > 0, grad_in, self.gamma * grad_in * np.exp(x))
        return X  # 返回输入的梯度乘上relu的梯度


class MaxPool(Layer):  # 池化层类
    def __init__(self, kernel_size,
                 stride=2,
                 padding=0
                 ):  # 构造函数
        super(MaxPool, self).__init__()
        """ 类构造参数 """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        """ 父类参数 """
        self.params = dict()  # 池化层应该是没有参数

    def forward(self, x):  # 前向传播: x四维（ N_samples×n_Channels×Height×Width）, 返回值out为池化层的输出
        nSamples, nChannels, height, width = x.shape  # 获取输入x的四个维度值
        outshape = (  # 计算输出的形状
            nSamples, nChannels,  # 样本数, 输出轨道数
            int((2 * self.padding + height - self.kernel_size) / self.stride) + 1,  # 输出高度
            int((2 * self.padding + width - self.kernel_size) / self.stride) + 1  # 输出宽度
        )
        out = np.zeros(outshape)  # 初始化输出
        if self.padding:  # 如果需要padding
            x_ = np.zeros((
                nSamples, nChannels,
                height + 2 * self.padding,
                width + 2 * self.padding
            ))
            x_[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = x
        else:
            x_ = x.copy()
        for i in range(outshape[0]):  # 遍历样本
            for j in range(outshape[1]):  # 遍历输出轨道
                for k in range(outshape[2]):  # 遍历像素点
                    for l in range(outshape[3]):  # 虽然很蠢, 但是用迭代生成器也不是很方便操作感觉
                        x1, y1 = k * self.stride, l * self.stride
                        x2, y2 = x1 + self.kernel_size, y1 + self.kernel_size
                        out[i, j, k, l] = np.nanmax(x_[i, j, x1:x2, y1:y2])  # 取窗口最大值
        return out

    def backward(self, grad_in, x):  # 反向传播: 此事grad_in的维度恰与MaxPool的输出相同, 因此需要找出grad_in中每个元素对应了x中哪块区域
        nSamples, nChannels, height, width = x.shape  # 获取输入的形状
        outshape = (  # 计算输出的形状: 也是grad_in的形状
            nSamples, nChannels,  # 样本数, 输出轨道数
            int((2 * self.padding + height - self.kernel_size) / self.stride) + 1,  # 输出高度
            int((2 * self.padding + width - self.kernel_size) / self.stride) + 1  # 输出宽度
        )
        grad_in_reshape = grad_in.reshape(outshape)
        x_ = np.zeros((  # 复现padding后的输入
            nSamples, nChannels,
            height + 2 * self.padding,
            width + 2 * self.padding
        ))
        if self.padding:  # 如果需要padding
            x_[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = x
        else:
            x_ = x.copy()
        grad_x = np.zeros_like(x_)  # 先设法对padding后的x求梯度, 然后只需取grad_x中间部分即可

        for i in range(outshape[0]):  # 遍历样本
            for j in range(outshape[1]):  # 遍历输出轨道
                for k in range(outshape[2]):  # 遍历像素点
                    for l in range(outshape[3]):  # grad_in的维度必然为outshape, 通过遍历前向传播中outshape中每个位置上元素表达式来反向求导
                        x1, y1 = k * self.stride, l * self.stride
                        x2, y2 = x1 + self.kernel_size, y1 + self.kernel_size
                        maxgrid = np.nanmax(x_[i, j, x1:x2, y1:y2])  # 找出对应该grad_in格子的原区域中最大值
                        grad_x[i, j, x1:x2, y1:y2] += grad_in_reshape[i, j,
                                                                      k, l] * (x_[i, j, x1:x2, y1:y2] == maxgrid)

        grad_x = grad_x[:, :, self.padding:-self.padding,
                 self.padding:-self.padding] if self.padding else grad_x  # 取出grad_x中不是padding的部分作为将要传递下去的梯度
        return grad_x

class BN(Layer):

    def __init__(self,eps =1e-7, momentum =0.9, mode = "train"):
        super(BN).__init__()
        self.eps =eps
        #self.input = x
        #n, c, h, w = x.shape
        self.momentum = momentum
        self.running_mean = np.zeros(3)
        self.running_var = np.zeros(3)
        self.gamma = 1
        self.beta = 0
        self.mode = mode
        self.bn_param = dict(  # 配置字典中包含学习率与权重衰减等超参数
            eps=1e-5,
            momentum=0.9,
            running_mean=None,
            running_var=None
        )

    def add_dim(x, dim):
        return np.expand_dims(x, axis=dim) # batch

    def forward(self,x):
        ib, ic, ih, iw = x.shape

        x = x.transpose(1, 0, 2, 3).reshape([ic, -1]) # n,c,h,w ->c, n*h*w
        if self.mode =="train":

            self.mean = np.mean(x, axis=0) # 每个channel的均值
            self.mean = self.add_dim(self.mean, 1) # 与后面的self.input 维度一致
            self.var = np.var(x, axis=0) #每个channel的方差
            #self.var = np.sqrt(self.var + self.eps)
            self.var = self.add_dim(self.var , 1)
            self.gamma = self.add_dim(self.gamma, 1)
            self.beta = self.add_dim(self.beta, 1)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) *self.mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) *self.var
            self.input_ = (x -  self.running_mean)/(self.running_var + self.eps)
            dout = (self.input_*self.gamma +self.beta ).reshape(ic,ib, ih, iw).transpose(1, 0, 2, 3)
            self.cache = (self.input_, self.gamma, (x - self.running_mean, self.running_var + self.eps))
        elif self.mode =="test":
            x_hat = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
            dout = self.gamma * x_hat + self.beta
        else:
            raise ValueError("Invalid forward batch normlization mode")
        return dout, self.cache


    def backward(self, dout):
        N, D = dout.shape
        x_, gamma, x_minus_mean, var_plus_eps =self.cache

        # calculate gradients
        dgamma = np.sum(x_ * dout, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_ = np.matmul(np.ones((N,1)), gamma.reshape((1, -1))) * dout
        dx = N * dx_ - np.sum(dx_, axis=0) - x_ * np.sum(dx_ * x_, axis=0)
        dx *= (1.0/N) / np.sqrt(var_plus_eps)

        return dx, dgamma, dbeta


    def batchnorm_forward(self,x):
        # read some useful parameter
        N, D = x.shape
        eps = self.bn_param.get('eps', 1e-5)
        momentum = self.bn_param.get('momentum', 0.9)
        running_mean = self.bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = self.bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
        if self.bn_param['running_mean'] == None:
            running_mean = np.zeros(D, dtype=x.dtype)
            running_var = np.zeros(D, dtype=x.dtype)

        # BN forward pass
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        x_ = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = self.gamma * x_ + self.beta
        out=np.array(out)

        # update moving average
        running_mean = momentum * running_mean + (1-momentum) * sample_mean
        running_var = momentum * running_var + (1-momentum) * sample_var
        self.bn_param['running_mean'] = running_mean
        self.bn_param['running_var'] = running_var

        # storage variables for backward pass
        cache = (x_, self.gamma, x - sample_mean, sample_var + eps)

        return out, cache


    def batchnorm_backward(self,dout, cache):
        # extract variables
        N, D = dout.shape
        x_, gamma, x_minus_mean, var_plus_eps = cache

        # calculate gradients
        dgamma = np.sum(x_ * dout, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_ = np.matmul(np.ones((N,1)), gamma.reshape((1, -1))) * dout
        dx = N * dx_ - np.sum(dx_, axis=0) - x_ * np.sum(dx_ * x_, axis=0)
        dx *= (1.0/N) / np.sqrt(var_plus_eps)

        return dx, dgamma, dbeta

    def update(self, lr, dgamma, dbeta):
        self.gamma -= dgamma *lr
        self.beta -= dbeta*lr

    def bn_forward_naive(self,x, mode="train", eps=1e-5, momentum=0.9):
        n, ic, ih, iw = x.shape
        out = np.zeros(x.shape)
        if mode == 'train':
            batch_mean = np.zeros(self.running_mean.shape)
            batch_var = np.zeros(self.running_var.shape)
            for i in range(ic):
                batch_mean[i] = np.mean(x[:, i, :, :])
                batch_var[i] = np.sum((x[:, i, :, :] - batch_mean[i]) ** 2) / (n * ih * iw)
            for i in range(ic):
                out[:, i, :, :] = (x[:, i, :, :] - batch_mean[i]) / np.sqrt(batch_var[i] + eps)
                out[:, i, :, :] = out[:, i, :, :] * self.gamma + self.beta
            # update
            self.running_mean = self.running_mean * momentum + batch_mean * (1 - momentum)
            self.running_var = self.running_var * momentum + batch_var * (1 - momentum)
        elif mode == 'test':
            for i in range(ic):
                out[:, i, :, :] = (x[:, i, :, :] - self.running_mean[i]) / np.sqrt(self.running_var[i] + eps)
                out[:, i, :, :] = out[:, i, :, :] * self.gamma + self.beta
        else:
            raise ValueError('Invalid forward BN mode: %s' % mode)
        return out

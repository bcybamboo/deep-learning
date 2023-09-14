'''
改DAPPM，多尺度融合中换成跨层融合
'''

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


bn_mom=0.1


# 论文作者修改过了的resblock
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, out_channels, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, out_channels, kernel_size=3,
                            stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2D(out_channels, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(out_channels, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Layer):
    expansion = 2  # 通道扩大倍数

    def __init__(self, inplanes, out_channels, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, out_channels, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels, momentum=bn_mom)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=stride,
                            padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels, momentum=bn_mom)
        self.conv3 = nn.Conv2D(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        if self.no_relu:
            return out
        else:
            return self.relu(out)


# 论文中提出的特征提取结构 实验中效果比PPM更好
class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()

        # 获得不同感受野
        self.scale0 = nn.Sequential(
            nn.BatchNorm2D(inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
        )
        self.scale1 = nn.Sequential(nn.AvgPool2D(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2D(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2D(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2D(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2D(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2D(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
                                    nn.BatchNorm2D(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        # 对不同感受野特征图进行特征提取
        self.process1 = nn.Sequential(
            nn.BatchNorm2D(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2D(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2D(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2D(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )

        # concat层
        self.compression = nn.Sequential(
            nn.BatchNorm2D(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes * 5, outplanes, kernel_size=1, bias_attr=False),
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2D(inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(inplanes, outplanes, kernel_size=1, bias_attr=False),
        )

    def forward(self, x):
        # x.shape: NCHW

        w = x.shape[-1]
        h = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[h, w],
                                                   mode='bilinear'))))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[h, w],
                                                    mode='bilinear') + x_list[0]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[h, w],
                                                   mode='bilinear') + x_list[1])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[h, w],
                                                   mode='bilinear') + x_list[2])))
        # NCHW
        out = self.compression(paddle.concat(x_list, 1)) + self.shortcut(x)

        return out


class DDRNetHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(DDRNetHead, self).__init__()
        self.bn1 = nn.BatchNorm2D(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2D(inplanes, interplanes, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(interplanes, outplanes, kernel_size=1, padding=0, bias_attr=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out


class DualResNet(nn.Layer):  # 双边残存网络
    def __init__(self, num_classes,in_channels=3,layers=[2, 2, 2, 2], planes=64, spp_planes=128, head_planes=128, augment=False,
                 pretrained=None):
        super(DualResNet, self).__init__()

        hige_planes = planes * 2
        self.augment = augment
        self.pretrained = pretrained
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels=planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(planes, momentum=bn_mom),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(BasicBlock, planes, planes, layers[0])
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
        self.compression3 = nn.Sequential(
            nn.Conv2D(planes * 4, hige_planes, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(hige_planes, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2D(planes * 8, hige_planes, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(hige_planes, momentum=bn_mom),
        )

        self.down3 = nn.Sequential(
            nn.Conv2D(hige_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(planes * 4, momentum=bn_mom),
        )

        self.down4 = nn.Sequential(
            nn.Conv2D(hige_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(planes * 4, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(planes * 8, momentum=bn_mom),
        )

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, hige_planes, 2)

        self.layer4_ = self._make_layer(BasicBlock, hige_planes, hige_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, hige_planes, hige_planes, 1)

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        # DDRNetHead 分割头
        if self.augment:
            self.seghead_extra = DDRNetHead(hige_planes, head_planes, num_classes)

        self.final_layer = DDRNetHead(planes * 4, head_planes, num_classes)
        self.init_weight()

    # 残差结构的生成
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:  # basic.expansion=1   bottle.expansion=2
            downsample = nn.Sequential(  # 使用conv进行下采样
                nn.Conv2D(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))  # 1个block
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))  # 最后一个层使用relu输出
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x.shape: NCHW
        input = x
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        layers = []
        x = self.conv1(x)  # conv1
#（8，64，56，56）
        x = self.layer1(x)
        layers.append(x)  # conv2 layer[0]

        x = self.layer2(self.relu(x))
        layers.append(x)  # conv3 layer[1]
#（8，128，28，28）
        x = self.layer3(self.relu(x))  # 低分辨率层#（8，256，14，14）
        layers.append(x)  # conv4 layer[2]
        x_ = self.layer3_(self.relu(layers[1]))  # 复制一份 高分辨率层

        # Bilateral fusion 论文中的融合方法
        x = x + self.down3(self.relu(x_))  # 高分辨率 通过conv下采样 与低分辨率相加
        x_ = x_ + F.interpolate(  # 低分辨率上采样后和高分辨率相加
            self.compression3(self.relu(layers[2])),  # 1x1conv 将通道下降
            size=[height_output, width_output],
            mode='bilinear')

        if self.augment:
            temp = x_
        # Conv5_1       使用basicblock
        x = self.layer4(self.relu(x))  # 低分辨率层
        layers.append(x)  # conv5_1 layer[3]
        x_ = self.layer4_(self.relu(x_))  # 高分辨率

        # Bilateral fusion
        x = x + self.down4(self.relu(x_))  # 高分辨率 通过conv下采样 与低分辨率相加
        x_ = x_ + F.interpolate(  # 低分辨率上采样后和高分辨率相加
            self.compression4(self.relu(layers[3])),  # 1x1conv 将通道下降
            size=[height_output, width_output],
            mode='bilinear')
        # print(x_)
        # Conv5_1阶段     使用bottleblock
        x_ = self.layer5_(self.relu(x_))  # 高分通道
        x = F.interpolate(  # 低分通道
            self.spp(self.layer5(self.relu(x))),  # 使用DAPPM进行多尺度特征融合
            size=[height_output, width_output],
            mode='bilinear')
        x_ = self.final_layer(x + x_)

        if self.augment:  # 辅助计算损失
            x_extra = self.seghead_extra(temp)
            return [x_, x_extra]
        else:
            return [F.interpolate(x_,
                                  size=[input.shape[-2], input.shape[-1]],
                                  mode='bilinear')]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    param_init.kaiming_normal_init(m.weight)
                elif isinstance(m, nn.BatchNorm2D):
                    param_init.constant_init(m.weight, value=1)
                    param_init.constant_init(m.bias, value=0)

@manager.MODELS.add_component
def DDRNet_23(**kwargs):
    return DualResNet(
        layers=[2, 2, 2, 2],
        planes=64,
        spp_planes=128,
        head_planes=128,
        **kwargs)



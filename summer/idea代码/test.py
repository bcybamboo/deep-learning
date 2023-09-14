# # import paddle
# # import paddle.nn as nn
# #
# # # class ChannelShuffle(nn.Layer):
# # #     def __init__(self, groups):
# # #         super(ChannelShuffle, self).__init__()
# # #         self.groups = groups
# # #
# # #     def forward(self, x):
# # #         batch_size, num_channels, height, width = x.shape
# # #         channels_per_group = num_channels // self.groups
# # #
# # #         # Reshape input tensor
# # #         x = paddle.reshape(x, shape=[batch_size, self.groups, channels_per_group, height, width])
# # #
# # #         # Transpose and reshape back
# # #         x = paddle.transpose(x, perm=[0, 2, 1, 3, 4])
# # #         x = paddle.reshape(x, shape=[batch_size, num_channels, height, width])
# # #
# # #         return x
# #
# # def channel_shuffle(x, groups):
# #     ### type: (torch.Tensor, int) -> torch.Tensor
# #     # print(x.shape)
# #     batchsize, num_channels, height, width = x.shape
# #     channels_per_group = num_channels // groups
# #
# #     # reshape
# #     x = paddle.reshape(x, [batchsize, groups, channels_per_group, height, width])
# #
# #     x = paddle.transpose(x, [0, 2, 1, 3, 4])
# #
# #     # flatten
# #     x = paddle.reshape(x, [batchsize, num_channels, height, width])
# #
# #     return x
# #
# # # Example usage
# # x = paddle.randn(shape=[1, 32, 224, 224])  # Example input tensor
# # # shuffle = ChannelShuffle(groups=8)  # Create ChannelShuffle layer with 4 groups
# # # output = shuffle(x)  # Apply channel shuffle operation
# # output = channel_shuffle(x, groups=8)
# # print(x)
# # print("------------")
# # print(output)
#
# import paddle
# import paddle.nn as nn
#
#
# class ChannelAttention(nn.Layer):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2D(1)
#         self.max_pool = nn.AdaptiveMaxPool2D(1)
#         self.fc1 = nn.Conv2D(in_channels, in_channels // reduction_ratio, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2D(in_channels // reduction_ratio, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Layer):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2D(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = paddle.mean(x, axis=1, keepdim=True)
#         max_out = paddle.max(x, axis=1, keepdim=True)
#         out = paddle.concat([avg_out, max_out], axis=1)
#         out = self.conv(out)
#         return self.sigmoid(out)
#
#
# class CBAM(nn.Layer):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_att = ChannelAttention(in_channels, reduction_ratio)
#         self.spatial_att = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         out = x * self.channel_att(x)
#         out = out * self.spatial_att(out)
#         return out
#
#
# # 测试 CBAM 模块
# input_data = paddle.randn([1, 64, 32, 32])  # 假设输入数据为 64 通道的 32x32 图像
# cbam = CBAM(in_channels=1024)
# output = cbam(input_data)
# print(output.shape)

import paddle
import paddle.nn as nn


class EdgeDetectionNet(nn.Layer):
    def __init__(self):
        super(EdgeDetectionNet, self).__init__()

        # Stage 1
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Stage 2
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Stage 3
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Stage 4
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Stage 5
        self.dilation = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x1 = self.relu1(x)
        x1 = self.pool1(x1)

        # Stage 2
        x = self.conv2(x)
        x2 = self.relu2(x)
        x2 = self.pool2(x2)

        # Stage 3
        x = self.conv3(x)
        x3 = self.relu3(x)
        x3 = self.pool3(x3)

        # Stage 4
        x = self.conv4(x)
        x4 = self.relu4(x)
        x4 = self.pool4(x4)

        # Stage 5
        x = self.dilation(x)
        x5 = self.conv5(x)
        x5 = self.relu5(x5)

        # add


        return x


# 创建边缘检测网络实例
net = EdgeDetectionNet()

# 输入数据的维度为 [batch_size, channels, height, width]
input_data = paddle.randn((2, 3, 1024, 1024))

# 使用边缘检测网络进行推理
output = net(input_data)

# 打印输出结果的形状
print(output.shape)
"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-02-18 22:50:12
"""


import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局自适应最大池化

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(
            self.relu1(self.fc1(self.avg_pool(x)))
        )  # x(B,512,26,26)    avg_out: (B,512,26,26)->(B,512,1,1)->(B,64,1,1)->(B,512,1,1)
        max_out = self.fc2(
            self.relu1(self.fc1(self.max_pool(x)))
        )  # x(B,512,26,26)    max_out: (B,512,26,26)->(B,512,1,1)->(B,64,1,1)->(B,512,1,1)
        out = avg_out + max_out  # (B,512,1,1)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 逐点求均值  (B,1,26,26)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 逐点求最大值  (B,1,26,26)
        x = torch.cat([avg_out, max_out], dim=1)  # 叠加两个空间注意力机制层    (B,2,26,26)
        x = self.conv1(x)  # 融合两个空间注意力层   (B,1,26,26)
        return self.sigmoid(x)


class cbam_sc_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_sc_block, self).__init__()
        self.channelattention = ChannelAttention(in_planes=channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # ******************** spatial attention ********************
        x = x * self.spatialattention(
            x
        )  # x(B,512,26,26) * spatialattention(B,1,26,26) -> (B,512,26,26)

        # ******************** channel attention ********************
        x = x * self.channelattention(
            x
        )  # x(B,512,26,26) * channelattention(B,512,1,1) -> (B,512,26,26)

        return x  # (B,512,26,26)

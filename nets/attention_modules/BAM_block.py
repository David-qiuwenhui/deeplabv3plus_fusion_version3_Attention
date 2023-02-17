"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-02-16 22:52:24
"""
from torch import nn
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module("flatten", Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module(
                "fc%d" % i, nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            self.ca.add_module("bn%d" % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module("relu%d" % i, nn.ReLU())
        self.ca.add_module("last_fc", nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, kernel=None, dia_val=None):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module(
            "conv_reduce1",
            nn.Conv2d(
                kernel_size=1, in_channels=channel, out_channels=channel // reduction
            ),
        )
        self.sa.add_module("bn_reduce1", nn.BatchNorm2d(channel // reduction))
        self.sa.add_module("relu_reduce1", nn.ReLU())
        for i in range(num_layers):
            dilation_rate = dia_val[i]
            self.sa.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    kernel_size=kernel,
                    in_channels=channel // reduction,
                    out_channels=channel // reduction,
                    padding=(dilation_rate * kernel - dilation_rate) // 2,
                    dilation=dilation_rate,
                ),  # 带空洞卷积的3*3卷积层
            )
            self.sa.add_module("bn_%d" % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module("relu_%d" % i, nn.ReLU())
        self.sa.add_module(
            "last_conv", nn.Conv2d(channel // reduction, 1, kernel_size=1)
        )

    def forward(self, x):
        res = self.sa(x)  # (C,H,W) -> (1,1,1)
        return res


class BAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel=None, dia_val=None):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(
            channel=channel, reduction=reduction, kernel=kernel, dia_val=dia_val
        )
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out

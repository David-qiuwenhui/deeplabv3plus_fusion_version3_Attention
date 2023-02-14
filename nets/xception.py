import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

bn_mom = 0.0003


# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        activate_first=True,
        inplace=True,
    ):
        super(SeparableConv2d, self).__init__()
        # 首层激活层
        self.activate_first = activate_first
        self.relu0 = nn.ReLU(inplace=inplace)  # inplace=True改变输入数据
        # 逐通道卷积（特征提取）
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        # 逐点卷积（改变特征维度）
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 首层ReLu激活
        if self.activate_first:
            x = self.relu0(x)
        # 逐通道卷积
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        # 逐点卷积
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        strides=1,
        atrous=None,
        grow_first=True,
        activate_first=True,
        inplace=True,
    ):
        super(Block, self).__init__()
        # atrous_rate
        if atrous == None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous_list = [atrous] * 3
            atrous = atrous_list
        idx = 0
        self.head_relu = True

        # shortcut connect 如果feature的size和channels不改变 使用map映射连接
        # 若输入和输出通道有调整或s!=1进行下采样 使用Conv2d k1x1进行卷积处理的捷径分支
        if out_filters != in_filters or strides != 1:
            # 调整channels或者feature_size
            self.skip = nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=1,
                stride=strides,
                bias=False,
            )
            self.skipbn = nn.BatchNorm2d(num_features=out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip = None

        self.hook_layer = None

        # 若选择在第一层改变channels
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(
            in_channels=in_filters,
            out_channels=filters,
            kernel_size=3,
            stride=1,
            padding=1 * atrous[0],
            dilation=atrous[0],
            bias=False,
            activate_first=activate_first,
            inplace=self.head_relu,
        )
        self.sepconv2 = SeparableConv2d(
            in_channels=filters,
            out_channels=out_filters,
            kernel_size=3,
            stride=1,
            padding=1 * atrous[1],
            dilation=atrous[1],
            bias=False,
            activate_first=activate_first,
        )
        self.sepconv3 = SeparableConv2d(
            in_channels=out_filters,
            out_channels=out_filters,
            kernel_size=3,
            stride=strides,
            padding=1 * atrous[2],
            dilation=atrous[2],
            bias=False,
            activate_first=activate_first,
            inplace=inplace,
        )

    def forward(self, inp):
        # shortcut 捷径分支连接
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp  # map 直接映射的捷径分支

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x  # 通过hook的方式保存浅层次的特征
        x = self.sepconv3(x)

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, downsample_factor):
        """Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if downsample_factor == 8:
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError("xception.py: output stride=%d is not supported." % os)
        # downsample_factor=8, stride_list=[2, 1, 1]

        # ---------- session1 ----------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=bn_mom)
        # do relu here

        # ------ session2 ------
        self.block1 = Block(in_filters=64, out_filters=128, strides=2)
        self.block2 = Block(
            in_filters=128, out_filters=256, strides=stride_list[0], inplace=False
        )  # strides=2
        self.block3 = Block(
            in_filters=256, out_filters=728, strides=stride_list[1]
        )  # strides=1

        # ------ session3 ------
        rate = 16 // downsample_factor  # 2
        self.block4 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block5 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block6 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block7 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)

        self.block8 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block9 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block10 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block11 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)

        self.block12 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block13 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block14 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)
        self.block15 = Block(in_filters=728, out_filters=728, strides=1, atrous=rate)

        self.block16 = Block(
            in_filters=728,
            out_filters=728,
            strides=1,
            atrous=[1 * rate, 1 * rate, 1 * rate],
        )  # atrous=[2, 2, 2]
        self.block17 = Block(
            in_filters=728,
            out_filters=728,
            strides=1,
            atrous=[1 * rate, 1 * rate, 1 * rate],
        )  # atrous=[2, 2, 2]
        self.block18 = Block(
            in_filters=728,
            out_filters=728,
            strides=1,
            atrous=[1 * rate, 1 * rate, 1 * rate],
        )  # atrous=[2, 2, 2]
        self.block19 = Block(
            in_filters=728,
            out_filters=728,
            strides=1,
            atrous=[1 * rate, 1 * rate, 1 * rate],
        )  # atrous=[2, 2, 2]

        # ------ session4 ------
        self.block20 = Block(
            in_filters=728,
            out_filters=1024,
            strides=stride_list[2],
            atrous=rate,
            grow_first=False,
        )  # strides=1
        self.conv3 = SeparableConv2d(
            in_channels=1024,
            out_channels=1536,
            kernel_size=3,
            stride=1,
            padding=1 * rate,
            dilation=rate,
            activate_first=False,
        )  # p2, d2

        self.conv4 = SeparableConv2d(
            in_channels=1536,
            out_channels=1536,
            kernel_size=3,
            stride=1,
            padding=1 * rate,
            dilation=rate,
            activate_first=False,
        )  # p2, d2

        self.conv5 = SeparableConv2d(
            in_channels=1536,
            out_channels=2048,
            kernel_size=3,
            stride=1,
            padding=1 * rate,
            dilation=rate,
            activate_first=False,
        )  # p2, d2
        self.layers = []

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, input):
        self.layers = []  # input(bs,3,512,512)
        # ------ session1 ------
        x = self.conv1(input)  # x(bs,32, 256, 256)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # x(bs,64,256,256)
        x = self.bn2(x)
        x = self.relu(x)
        # ------ session2 ------
        x = self.block1(x)  # (bs,128,128,128)
        x = self.block2(x)  # (bs,256,64,64)
        low_featrue_layer = (
            self.block2.hook_layer
        )  # lower_feature_layer(bs,256,128,128)
        x = self.block3(x)  # (bs,728,64,64)
        # ------ session3 ------
        x = self.block4(x)  # (bs,728,64,64)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)  # (bs,728,64,64)
        # ------ session4 ------
        x = self.block20(x)  # (bs,1024,64,64)
        x = self.conv3(x)  # (bs,1536,64,64)
        x = self.conv4(x)  # (bs,1536,64,64)
        x = self.conv5(x)  # (bs,2048,64,64)
        return (
            low_featrue_layer,
            x,
        )  # low_feature_layer tensor(bs,256,128,128)  x(bs,2048,64,64)


def xception(pretrained=False, downsample_factor=8):
    model = Xception(downsample_factor)
    # 载入模型主干部分的预训练权重
    if pretrained:
        model.load_state_dict(torch.load("./model_data/xception_pytorch_imagenet.pth"))
    return model

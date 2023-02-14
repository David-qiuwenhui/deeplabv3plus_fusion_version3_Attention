"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-02-02 15:59:39
"""


from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:  # min_ch限制channels的下限值
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)  # '//'取整除
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        padding = (kernel_size - 1) // 2  # 取整除
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )  # inplace=True 不创建新的对象，直接对原始对象进行修改


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(ch=input_c // squeeze_factor, divisor=8)
        self.fc1 = nn.Conv2d(
            input_c, squeeze_c, kernel_size=1
        )  # fc1: expand_channel // 4 (Conv2d 1x1代替全连接层)
        self.fc2 = nn.Conv2d(
            squeeze_c, input_c, kernel_size=1
        )  # fc2: expand_channel (Conv2d 1x1代替全连接层)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 自适应全局平均池化处理
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x  # SE通道注意力机制与Conv3x3主分支结果相乘


class InvertedResidualConfig:
    def __init__(
        self,
        input_c: int,
        kernel: int,
        expanded_c: int,
        out_c: int,
        use_se: bool,
        activation: str,
        stride: int,
        width_multi: float,
    ):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(
        channels: int, width_multi: float
    ):  # 获取8的整数倍的channels（更大化利用硬件资源和加速训练）
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(
        self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]
    ):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 检测是否使用shortcu捷径分支（stride=1不进行下采样 and input_c==output_c）
        self.use_res_connect = cnf.stride == 1 and cnf.input_c == cnf.out_c

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 使用conv2d 1*1卷积模块进行升维操作
        # Expand block
        if cnf.expanded_c != cnf.input_c:
            layers.append(
                ConvBNActivation(
                    cnf.input_c,
                    cnf.expanded_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise block 逐通道卷积Depthwise Conv
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.expanded_c,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_c,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # SqueezeExcitation attention block
        if cnf.use_se:  # 使用SE通道注意力机制
            layers.append(
                SqueezeExcitation(cnf.expanded_c)
            )  # SqueezeExcitation(AdaptiveAvgPool->fc1->ReLU->fc2->hardsigmoid  input*SE_result

        # Project block 逐点卷积Pointwise Conv
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity,
            )
        )  # nn.Identity 线性激活

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (
            isinstance(inverted_residual_setting, List)
            and all(
                [
                    isinstance(s, InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = InvertedResidual

        if norm_layer is None:  # norm_layer使用 BatchNormalization
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            # eps – a value added to the denominator for numerical stability. Default: 1e-5
            # momentum - the value used for the running_mean and running_var computation. Default: 0.1
        layers: List[nn.Module] = []

        # 第一个Conv Block升高维度并进行两倍下采样
        # building first layer(conv2d k3, s2) output_c=16
        firstconv_output_c = inverted_residual_setting[0].input_c  # 16
        layers.append(
            ConvBNActivation(
                in_planes=3,
                out_planes=firstconv_output_c,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )  # layers0: Conv2d -> BN -> ReLU6

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        self.features = nn.Sequential(*layers)  # mobilenetV3的特征提取模块feature extractor

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x = self.features(x)  # x(B, 3, H, W) -> x(B, 960, H/32, W/32)
        mid_level = 6
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == mid_level:
                mid_level_features = x

        return mid_level_features, x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large_deeplabv3plus(reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    Args:
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0  # alpha-Width Multiplier（alpha超参数调整卷积核的数量）
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    # partial 预先定义function函数中的参数（已传入的参数不可进行修改，调用时可传入未提前传入的参数）
    reduce_divider = 2 if reduced_tail else 1  # 缩减最后两个bneck模块的channels参数

    # 定义倒残差模块的参数 InvertedResidualConfig(input_c, kernel, expanded_c, out_c, use_se, activation, stride)
    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            "HS",
            1,
        ),
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            "HS",
            1,
        ),
    ]

    return MobileNetV3(inverted_residual_setting)  # 倒残差模块的超参数


def mobilenet_v3_large_backbone(model_type="large"):
    if model_type == "large":
        backbone = mobilenet_v3_large_deeplabv3plus()
    return backbone

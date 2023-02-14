"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-02-06 23:02:12
"""
from collections import namedtuple
from functools import partial
from typing import Callable, List, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from nets.mobilevit_block import MobileViTBlock


# BatchNorm2d 标准化层的超参数
BN_MOMENTUM = 0.01
EPS = 0.001


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


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        if padding == None:
            padding = (kernel_size - 1) // 2  # 取整除

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
            nn.BatchNorm2d(out_planes, eps=EPS, momentum=BN_MOMENTUM),
            activation_layer(inplace=True),
        )  # inplace=True 不创建新的对象，直接对原始对象进行修改


def ConvBN(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=None,
    groups=1,
    activation_layer=None,
):
    if padding == None:
        padding = (kernel_size - 1) // 2  # 取整除

    if activation_layer is None:
        activation_layer = nn.ReLU6
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(out_planes, eps=EPS, momentum=BN_MOMENTUM))
    return result


class RepVGGplusBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
        use_post_se=False,
        activation_layer=None,
    ):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()
        # self.nonlinearity = nn.ReLU6()

        # 引入通道注意力机制
        # RepVGGPlus的SE通道注意力模块在非线性激活模块后使用
        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            # ******************** BatchNorm层分支 ********************
            # 输入和输出不发生通道数量变化和卷积步长为1时引入BN分支
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None

            # ******************** 3*3卷积分支 ********************
            self.rbr_dense = ConvBN(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups,
                activation_layer=activation_layer,
            )

            # ******************** 1*1卷积分支 ********************
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = ConvBN(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                activation_layer=activation_layer,
            )

    def forward(self, x):
        # ********** 参数重构化的部署模式 **********
        if self.deploy:
            return self.post_se(self.nonlinearity(self.rbr_reparam(x)))

        # ********** 普通多分支结构 **********
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        out = self.post_se(self.nonlinearity(out))
        return out

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将1*1卷积层填充成3*3卷积层
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            # 3*3和1*1卷积分支 包含卷结层和批量标准化层
            # 卷积核权重参数与批量标准化核的均值、方差、放大倍率gamma和偏移量beta参数
            kernel, running_mean, running_var, gamma, beta, eps = (
                branch.conv.weight,
                branch.bn.running_mean,
                branch.bn.running_var,
                branch.bn.weight,
                branch.bn.bias,
                branch.bn.eps,
            )
            # eps是一个很小的值防止BN计算的过程中分母为零
        else:
            # BN分支 批量标准化层
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                # 创建等效卷积层
                # Construct and store the identity kernel in case it is used multiple times
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            # nn.Indentity层的等效卷积核参数与批量标准化核的均值、方差、放大倍率gamma和偏移量beta参数
            kernel, running_mean, running_var, gamma, beta, eps = (
                self.id_tensor,
                branch.running_mean,
                branch.running_var,
                branch.weight,
                branch.bias,
                branch.eps,
            )
            # eps是一个很小的值防止BN计算的过程中分母为零

        # 3*3卷积分支、1*1卷积分支、批量标准化分支的等效权重和偏执参数
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        # 获得等效合并分支的卷积核权重和偏执参数
        kernel, bias = self.get_equivalent_kernel_bias()
        # 构造等效合并分支的卷积核 权重和偏执
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        # 将等效合并分支参数载入卷积核
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # 删除原来的多分支卷积核
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


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
        in_planes: int,
        expanded_planes: int,
        out_planes: int,
        kernel: int,
        stride: int,
        activation: str,
        use_se: bool,
        width_multi: float,
    ):
        self.in_planes = self.adjust_channels(in_planes, width_multi)
        self.expanded_planes = self.adjust_channels(expanded_planes, width_multi)
        self.out_planes = self.adjust_channels(out_planes, width_multi)
        self.kernel = kernel
        self.stride = stride
        self.use_se = use_se
        self.use_hs = activation == "HS"

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        # 获取8的整数倍的channels（更大化利用硬件资源和加速训练）
        return _make_divisible(channels * width_multi, 8)


class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        expanded_planes,
        out_planes,
        kernel=3,
        stride=1,
        use_hs=False,
        use_se=False,
        downsample=None,
        deploy=False,
        repvgg_use_se=False,
    ):
        super().__init__()

        # ******************** 非线性激活层 ********************
        self.activation_layer = nn.Hardswish if use_hs else nn.ReLU6
        self.activation = self.activation_layer(inplace=True)

        # ******************** shortcut连接的下采样层 ********************
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

        # ******************** 主分支通路 ********************
        layers: List[nn.Module] = []
        layers.append(
            ConvBNActivation(
                in_planes,
                expanded_planes,
                kernel_size=1,
                stride=1,
                activation_layer=self.activation_layer,
            )
        )
        layers.append(
            RepVGGplusBlock(
                in_channels=expanded_planes,
                out_channels=expanded_planes,
                kernel_size=kernel,
                stride=stride,
                groups=expanded_planes,
                deploy=deploy,
                use_post_se=repvgg_use_se,
                activation_layer=self.activation_layer,
            )
        )  # 使用了逐通道卷积

        # 引入通道注意力机制
        if use_se:
            layers.append(SqueezeExcitation(input_c=expanded_planes, squeeze_factor=4))
        layers.append(
            ConvBNActivation(
                expanded_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.outchannels = out_planes
        self.is_stride = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        residual = self.downsample(residual)
        out += residual
        out = self.activation(out)

        return out


class InvertedBotteneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        expanded_planes,
        out_planes,
        kernel=3,
        stride=1,
        use_hs=False,
        use_se=False,
        downsample=None,
        deploy=False,
        repvgg_use_se=False,
    ):
        super().__init__()

        # ******************** 非线性激活层 ********************
        self.activation_layer = nn.Hardswish if use_hs else nn.ReLU6
        self.activation = self.activation_layer(inplace=True)

        # ******************** 跳跃连接的下采样层 ********************
        if downsample is None:
            if in_planes != out_planes or stride != 1:
                self.downsample = ConvBNActivation(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    activation_layer=nn.Identity,
                )
            else:
                self.downsample = nn.Identity()
        else:
            self.downsample = downsample

        # ******************** 主分支通路 ********************
        layers: List[nn.Module] = []
        layers.append(
            ConvBNActivation(
                in_planes,
                expanded_planes,
                kernel_size=1,
                stride=1,
                activation_layer=self.activation_layer,
            )
        )
        layers.append(
            RepVGGplusBlock(
                in_channels=expanded_planes,
                out_channels=expanded_planes,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                dilation=1,
                groups=expanded_planes,
                deploy=deploy,
                use_post_se=repvgg_use_se,
                activation_layer=self.activation_layer,
            ),
        )
        # 引入通道注意力机制
        if use_se:
            layers.append(SqueezeExcitation(input_c=expanded_planes, squeeze_factor=4))
        layers.append(
            ConvBNActivation(
                expanded_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.outchannels = out_planes
        self.is_stride = stride > 1
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        residual = self.downsample(residual)
        out += residual
        out = self.activation(out)

        return out


class StageModule(nn.Module):
    def __init__(
        self,
        input_branches,
        output_branches,
        c,
        expanded_rate,
        baseblock_use_hs,
        baseblock_use_se,
        deploy,
        repvgg_use_se,
    ):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2**i)  # 对应第i个分支的通道数
            # TODO: 修改了stage1到stage4的patch_w和patch_h
            patch_size_list = [16, 8, 4, 2]
            patch_size = patch_size_list[i]
            branch = nn.Sequential(
                BasicBlockNew(
                    in_planes=w,
                    expanded_planes=w * expanded_rate,
                    out_planes=w,
                    kernel=3,
                    stride=1,
                    use_hs=baseblock_use_hs,
                    use_se=baseblock_use_se,
                    deploy=deploy,
                    repvgg_use_se=repvgg_use_se,
                ),
                MobileViTBlock(
                    in_channels=w,
                    transformer_dim=w,
                    ffn_dim=w * 2,
                    n_transformer_blocks=1,
                    head_dim=w // 4,
                    attn_dropout=0.0,
                    dropout=0.1,
                    ffn_dropout=0.0,
                    patch_h=patch_size,
                    patch_w=patch_size,
                    conv_ksize=3,
                ),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=c * (2**j),
                                out_channels=c * (2**i),
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_features=c * (2**i), eps=EPS, momentum=BN_MOMENTUM
                            ),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode="bilinear"),
                        )
                    )
                else:
                    # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=c * (2**j),
                                    out_channels=c * (2**j),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(
                                    num_features=c * (2**j),
                                    eps=EPS,
                                    momentum=BN_MOMENTUM,
                                ),
                                nn.ReLU(inplace=True),
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=c * (2**j),
                                out_channels=c * (2**i),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_features=c * (2**i), eps=EPS, momentum=BN_MOMENTUM
                            ),
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum(
                        [
                            self.fuse_layers[i][j](x[j])
                            for j in range(len(self.branches))
                        ]
                    )
                )
            )

        return x_fused


class DeepLabV3PlusFusion(nn.Module):
    def __init__(
        self,
        base_channel,
        inverted_residual_setting: List,
        expanded_rate,
        baseblock_use_hs=False,
        baseblock_use_se=False,
        deploy=False,
        repvgg_use_se=False,
    ):
        super().__init__()

        self.deploy = deploy
        DeepLabStageModule = partial(
            StageModule,
            c=base_channel,
            expanded_rate=expanded_rate,
            baseblock_use_hs=baseblock_use_hs,
            baseblock_use_se=baseblock_use_se,
            deploy=deploy,
            repvgg_use_se=repvgg_use_se,
        )
        # ******************** Conv1 ********************
        self.conv1 = ConvBNActivation(
            in_planes=3, out_planes=16, kernel_size=3, stride=2, groups=1
        )
        self.conv2 = ConvBNActivation(
            in_planes=16, out_planes=32, kernel_size=3, stride=2, groups=1
        )

        # ******************** Stage1 ********************
        stage1: List[nn.Module] = []
        stage1_setting = inverted_residual_setting["stage1"]
        for cnf in stage1_setting:
            stage1.append(
                InvertedBotteneck(
                    cnf.in_planes,
                    cnf.expanded_planes,
                    cnf.out_planes,
                    cnf.kernel,
                    cnf.stride,
                    cnf.use_hs,
                    cnf.use_se,
                    deploy=deploy,
                    repvgg_use_se=repvgg_use_se,
                )
            )
        stage1.append(
            # TODO: 修改了stage1的patch_w和patch_h
            MobileViTBlock(
                in_channels=32,
                transformer_dim=32,
                ffn_dim=64,
                n_transformer_blocks=1,
                head_dim=8,
                attn_dropout=0.0,
                dropout=0.1,
                ffn_dropout=0.0,
                patch_h=16,
                patch_w=16,
                conv_ksize=3,
            ),
        )
        self.stage1 = nn.Sequential(*stage1)

        # ******************** Transition1 ********************
        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=base_channel,
                        out_channels=base_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channel, eps=EPS, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=base_channel,
                        out_channels=base_channel * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channel * 2, eps=EPS, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ******************** Stage2 ********************
        self.stage2 = nn.Sequential(
            DeepLabStageModule(input_branches=2, output_branches=2),
        )

        # ******************** Transition2 ********************
        self.transition2 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=base_channel * 2,
                        out_channels=base_channel * 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        num_features=base_channel * 4, eps=EPS, momentum=BN_MOMENTUM
                    ),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ******************** Stage3 ********************
        self.stage3 = nn.Sequential(
            DeepLabStageModule(
                input_branches=3,
                output_branches=3,
            ),
        )

        # ******************** transition3 ********************
        self.transition3 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=base_channel * 4,
                        out_channels=base_channel * 8,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        num_features=base_channel * 8, eps=EPS, momentum=BN_MOMENTUM
                    ),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ******************** Stage4 ********************
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            DeepLabStageModule(
                input_branches=4,
                output_branches=1,
            ),
        )

    def forward(self, x):
        # ******************** Conv1 ********************
        x = self.conv1(x)
        conv1_features = x  # Conv1层的特征图
        x = self.conv2(x)
        conv2_features = x

        # ******************** Stage1 ********************
        x = self.stage1(x)
        stage1_features = x  # Stage1层的特征图

        # ******************** Transition1 ********************
        x = [
            trans(x) for trans in self.transition1
        ]  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8)]

        # ******************** Stage2 ********************
        x = self.stage2(x)  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8)]
        stage2_features = x[0]  # Stage2层的特征图

        # ******************** Transition2 ********************
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]),
        ]
        # 新的分支由此stage尺度最小的特征下采样和升高维度得到
        # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16)]

        # ******************** Stage3 ********************
        x = self.stage3(x)  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16)]
        stage3_features = x[0]  # Stage3层的特征图

        # ******************** Transition3 ********************
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]
        # 新的分支由此stage尺度最小的特征下采样和升高维度得到
        # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16), x3(B,256,H/32,W/32)]

        # ******************** Stage4 ********************
        x = self.stage4(x)  # x[x0(B,32,H/4,W/4)] 所有分支上采样至(H/4,W/4)后逐像素点相加输出
        stage4_features = x[0]  # Stage4层的特征图

        Outputs = namedtuple(
            "outputs",
            ["main", "conv1", "conv2", "stage1", "stage2", "stage3", "stage4"],
        )

        return Outputs(
            main=x[0],
            conv1=conv1_features,
            conv2=conv2_features,
            stage1=stage1_features,
            stage2=stage2_features,
            stage3=stage3_features,
            stage4=stage4_features,
        )

    # def switch_fusion_backbone_to_deploy(self):
    #     for m in self.modules():
    #         if hasattr(m, "switch_to_deploy"):
    #             m.switch_to_deploy()
    #     self.deploy = True


def deeplabv3plus_fusion_backbone():
    # hrnet_w18, hrnet_w32, hrnet_w48
    model_cfg = dict(
        base_channel=32,
        width_multi=1.0,
        expanded_rate=3,
        baseblock_use_hs=False,
        baseblock_use_se=False,
        deploy=False,
        repvgg_use_se=False,
    )

    bneck_conf = partial(InvertedResidualConfig, width_multi=model_cfg["width_multi"])
    # 定义Stage1模块倒残差模块的参数 InvertedResidualConfig
    # in_planes, expanded_planes, out_planes, kernel, stride, activation, use_se, width_multi
    stage1_setting = [
        bneck_conf(32, 96, 32, 3, 1, "RE", False),
        # bneck_conf(32, 128, 32, 3, 1, "RE", False),
    ]
    inverted_residual_setting = dict(stage1=stage1_setting)

    return DeepLabV3PlusFusion(
        base_channel=model_cfg["base_channel"],
        inverted_residual_setting=inverted_residual_setting,
        expanded_rate=model_cfg["expanded_rate"],
        baseblock_use_hs=model_cfg["baseblock_use_hs"],
        baseblock_use_se=model_cfg["baseblock_use_se"],
        deploy=model_cfg["deploy"],
        repvgg_use_se=model_cfg["repvgg_use_se"],
    )


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    import copy

    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


# model = deeplabv3plus_fusion_backbone()
# print(model)
# deploy_model = repvgg_model_convert(model)
# print(deploy_model)

"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-01-30 12:25:20
"""
from typing import Dict, OrderedDict
import torch
import torch.nn as nn


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):  # 进行关系测试，检测return_layers是否是model模块的子集
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if (
                name in self.return_layers
            ):  # return_layer = {dict: 2} {'layer1':'low_features', 'layer4':'out'}
                out_name = self.return_layers[name]
                out[out_name] = x
        return out  # out = {OrderedDict:2} {'low_features':Tensor(B,256,H/4,W/4), 'out':Tensor{B,2048,H/8,W/8}}


# ResNet18/34 网络的residual模块
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中的主分支卷积核个数不发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 虚线的残差结构--下采样操作
        if self.downsample is not None:
            identity = self.downsample(x)  # short cut分支

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ResNet50/101/152 网络的residual模块 -- 1*1 3*3 1*1卷积用来降低维度、卷积处理、升高维度
class Bottleneck(nn.Module):
    expansion = 4  # 主分支中的卷积核个数发生变化，第三层卷积核个数增大至第一、二层的四倍

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,
        groups=1,
        width_per_group=64,
    ):
        super(Bottleneck, self).__init__()
        width = (
            int(out_channel * (width_per_group / 64.0)) * groups
        )  # width_per_group, groups, width用于ResNetXt的组卷积操作

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width)
        # ----------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            groups=groups,  # 分组卷积运算
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(width)
        # ----------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 带虚线的残差结构--下采样操作
        if self.downsample is not None:
            identity = self.downsample(x)  # short cut分支

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block类型为BasicBlock / Bottleneck
    # block_num为残差结构中 conv2_x ~ conv5_x 残差块的个数，是一个列表
    def __init__(
        self,
        block,
        blocks_num,
        num_classes=1000,
        include_top=True,
        groups=1,
        width_per_group=64,
        stride_groups=None,
    ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channel,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, channel=64, block_num=blocks_num[0], stride=stride_groups[0]
        )
        self.layer2 = self._make_layer(
            block, channel=128, block_num=blocks_num[1], stride=stride_groups[1]
        )
        self.layer3 = self._make_layer(
            block, channel=256, block_num=blocks_num[2], stride=stride_groups[2]
        )
        self.layer4 = self._make_layer(
            block, channel=512, block_num=blocks_num[3], stride=stride_groups[3]
        )

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.fc = nn.Linear(
                in_features=512 * block.expansion, out_features=num_classes
            )

        # 权重参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channel, block_num, stride=1):  # channel为第一层卷积核的个数
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=channel * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        )

        self.in_channel = channel * block.expansion  # 更改in_channel，方便多层residual模块堆叠

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )

        return nn.Sequential(
            *layers
        )  # *list将列表unpacke解开成多个独立的参数，传入函数(即将layers列表转化为非关键字参数传入)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# ResNet网络
def resnet18(num_classes=1000, include_top=True):
    return ResNet(
        block=BasicBlock,
        blocks_num=[2, 2, 2, 2],
        num_classes=num_classes,
        include_top=include_top,
    )


def resnet34(num_classes=1000, include_top=True):
    return ResNet(
        block=BasicBlock,
        blocks_num=[3, 4, 6, 3],
        num_classes=num_classes,
        include_top=include_top,
    )


def resnet50(num_classes=1000, include_top=True):
    return ResNet(
        blocks=Bottleneck,
        blocks_num=[3, 4, 6, 3],
        num_classes=num_classes,
        include_top=include_top,
    )


def resnet101(num_classes=1000, include_top=True):
    return ResNet(
        block=Bottleneck,
        blocks_num=[3, 4, 23, 3],
        num_classes=num_classes,
        include_top=include_top,
    )


def resnet152(num_classes=1000, include_top=True):
    return ResNet(
        block=Bottleneck,
        blocks_num=[3, 8, 36, 3],
        num_classes=num_classes,
        include_top=include_top,
    )


# ResNetXt网络
def resnext50_32x4d(num_classes=1000, include_top=True, stride_groups=None):
    groups = 32
    width_per_group = 4
    return ResNet(
        block=Bottleneck,
        blocks_num=[3, 4, 6, 3],
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=width_per_group,
        stride_groups=stride_groups
    )


def resnext50_32x4d_backbone(pretrained=False, downsample_factor=8, backbone_path=""):
    if downsample_factor == 8:
        stride_groups = [1, 2, 1, 1]
    elif downsample_factor == 16:
        stride_groups = [1, 2, 2, 1]
    elif downsample_factor == 32:
        stride_groups = [1, 2, 2, 2]

    backbone = resnext50_32x4d(stride_groups=stride_groups)

    if pretrained:
        backbone.load_state_dict(torch.load(backbone_path))
    
    return_layers = {
        "layer1": "low_features", 
        "layer4": "main",
    }
    
    backbone = IntermediateLayerGetter(backbone, return_layers)
    return backbone
    
    
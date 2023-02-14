"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-01-31 11:15:36
"""
import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    # 3*3 Convolution
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 3*3 Conv
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # 3*3 Conv
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(num_features=planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # residual模块 shortcut分支是否需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1*1 Conv
        self.conv1 = nn.Conv2d(
            in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False
        )  # channels: inplanes -> planes
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=BN_MOMENTUM)

        # 3*3 Conv
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,  # channels: planes -> planes
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes, momentum=BN_MOMENTUM)

        # 1*1 Conv
        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes * self.expansion,  # channels: planes -> planes * 4
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=planes * self.expansion, momentum=BN_MOMENTUM
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # residual模块 shortcut分支是否需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.num_inchannels[branch_index],
                    out_channels=num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM,
                ),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.num_inchannels[branch_index],
                planes=num_channels[branch_index],
                stride=stride,
                downsample=downsample,
            )
        )  # Basic Block
        self.num_inchannels[branch_index] = (
            num_channels[branch_index] * block.expansion
        )  # 更新inchannels的大小（在basic block的expansion不是1的情况下会进行更新）
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    inplanes=self.num_inchannels[branch_index],
                    planes=num_channels[branch_index],
                )
            )  # 重复堆叠 Basic Block

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for branch_index in range(num_branches):
            branches.append(
                self._make_one_branch(branch_index, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                # 当前特征图的维度大于目标维度 进行降低维度操作
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=self.num_inchannels[j],
                                out_channels=self.num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                self.num_inchannels[i], momentum=BN_MOMENTUM
                            ),
                        )
                    )
                # 当前特征图的维度等于目标维度
                elif j == i:
                    fuse_layer.append(None)
                # 当前特征图的维度小于目标维度 进行升高维度和渐进式下采样操作
                else:
                    conv3x3s = []
                    for k in range(
                        i - j
                    ):  # 下采样至目标特征图大小是渐进式进行的，一次只使用3*3卷积下采样至原来尺寸的1/2，最后一次再进行调整维度并且不使用ReLU激活
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=self.num_inchannels[j],
                                        out_channels=self.num_inchannels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_features=self.num_inchannels[i],
                                        momentum=BN_MOMENTUM,
                                    ),
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=self.num_inchannels[j],
                                        out_channels=self.num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_features=self.num_inchannels[j],
                                        momentum=BN_MOMENTUM,
                                    ),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):  # x0(B, 32, H/4, W/4), x1(B, 64, H/8, W/8)
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])  # x0(B, 32, H/4, W/4), x1(B, 64, H/8, W/8)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            # y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(0, self.num_branches):
                if j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                elif i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNet_Classification(nn.Module):
    def __init__(self, num_classes, backbone):
        super(HighResolutionNet_Classification, self).__init__()
        num_filters = {
            "hrnetv2_w18": [18, 36, 72, 144],
            "hrnetv2_w32": [32, 64, 128, 256],
            "hrnetv2_w48": [48, 96, 192, 384],
        }[backbone]

        # ******************** Stage1 ********************
        # 3*3 Conv
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=BN_MOMENTUM)
        # 3*3 Conv
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # ******************** Stage2 (transition1 + (basic_block*4)*1) ********************
        self.layer1 = self._make_layer(
            block=Bottleneck, inplanes=64, planes=64, num_blocks=4
        )  # bottleneck * 4

        pre_stage_channels = [Bottleneck.expansion * 64]  # pre_stage_channels = 4 * 64
        num_channels = [num_filters[0], num_filters[1]]  # num_channels = [32, 64]
        self.transition1 = self._make_transition_layer(
            num_inchannels=pre_stage_channels, num_channels=num_channels
        )  # _make_transition_layer([256], [32, 64])
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=1,
            num_branches=2,
            block=BasicBlock,
            num_blocks=[4, 4],
            num_inchannels=num_channels,  # [32, 64]
            num_channels=num_channels,  # [32, 64]
        )
        # ******************** Stage3 (transition2 + (basicblock*4)*4) ********************
        num_channels = [
            num_filters[0],
            num_filters[1],
            num_filters[2],
        ]  # [32, 64, 128]
        self.transition2 = self._make_transition_layer(
            num_inchannels=pre_stage_channels, num_channels=num_channels
        )  # pre_stage_channels = [32, 64]
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=4,
            num_branches=3,
            block=BasicBlock,
            num_blocks=[4, 4, 4],
            num_inchannels=num_channels,
            num_channels=num_channels,
        )
        # ******************** Stage4 (transition3 + (basic_block*4)*3) ********************
        num_channels = [
            num_filters[0],
            num_filters[1],
            num_filters[2],
            num_filters[3],
        ]  # num_channels = [32, 64, 128, 256]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels
        )  # pre_stage_channels = [32, 64, 128]
        self.stage4, pre_stage_channels = self._make_stage(
            num_modules=3,
            num_branches=4,
            block=BasicBlock,
            num_blocks=[4, 4, 4, 4],
            num_inchannels=num_channels,
            num_channels=num_channels,
        )

        # ******************** Neck and Head ********************
        self.pre_stage_channels = (
            pre_stage_channels  # pre_stage_channels = [32, 64, 128, 256]
        )

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            block=Bottleneck, pre_stage_channels=pre_stage_channels
        )

        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        # 捷径分支
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # bottleneck.expansion=4, basicblock.expansion=1
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=planes * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample)
        )  # block: bottleneck or basicblock
        inplanes = planes * block.expansion  # 更新当前的输入通道数量
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_inchannels, num_channels):
        num_branches_pre = len(num_inchannels)
        num_branches_cur = len(num_channels)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels[i] != num_inchannels[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=num_inchannels[i],
                                out_channels=num_channels[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_features=num_channels[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # Transition 下采样分支 使用3*3对当前尺度最小的分支进行下采样产生新的一个更小尺度分支
                conv3x3s = [
                    nn.Sequential(  # Conv3x3 s2 p1增加一个分支
                        nn.Conv2d(
                            in_channels=num_inchannels[-1],
                            out_channels=num_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                ]
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        num_modules,
        num_branches,
        block,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
    ):
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    multi_scale_output,
                )  # 不同尺寸的feature maps 上下采样融合模块
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, block, pre_stage_channels):
        head_channels = [32, 64, 128, 256]

        incre_modules = []
        for i, channels in enumerate(
            pre_stage_channels
        ):  # pre_stage_channels=[32, 64, 128, 256]
            incre_module = self._make_layer(
                block=block,
                inplanes=channels,  # block=BottleNeck
                planes=head_channels[i],
                num_blocks=1,
                stride=1,
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * block.expansion
            out_channels = head_channels[i + 1] * block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        # ******************** Stage1 ********************
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # ******************** Stage2 (transition1 + (basic_block*4)*1) ********************
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # ******************** Stage3 (transition2 + (basicblock*4)*4) ********************
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # ******************** Stage4 (transition3 + (basic_block*4)*3) ********************
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # ------ incre_modules + downsamp_modules + final_layer (delete)------
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y


def hrnet_classification(pretrained=False, backbone="hrnetv2_w18"):
    model = HighResolutionNet_Classification(num_classes=1000, backbone=backbone)
    if pretrained:
        model_urls = {
            "hrnetv2_w18": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w18_imagenet_pretrained.pth",
            "hrnetv2_w32": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_imagenet_pretrained.pth",
            "hrnetv2_w48": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w48_imagenet_pretrained.pth",
        }
        state_dict = load_state_dict_from_url(
            url=model_urls[backbone], model_dir="./model_data"
        )
        model.load_state_dict(state_dict)

    return model


class hrnet_backbone_classification(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def forward(self, x):
        # ******************** Stage1 ********************
        x = self.model.conv1(x)  # x(bs, 3, H, W) -> x(bs, 64, H/2, H/2)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)  # x(bs, 64, H/4, W/4)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        low_level_features = self.model.layer1(
            x
        )  # low_level_features(bs, 256, H/4, W/4)
        x = low_level_features  # x(bs, 256, H/4, W/4)

        # ******************** Stage2 (transition1 + (basic_block*4)*1) ********************
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)  # x_list
        """
        x_list len=2
        (B, 32, H/4, W/4)
        (B, 64, H/8, W/8)
        
        y_list len=2
        (B, 32, H/4, W/4)
        (B, 64, H/8, W/8)
        """

        # ******************** Stage3 (transition2 + (basicblock*4)*4) ********************
        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)
        """
        x_list len=3
        (B, 32, H/4, W/4)
        (B, 64, H/8, W/8)
        (B, 64, H/16, W/16)

        y_list len=3
        (B, 32, H/4, W/4)
        (B, 64, H/8, W/8)
        (B, 64, H/16, W/16)
        """

        # ******************** Stage4 (transition3 + (basic_block*4)*3) ********************
        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        """
        x_list len=4
        (bs, 32, H/4, W/4)
        (bs, 64, H/8, W/8)
        (bs, 128, H/16, W/16)  
        (bs, 256, H/32, W/32)
    
        y_list len=4
        (bs, 32, H/4, W/4)
        (bs, 64, H/8, W/8)
        (bs, 128, H/16, W/16)  
        (bs, 256, H/32, W/32)
        """
        return y_list, low_level_features


class HRNet_Backbone(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        self.backbone = hrnet_backbone_classification(
            backbone=backbone, pretrained=pretrained
        )

        last_inp_channels = np.sum(
            self.backbone.model.pre_stage_channels, dtype=int
        )  # pre_stage_channels = [32, 64, 128, 256]  channels_sum=480

        # new last layer
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        # ******************** Backbone ********************
        H, W = inputs.size(2), inputs.size(3)  # inputs(bs, 3, H, W)
        x, low_level_features = self.backbone(inputs)
        # x0(B, 32, H/4, W/4), x1(B, 64, H/8, W/8), x2(B, 128, H/16, W/16), x3(B, 256, H/32, W/32)
        # low_level_features(B,256,H/4,W/4)

        # ******************** Upsampling ********************
        x0_h, x0_w = x[0].size(2), x[0].size(3)  # H/4, W/4
        x1 = F.interpolate(
            input=x[1], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )  # x1(B, 64, H/4, W/4)
        x2 = F.interpolate(
            input=x[2], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )  # x2(B, 128, H/4, W/4)
        x3 = F.interpolate(
            input=x[3], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )  # x3(B, 256, H/4, W/4)

        # ******************** Concat feature maps ********************
        x = torch.cat(tensors=[x[0], x1, x2, x3], dim=1)  # x(B, 480, H/4, W/4)

        # ******************** Fusion Convolution (main branch) ********************
        x = self.last_layer(x)  # x(B, last_inp_channels, H/4, W/4)
        return low_level_features, x

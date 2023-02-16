import torch
from torch import nn
from collections import OrderedDict


class SKAttention(nn.Module):
    def __init__(self, channel, kernels=[1, 3, 5, 7], reduction=4, group=1, L=16):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=group,
                                ),
                            ),
                            ("bn", nn.BatchNorm2d(channel)),
                            ("relu", nn.ReLU()),
                        ]
                    )
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)  # 选择哪一个kernel size

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, dim=0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel

        attention_weights = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weights * feats).sum(0)
        return V

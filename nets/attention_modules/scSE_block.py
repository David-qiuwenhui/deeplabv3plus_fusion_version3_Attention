import torch
import torch.nn as nn


class cSE(nn.Module):
    def __init__(self, in_channels, squeeze_rate=None):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(
            in_channels, in_channels // squeeze_rate, kernel_size=1, bias=False
        )
        self.Conv_Excitation = nn.Conv2d(
            in_channels // squeeze_rate, in_channels, kernel_size=1, bias=False
        )
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # (B,C,H,W) -> (B,C,1,1)
        z = self.Conv_Squeeze(z)  # (B,C,1,1) -> (B,C/2,1,1)
        z = self.Conv_Excitation(z)  # (B,C/2,1,1) -> (B,C,1,1)
        z = self.norm(z)
        return U * z  # (B,C,H,W)


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1, bias=False
        )
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # (B,C,H,W) -> (B,1,H,W)
        q = self.norm(q)
        return U * q  # (B,C,H,W)


class scSE(nn.Module):
    def __init__(self, in_channels, squeeze_rate):
        super().__init__()
        self.cSE = cSE(in_channels, squeeze_rate=squeeze_rate)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)  # (B,C,H,W)
        U_cse = self.cSE(U)  # (B,C,H,W)
        return U_cse + U_sse

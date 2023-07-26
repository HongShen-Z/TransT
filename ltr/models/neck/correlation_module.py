import torch
from torch import nn
from torch.nn import functional as F


def xcorr_depthwise(x, kernel, channel):
    batch = kernel.size(0)
    # channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, padding=3, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


# for deployment acceleration
def xcorr_depthwise_deploy(x, kernel, channel):
    # kernel: [channel, 1, ksize, ksize]
    # x: [1, channel, h, w]
    out = F.conv2d(x, kernel, padding=3, groups=channel)
    return out


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Correlation(nn.Module):
    def __init__(self, chn=128):
        super(Correlation, self).__init__()
        self.chn = chn
        self.sep_conv1 = SepConv(chn*2, chn, 3, 1, 1)
        self.sep_conv2 = SepConv(chn*2, chn, 3, 1, 1)
        self.sep_conv3 = SepConv(chn*2, chn, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, kernel):
        # (128, 16, 16)
        if torch.onnx.is_in_onnx_export():
            xcorr = xcorr_depthwise_deploy
        else:
            xcorr = xcorr_depthwise
        new_features = xcorr(x, kernel, self.chn)
        x = torch.cat((x, new_features), 1)
        x = self.sep_conv1(x)

        new_features = xcorr(x, kernel, self.chn)
        x = torch.cat((x, new_features), 1)
        x = self.sep_conv2(x)

        new_features = xcorr(x, kernel, self.chn)
        x = torch.cat((x, new_features), 1)
        x = self.sep_conv3(x)

        return x


def build_correlation(chn):
    return Correlation(chn)

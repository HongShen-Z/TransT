"""
Creates a MultiMax Model as defined in https://arxiv.org/pdf/2008.08178.pdf

"Discovering Multi-Hardware Mobile Models via Architecture Search"
Grace Chu, Okan Arikan, Gabriel Bender, Weijun Wang,
Achille Brighton, Pieter-Jan Kindermans, Hanxiao Liu,
Berkin Akin, Suyog Gupta, and Andrew Howard
"""
import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Identity(),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MultiMax(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(MultiMax, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 8)
        # 256,256,3 -> 128,128,32
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # # building last several layers
        # self.conv = conv_1x1_bn(input_channel, exp_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output_channel = 1280
        # output_channel = _make_divisible(output_channel * width_mult, 8) if width_mult > 1.0 else output_channel
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel, bias=False),
        #     nn.BatchNorm2d(output_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def multimax(pretrained=False, **kwargs):
    """
    Constructs a MultiMax model
    """
    cfgs = [
        # k, t, c, s
        # 128,128,32 -> 64,64,32    1/4
        [3,   3,  32, 2],
        # 64,64,32 -> 32,32,64      1/8
        [5,   6,  64, 2],
        [3,   2,  64, 1],
        [3,   2,  64, 1],
        # 32,32,64 -> 16,16,128     1/16
        [5,   6,  128, 2],
        [3,   4,  128, 1],
        [3,   3,  128, 1],
        [3,   3,  128, 1],
        [3,   6,  128, 1],
        [3,   3,  128, 1]
        # # 16,16,128 -> 8,8,160      1/32
        # [3,   6, 160, 2],
        # [5,   4, 160, 1],
        # [3,   5, 160, 1],
        # [5,   4, 160, 1]
    ]
    model = MultiMax(cfgs, **kwargs)
    if pretrained:
        state_dict = torch.load('/mnt/e/Postgra/learningcode/cache/torch/checkpoints/mobilenetv3-large-1cd25616.pth')
        model.load_state_dict(state_dict, strict=False)
    return model

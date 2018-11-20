import sys, os
sys.path.insert(0, '/home/marcelo/PyTorch/MobileNet-V2-Pytorch/')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

# Others
import math

# Internal
from DSConv import DSConv2d
from DSConvEngine import DSConvEngine
import MobileNetV2

# Pytorch
import torch
import torch.nn as nn
import torchvision

###V2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def block_mobilenet(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000):
    block_model = BlockMobileNetV2(block_size, n_class=num_classes)
    if pretrained:
        model = MobileNetV2.MobileNetV2()
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load('/home/marcelo/PyTorch/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'))
        block_model = torch.nn.DataParallel(block_model).cuda()
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)

    return block_model

def conv_bn(inp, oup, stride, block_size):
    return nn.Sequential(
        DSConv2d(inp, oup, 3, block_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, block_size):
    return nn.Sequential(
        DSConv2d(inp, oup, 1, block_size, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, block_size):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            DSConv2d(inp, inp * expand_ratio, 1, block_size, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            DSConv2d(inp * expand_ratio, inp * expand_ratio, 3, block_size, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            DSConv2d(inp * expand_ratio, oup, 1, block_size, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BlockMobileNetV2(nn.Module):
    def __init__(self, block_size, n_class=1000, input_size=224, width_mult=1.):
        super(BlockMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2, block_size)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel,  s, t, block_size))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, block_size))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, block_size))
        self.features.append(nn.AvgPool2d(int(input_size/32)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

if __name__=="__main__":
    model = block_mobilenet(pretrained=True, block_size=128, bit_nmb=3)
    k = 0
    for mod in model.modules():
        if isinstance(mod, nn.modules.conv.Conv2d):
            print("There is a Conv2d here")
        if isinstance(mod, DSConv2d):
            print(mod.weight.data.numpy().shape)
            print(mod.alpha.data.numpy().shape)
            print()
            input('')
            for param in mod.parameters():
                k+=1
    print("There are", k, "DSConvs parameters")

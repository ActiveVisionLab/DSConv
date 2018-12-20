import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

# Others
from ..DSConv2d import DSConv2d
from ..DSConvEngine import DSConvEngine
from ..Activation import transform_activation

# PyTorch
import torch
import torch.nn as nn
import torchvision

def conv3x3(in_planes, out_planes, block_size = 32, stride=1):
    """3x3 convolution with padding"""
    return DSConv2d(in_planes, out_planes, kernel_size=3, block_size=block_size, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, block_size=32, stride=1, downsample=None, m=3, e=3):
        super(BasicBlock, self).__init__()

        self.e = e
        self.m = m
        self.block_size = block_size

        self.conv1 = conv3x3(inplanes, planes, stride=stride, block_size=block_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, block_size=block_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = transform_activation(out, self.e, self.m, self.block_size)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = transform_activation(out, self.e, self.m, self.block_size)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_size=32, stride=1, downsample=None, m=3, e=3):
        super(Bottleneck, self).__init__()

        self.m = m
        self.e = e
        self.block_size = block_size

        self.conv1 = DSConv2d(inplanes, planes, kernel_size=1, block_size=block_size, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2=DSConv2d(planes,planes,kernel_size=3,block_size=block_size,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DSConv2d(planes, planes*self.expansion, kernel_size=1, block_size=block_size, bias=False)
        self.bn3 =  nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = transform_activation(out, self.e, self.m, self.block_size)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = transform_activation(out, self.e, self.m, self.block_size)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out+=residual
        out = self.relu(out)
        out = transform_activation(out, self.e, self.m, self.block_size)

        return out

class CompleteBlockResNet(nn.Module):
    def __init__(self, block, layers, block_size=32, num_classes = 1000, e=5, m=6):
        self.inplanes = 64
        super(CompleteBlockResNet, self).__init__()

        self.m = m
        self.e = e
        self.block_size = block_size

        self.conv1 = DSConv2d(3, 64, kernel_size=7, block_size=block_size, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],block_size=block_size)
        self.layer2 = self._make_layer(block, 128, layers[1],block_size=block_size,stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],block_size=block_size,stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],block_size=block_size,stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for mod in self.modules():
            if isinstance(mod, DSConv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(mod.alpha, 1)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def _make_layer(self, block, planes, blocks, block_size=32, stride=1):
        downsample = None
        if stride!=1 or self.inplanes !=planes*block.expansion:
            downsample = nn.Sequential(
                DSConv2d(self.inplanes, planes*block.expansion, kernel_size=1, block_size=block_size,stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, block_size, stride, downsample, self.m, self.e))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_size, m=self.m, e = self.e))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = transform_activation(x, self.e, self.m, self.block_size)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def complete_block_resnet101(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000, acte=5, actm=6):
    """Constructs a ResNet101 model
    """

    block_model =  CompleteBlockResNet(Bottleneck, [3, 4, 23, 3], block_size=block_size, num_classes=num_classes, e=acte, m=actm)

    if pretrained==True:
        model = torchvision.models.resnet101(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)

    return block_model

def complete_block_resnet50(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000, acte=5, actm=6):
    """ Constructs a ResNet50 model
    """
    block_model = CompleteBlockResNet(Bottleneck, [3, 4, 6, 3], block_size = block_size, num_classes=num_classes, e=acte, m=actm)
    if pretrained==True:
        model = torchvision.models.resnet50(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)
    return block_model

def complete_block_resnet34(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000, acte=5, actm=6):
    """ Constructs a ResNet34 model
    """
    block_model = CompleteBlockResNet(BasicBlock, [3, 4, 6, 3], block_size=block_size, num_classes=num_classes, e=acte, m=actm)

    if pretrained==True:
        model = torchvision.models.resnet34(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)

    return block_model

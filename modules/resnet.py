import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

# Others
from ..DSConv2d import DSConv2d
from ..DSConvEngine import DSConvEngine

# PyTorch
import torch
import torch.nn as nn
import torchvision

def conv3x3(in_planes, out_planes, block_size = 32, stride=1):
    """3x3 convolution with padding"""
    return DSConv2d(in_planes, out_planes, kernel_size=3, block_size=block_size, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, block_size=32, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_size=32, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out+=residual
        out = self.relu(out)

        return out

class BlockResNet(nn.Module):
    def __init__(self, block, layers, block_size=32, num_classes = 1000):
        self.inplanes = 64
        super(BlockResNet, self).__init__()
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

        for m in self.modules():
            if isinstance(m, DSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.alpha, 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, block_size=32, stride=1):
        downsample = None
        if stride!=1 or self.inplanes !=planes*block.expansion:
            downsample = nn.Sequential(
                DSConv2d(self.inplanes, planes*block.expansion, kernel_size=1, block_size=block_size,stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, block_size, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def block_resnet101(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000):
    """Constructs a ResNet101 model
    """

    block_model =  BlockResNet(Bottleneck, [3, 4, 23, 3], block_size=block_size, num_classes=num_classes)

    if pretrained==True:
        model = torchvision.models.resnet101(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)

    return block_model

def block_resnet50(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000):
    """ Constructs a ResNet50 model
    """
    block_model = BlockResNet(Bottleneck, [3, 4, 6, 3], block_size = block_size, num_classes=num_classes)
    if pretrained==True:
        model = torchvision.models.resnet50(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)
    return block_model

def block_resnet34(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000):
    """ Constructs a ResNet34 model
    """
    block_model = BlockResNet(BasicBlock, [3, 4, 6, 3], block_size=block_size, num_classes=num_classes)

    if pretrained==True:
        model = torchvision.models.resnet34(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)

    return block_model

if __name__ == "__main__":
    model = block_resnet50(pretrained=True, block_size=128, bit_nmb = 3)
    print(model)
    input('')
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
    print("There are", k, "BlockConvs parameters")

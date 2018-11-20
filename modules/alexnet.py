import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

# Internal
from DSConv import DSConv2d
from DSConvEngine import DSConvEngine

# PyTorch
import torch
import torch.nn as nn
import torchvision

class BlockAlexNet(nn.Module):
    def __init__(self, block_size=32, num_classes=1000):
        super(BlockAlexNet, self).__init__()
        self.features = nn.Sequential(
            DSConv2d(3,64,block_size=block_size,kernel_size=11,stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            DSConv2d(64,192,block_size=block_size,kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            DSConv2d(192, 384,block_size=block_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DSConv2d(384, 256,block_size=block_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DSConv2d(256, 256,block_size=block_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

def block_alexnet(pretrained=False, bit_nmb=8, block_size=32, num_classes=1000):
    block_model = BlockAlexNet(block_size=block_size, num_classes=num_classes)
    if pretrained:
        model = torchvision.models.alexnet(pretrained=True)
        eng = DSConvEngine(block_size, bit_nmb)
        block_model = eng(model, block_model)

    return block_model

if __name__=="__main__":
    model = block_alexnet(pretrained=True, block_size=128, bit_nmb=3)
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

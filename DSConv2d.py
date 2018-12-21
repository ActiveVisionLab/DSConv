# Other
import math
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd, Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

class DSConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, block_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, KDSBias=False, CDS=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        blck_numb = math.ceil(((in_channels)/(block_size*groups)))
        super(DSConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        # KDS weight From Paper
        self.alpha = Parameter(torch.Tensor(out_channels, blck_numb, *kernel_size))

        # KDS bias From Paper
        self.KDSBias = KDSBias
        self.CDS = CDS

        if KDSBias:
            self.KDSb = Parameter(torch.Tensor(out_channels, blck_numb, *kernel_size))
        if CDS:
            self.CDSw = Parameter(torch.Tensor(out_channels))
            self.CDSb = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def get_weight_res(self):
        # Include expansion of alpha and multiplication with weights to include in the convolution layer here
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Include KDSBias
        if self.KDSBias:
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Handy definitions:
        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth//nmb_blocks

        llb = total_depth-(nmb_blocks-1)*bs

        # Casting the Alpha values as same tensor shape as weight
        for i in range(nmb_blocks):
            length_blk = llb if i==nmb_blocks-1 else bs

            shp = self.alpha.shape # Notice this is the same shape for the bias as well
            to_repeat=self.alpha[:, i, ...].view(shp[0],1,shp[2],shp[3]).clone()
            repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i*bs:(i*bs+length_blk), ...] = repeated.clone()

            if self.KDSBias:
                to_repeat = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i*bs:(i*bs+length_blk), ...] = repeated.clone()

        if self.CDS:
            to_repeat = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repeat.expand_as(self.weight)
            print(repeated.shape)

        # Element-wise multiplication of alpha and weight
        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias:
            weight_res = torch.add(weight_res, KDSBias_res)
        return weight_res

    def forward(self, input):
        # Get resulting weight
        weight_res = self.get_weight_res()

        # Returning convolution
        return F.conv2d(input, weight_res, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)

if __name__ == "__main__":
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.conv1 = DSConv2d(3, 10, 3, 32, KDSBias=True, CDS=True)
            self.fc1 = nn.Linear(10*3*3, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = x.view(-1, 10*3*3)
            x = self.fc1(x)
            return x

    net = Test()
    criterion = nn.CrossEntropyLoss()
    inputs = np.random.rand(1, 3, 32, 32)
    labels = np.random.random_integers(9, size=(100))
    outputs = net(torch.Tensor(inputs))
    loss = criterion(outputs, torch.Tensor(labels).long())
    loss.backward()

    for mod in net.modules():
        if isinstance(mod, DSConv2d):
            print(mod.alpha)
            print(mod.weight)
            print(mod.KDSb)

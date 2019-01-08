# DSConv: Efficient Convolution Operator

This repository contains the code used to implement the system reported in the DSConv paper. The code is based on PyTorch and implements the convolution operator, the quantization method and the quantization of the activations. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

In order to run the code in your machine (tested in Ubuntu 16), you will need the following python modules:

+ [Python v 3.5.2](https://realpython.com/installing-python/)
+ [PyTorch 0.4.1](https://pytorch.org/get-started/locally/)

In case you need to use mobilenet, you will have to clone ericsun99's repo [MobileNet-V2-Pytorch](https://github.com/ericsun99/MobileNet-V2-Pytorch) in the DSConv folder:

```
cd path/to/DSConv/
git clone https://github.com/ericsun99/MobileNet-V2-Pytorch
```

### Installing and Usage


DSConv is a python package, and the DSConv module inherit from pytorch modules. It can be used as any pytorch layer and it can be substituted promptly into any model.
To use any of the built models you can just import them as:
```
from DSConv.modules.alexnet import block_alexnet
from DSConv.modules.resnet import block_resnet34, block_resnet50, block_resnet101
from DSConv.modules.mobilenet import block_mobilenet
```

Each module has 4 arguments `(pretrained, bit_nmb, block_size, num_classes)`. The `pretrained` module simply converts from ImageNet trained using Conv to DSConv with no retraining. The `bit_nmb` argument specifies the number of bits from the module. `block_size` specifies the number of channels that are contained in one module, and `num_classes` is the number of outputs in inference. The module can be run (and trained) as:

```
module = block_resnet50(True, 3, 128, 1000)
output = module(image)
```

## Reference
Marcelo Gennari, Roger Fawcett, Victor Adrian Prisacariu, [*DSConv: Efficient Convolution Operator*](https://arxiv.org/abs/1901.01928)


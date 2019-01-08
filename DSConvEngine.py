# Internal
from .DSConv2d import DSConv2d
from .Quantizer import Quantizer

# Others
import math
import time

# Pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DSConvEngine:

    def __init__(self, bs_size, nmb_bits):
        self.bs_size = bs_size
        self.nmb_bits = nmb_bits
        self.quantizer = Quantizer(nmb_bits)

    def toDSConv(self, model):
        alpha_tensors = []
        int_tensors = []
        for mod in model.modules():
            if isinstance(mod, nn.modules.conv.Conv2d):
                groups = mod.groups
                new_tensor, sblck, alpValues = self.tensor_to_block(mod.weight.data, groups)
                mod.weight.data = new_tensor
                int_tensors.append(sblck)
                alpha_tensors.append(alpValues)

        return model, int_tensors, alpha_tensors

    def tensor_to_block(self, array, groups=1):
        # Assuming the shape of the array to be:
        # [channel, depth, height, width]
        new_tensor = torch.empty(array.shape)
        int_tensor = torch.empty(array.shape)
        nmb_blocks = math.ceil((array.shape[1])/(self.bs_size*groups))
        bs = self.bs_size
        alp_tensor = torch.empty((array.shape[0], nmb_blocks, array.shape[2],
                              array.shape[3]))

        # Iterates through every block
        for i in range(nmb_blocks):
            if i ==nmb_blocks-1:
                blck, sblck, alp = self.quantizer.quantize_block(array[:, i*bs:, ...])
                new_tensor[:, i*bs:, ...] = blck
                int_tensor[:, i*bs:, ...] = sblck
                alp_tensor[:, i, ...] = alp
            else:
                blck, sblck, alp = self.quantizer.quantize_block(array[:, i*bs:(i+1)*bs, ...])
                new_tensor[:, i*bs:(i+1)*bs, ...] = blck
                int_tensor[:, i*bs:(i+1)*bs, ...] = sblck
                alp_tensor[:, i, ...] = alp

        return new_tensor, int_tensor, alp_tensor

    def __call__(self, model, block_model):
        print("Returning pretrained model with bit length", self.nmb_bits, "and block size of", self.bs_size)

        start = time.time()
        new_model, sblocks, alpha_tensors = self.toDSConv(model)

        i = 0
        for bmod, mod, newmod in zip(block_model.modules(), model.modules(), new_model.modules()):
            if isinstance(bmod, DSConv2d):
                prev = bmod.intweight.data.shape
                bmod.intweight.data = sblocks[i]
                pos = bmod.intweight.data.shape
                assert(prev==pos), "Original model int weight must be the same shape as DSConv version"
                prev = bmod.alpha.data.shape
                bmod.alpha.data = alpha_tensors[i]
                pos = bmod.alpha.data.shape
                assert(prev==pos), "Original model alpha values must be the same shape as DSConv version"
                prev = bmod.weight.data.shape
                bmod.weight.data = newmod.weight.data
                pos = bmod.weight.data.shape
                assert(prev==pos), "Original model weights must be the same shape as DSConv version"
                i+=1
            if isinstance(bmod, nn.Linear):
                bmod.weight.data = mod.weight.data.cpu()
                bmod.bias.data = mod.bias.data.cpu()

            if isinstance(bmod, nn.BatchNorm2d):
                bmod.weight.data = mod.weight.data.cpu()
                bmod.bias.data = mod.bias.data.cpu()
                bmod.running_mean.data = mod.running_mean.data.cpu()
                bmod.running_var.data = mod.running_var.data.cpu()

        end = time.time()
        print("It took", end-start, "seconds for conversion")

        return block_model

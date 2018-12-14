import numpy as np
import math
import time

# PyTorch
import torch

def transform_activation(tensor, mantissa, exponent, blk):
    # Assuming the shape of the array to be:
    # [channel, depth, height, width]
    shp = tensor.shape
    number_of_blocks = math.ceil(shp[1]/blk)
    for i in range(number_of_blocks):
        if i == number_of_blocks-1:
            tensor[:, i*blk:, ...] = to_block(tensor[:, i*blk:, ...])
        else:
            tensor[:, i*blk:(1+i)*blk, ...] = to_block(tensor[:, i*blk:(1+i)*blk, ...])
    return tensor

def to_block(blocks, mant=5):
    # This receives a shape of the array of:
    # [channel, bs_size, h, w]
    blocks = quantize(blocks)
    #for h in range(blocks.shape[1]):
    #    for l in range(blocks.shape[2]):
    #        blocks[:, h, l] = quantize(blocks[:, h, l])
    return blocks

def quantize(activations, EXPONENT_WIDTH=4, MANTISSA_WIDTH=7):
    # This receives an array of shape:
    # [channel, bs_size, h, w]
    #print(activations[1, 0:32, 1, 1])
    int_log = find_exponent(activations, EXPONENT_WIDTH)
    max_exponent = find_max_exponent(int_log)
    quantized_activations = to_exponent_mantissa_width(activations, max_exponent, MANTISSA_WIDTH)
    return quantized_activations

def find_exponent(array, EXPONENT_WIDTH):
    MAX = 2**(EXPONENT_WIDTH-1)-1
    MIN = -2**(EXPONENT_WIDTH-1)
    absolute = torch.abs(array)
    value_log = torch.log2(absolute)
    value_log = torch.clamp(value_log, MIN, MAX)
    int_log = torch.floor(value_log)
    return int_log

def find_max_exponent(array):
    max_exponent, _ = torch.max(array, dim=1)
    return max_exponent

def to_exponent_mantissa_width(array, maxlog, MANTISSA_WIDTH):
    shp = array.shape
    maxlog = maxlog.unsqueeze(1)
    # NOTE THAT THIS -1 IS BECAUSE OF THE LEADING 1
    exponent_needed = (MANTISSA_WIDTH-maxlog-1)*torch.ones(shp).cuda()
    first_mant_w = torch.pow(2, exponent_needed)
    array = array*first_mant_w
    # Half LSB rounding:
    array = torch.round(array)
    array = array/first_mant_w

    return array

import math
import time

# PyTorch
import torch

def transform_activation(tensor, exponent, mantissa, blk):
    # Assuming the shape of the array to be:
    # [channel, depth, height, width]
    # This is going to split the tensor into two:
        # 1) one with dimensions [number_of_blocks, channel, blk, height, width]
        # 2) the other with dimensions [1, channel, rest, height, width]
    # Assuming that shp[1] is divisible by blk, there will be only one tensor
    # This slicing will guarantee that the transformation to BFP can be done in parallel, which should accelerate the computation

    shp = tensor.shape
    number_of_blocks = math.ceil(shp[1]/blk)
    if shp[1] % blk == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.reshape(tensor, (number_of_blocks, shp[0], blk, shp[2], shp[3]))
        tensor = quantize(tensor, exponent, mantissa)
        tensor = torch.reshape(tensor, (1, shp[0], shp[1], shp[2], shp[3]))
        tensor = tensor[0, ...]
        return tensor

    else:
        # shp[1] is not divisible by block size
        # Therefore two tensors will be created
        input('Activation is not divisible by block size')
        tensor = torch.unsqueeze(tensor, 0)

        if number_of_blocks == 1:
            # This means that the depth is less than the block size, so just one tensor will be created
            tensor = quantize(tensor, exponent, mantissa)
            tensor = tensor[0, ...]
            return tensor
        else:
            tensor1 = tensor[0:number_of_blocks-1, shp[0], blk, :]
            tensor1 = quantize(tensor1, exponent, mantissa)
            tensor1 = torch.reshape(tensor1, (1, shp[0], (number_of_blocks-1)*blk, shp[2], shp[3]))
            tensor2 = tensor[number_of_blocks, shp[0], (number_of_blocks-1)*blk:, ...]
            tensor2 = quantize(tensor2, exponent, mantissa)
            tensor2 = torch.reshape(tensor2, (1, shp[0], shp[1]-(number_of_blocks-1)*blk, shp[2], shp[3]))
            tensor[0, 0:(number_of_blocks-1)*blk, ...] = tensor1
            tensor[0, (number_of_blocks-1)*blk:, ...] = tensor2
            tensor = tensor[0, ...]
            return tensor

    return tensor

def quantize(activations, EXPONENT_WIDTH, MANTISSA_WIDTH):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    int_log = find_exponent(activations, EXPONENT_WIDTH)
    max_exponent = find_max_exponent(int_log)
    quantized_activations = to_exponent_mantissa_width(activations, max_exponent, MANTISSA_WIDTH)
    return quantized_activations

def find_exponent(array, EXPONENT_WIDTH):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    MAX = 2**(EXPONENT_WIDTH-1)-1
    MIN = -2**(EXPONENT_WIDTH-1)
    absolute = torch.abs(array)
    value_log = torch.log2(absolute)
    value_log = torch.clamp(value_log, MIN, MAX)
    int_log = torch.floor(value_log)
    return int_log

def find_max_exponent(array):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    max_exponent, _ = torch.max(array, dim=2)

    # The return is of shape [number_of_blocks, channel, h, w]
    return max_exponent

def to_exponent_mantissa_width(array, maxlog, MANTISSA_WIDTH):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    shp = array.shape
    maxlog = maxlog.unsqueeze(2)
    # NOTE THAT THIS -2 IS BECAUSE OF THE LEADING 1 AND THE FACT THAT THIS SHOULD BE IN 2s COMPLEMENT
    exponent_needed = (MANTISSA_WIDTH-maxlog-2)*torch.ones(shp).cuda()
    first_mant_w = torch.pow(2, exponent_needed)
    array = array*first_mant_w
    # Half LSB rounding:
    array = torch.round(array)
    # print(array[0, :, 0, 0]) # Uncomment to print integer values
    array = array/first_mant_w

    return array

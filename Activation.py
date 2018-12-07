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

    # Iterates through every channel
    for c in range(shp[0]):
        for i in range(number_of_blocks):
            if i == number_of_blocks -1:
                # Means this is the last block and possibly thinner 
                tensor[c, i*blk:, ...] = to_block(tensor[c, i*blk:, ...])
            else:
                tensor[c, i*blk:(1+i)*blk, ...] = to_block(tensor[c, i*blk:(1+i)*blk, ...])
    return tensor

def to_block(blocks, mant=5):
    # This receives a shape of the array of:
    # [bs_size, h, w]
    blocks = quantize(blocks)
    #for h in range(blocks.shape[1]):
    #    for l in range(blocks.shape[2]):
    #        blocks[:, h, l] = quantize(blocks[:, h, l])
    return blocks

def quantize(activations, mant=5):
    # This receives an array of shape:
    # [bs_size, h, w]
    absAct = torch.abs(activations)
    logAbs = torch.log2(absAct)
    intlogAbs = logAbs.type(torch.cuda.IntTensor)
    maxlogAbs, _ = torch.max(intlogAbs, dim=0)
    value = to_nearest(activations, maxlogAbs)
    return value

def to_nearest(array, maxlog, mant=5):
    # This receives arrays of size:
    # [bs_size, h, w]
    # [h, w]
    maxlog = maxlog.unsqueeze(0)
    maxlog = torch.clamp(maxlog, -10, 10) # To avoid inf
    maxlog.expand(array.shape[0], array.shape[1], array.shape[2])
    mantT = 5*torch.ones(array.shape).type(torch.cuda.IntTensor)
    diff = mantT-maxlog
    scale = torch.pow(2, diff).type(torch.cuda.FloatTensor)
    array = array*scale
    array = torch.clamp(array, -2**(mant-1), 2**(mant-1)-1)
    array = torch.round(array)
    array = array.type(torch.cuda.FloatTensor)
    array = array/scale
    return array



def transform_activation2(tensor, mantissa, exponent, blk):
    """
    This is a block floating point implementation
    Each block will share an exponent. We first pick the largest exponent and shift the rest
    :params: tensor is the tensor to be converted
    :params: mantissa is the number of bits that the mantissa will have
    :params: exponent is the number of bits that the shared exponent will have
    :params: blk is the block size of the tensor
    :return: tensor in block floating point format
    """


    # Just converting the tensor from torch to numpy
    shp = tensor.shape
    number_of_blocks = math.ceil(shp[1]/blk)
    # Loop through every block
    tensor = torch.clamp(tensor, -1, 4)
    precision = 1024
    tensor = precision*tensor
    tensor = tensor.to(torch.int32)
    tensor = tensor.to(torch.float32)
    tensor = tensor/precision
    #for w in range(shp[3]):
    #    for h in range(shp[2]):
    #        for c in range(shp[0]):
    #            for blk in range(number_of_blocks):

    #                if blk == number_of_blocks-1:
    #                    block = tensor[c, blk*number_of_blocks:, w, h]
    #                else:
    #                    block = tensor[c, blk*number_of_blocks:(blk+1)*number_of_blocks, w, h]
    #                block = quantize_block(block, 3, 3)

    #                if blk == number_of_blocks-1:
    #                    tensor[c, blk*number_of_blocks:, w, h] = block
    #                else:
    #                    tensor[c, blk*number_of_blocks:(blk+1)*number_of_blocks, w, h] = block
    return tensor

def quantize_block(block, mantissa, exponent):
    """Given a block, this function chooses the biggest exponent value and shifts all other values as in Block Floating Point

    :block: numpy array of the values in the block
    :mantissa: number of bits for the mantissa
    :exponent: number of bits for the exponent
    :returns: quantized block according to mantissa and exponent
    """
    #maximum = max(abs(block))
    #print(block)
    #values = generate_mantissa(mantissa, exponent)
    #print(values)
    #input('')
    precision= 10
    block = precision*block
    block = block.to(torch.int32)
    block = block.to(torch.float32)
    block = block/precision
    return block

def generate_mantissa(mantissa, exponent):
    """This method gets a mantissa and exponent number of bits and generates a
    list with all possible values

    :exponent: Number of bits for the exponent
    :mantissa: Number of bits for the mantissa
    :returns: List with all possible values for the mantissa

    """
    values = []
    for i in range(2**exponent):
        if i == 0:
            new_values = [(x/2**mantissa) for x in range(2**mantissa)]
        else:
            new_values = [2**(i-1)*(1+(x/(2**mantissa))) for x in range(2**mantissa)]
        values.extend(new_values)
    return values

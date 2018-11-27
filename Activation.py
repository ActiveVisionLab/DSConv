import numpy as np
import math

def transform_activation(tensor, mantissa, exponent, blk):
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
    tensor = tensor.detach().numpy()
    shp = tensor.shape
    number_of_blocks = math.ceil(shp[1]/blk)

    # Loop through every block
    for w in range(shp[3]):
        for h in range(shp[2]):
            for c in range(shp[0]):
                for blk in range(number_of_blocks):
                    if blk == number_of_blocks-1:
                        block = tensor[c, blk*number_of_blocks:, w, h]
                    else:
                        block = tensor[c, blk*number_of_blocks:(blk+1)*number_of_blocks, w, h]
                    print(len(block))
                    input('')

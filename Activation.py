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
    tensor = tensor.detach().cpu().numpy()
    shp = tensor.shape
    number_of_blocks = math.ceil(shp[1]/blk)
    print(shp)
    # Loop through every block
    for w in range(shp[3]):
        for h in range(shp[2]):
            for c in range(shp[0]):
                for blk in range(number_of_blocks):
                    if blk == number_of_blocks-1:
                        block = tensor[c, blk*number_of_blocks:, w, h]
                    else:
                        block = tensor[c, blk*number_of_blocks:(blk+1)*number_of_blocks, w, h]
                    quantize_block(block, 3, 3)

def quantize_block(block, mantissa, exponent):
    """Given a block, this function chooses the biggest exponent value and shifts all other values as in Block Floating Point

    :block: numpy array of the values in the block
    :mantissa: number of bits for the mantissa
    :exponent: number of bits for the exponent
    :returns: quantized block according to mantissa and exponent
    """
    maximum = max(abs(block))
    print(block)
    values = generate_mantissa(mantissa, exponent)
    print(values)
    input('')
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

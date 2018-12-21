# Internal
from .DSConv2d import DSConv2d
from .Quantizer import Quantizer

# Others
import math
import numpy as np
from tqdm import tqdm
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
                mod.weight.data = torch.tensor(new_tensor).float()
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
        for bmod, mod in zip(block_model.modules(), model.modules()):
            if isinstance(bmod, DSConv2d):
                prev = bmod.weight.data.shape
                bmod.weight.data = sblocks[i]
                pos = bmod.weight.data.shape
                assert(prev==pos), "Original model weight mus be the same shape as DSConv version"
                prev = bmod.alpha.data.shape
                bmod.alpha.data = alpha_tensors[i]
                pos = bmod.alpha.data.shape
                assert(prev==pos), "Original model alpha values must be the same shape as DSConv version"
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

if __name__ == "__main__":
    model = torchvision.models.resnet101(pretrained=True)
    model.eval()

    test = DSConvEngine(32, 3)
    new_model, sblocks, alpha_tensors = test.toDSConv(model)

    print("Done Converting to BFP")

    for i in range(len(sblocks)):
        alpha = alpha_tensors[i]
        int_block = sblocks[i]
        nmb_blocks = alpha.shape[1]
        total_depth = int_block.shape[1]
        bs = total_depth//nmb_blocks
        leftover_last_block = total_depth-(nmb_blocks-1)*bs
        print("Number blocks:", nmb_blocks)
        print("Block size:", bs)
        print("Total depth:", total_depth)
        print("Leftover:", leftover_last_block)
        alpha_result = np.empty(int_block.shape)
        for i in range(nmb_blocks):
            if i == nmb_blocks-1:
                # Leftover
                shp = alpha.shape
                to_repeat = alpha[:, i, ...].reshape((shp[0], 1, shp[2], shp[3]))
                alpha_result[:,i*bs:,...]=np.repeat(to_repeat, leftover_last_block, axis=1)
                input('')
            else:
                # Use block size
                shp = alpha.shape
                to_repeat = alpha[:, i, ...].reshape((shp[0], 1, shp[2], shp[3]))
                alpha_result[:,i*bs:(i+1)*bs,...]=np.repeat(to_repeat, bs, axis=1)

        effective_weight = np.multiply(alpha_result,int_block)
        print(alpha_result.shape)
        print(int_block.shape)
        print(effective_weight.shape)
        print(alpha_result[0])
        print(alpha_result[1])
        input('')

    # Evaluating new Model
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Loading Data
    valdir = '/home/marcelo/storage/ILSVRC2012/ILSVRC2012_val/'
    data = datasets.ImageFolder(valdir, preprocess)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, preprocess),
        batch_size = 20,
        shuffle = False,
        num_workers = 10,
        pin_memory=True)

    correct, total = 0, 0
    print("Starting Eval")
    with torch.no_grad():
        for i, (inputImage, target) in tqdm(enumerate(val_loader)):
            output = new_model(inputImage)
            _, predicted = torch.max(output.data, 1)

            total+=target.size(0)
            correct+=(predicted==target).sum().item()

            if i%100==0:
                print(i)
        print("Correct:", correct)
        print("Total:", total)
        print("Accuracy:", correct/total)

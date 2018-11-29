import torch
import torch.nn as nn


class ActivationEngine():

    """Converts from pretrained resnet to pretrained truncated resnet. """
    def __call__(self, model, activation_model):
        print("Returning pretrained model with truncated Activation")
        for tmod, mod in zip(activation_model.modules(), model.modules()):
            if isinstance(tmod, nn.Conv2d):
                tmod.weight.data = mod.weight.data.cpu()

            elif isinstance(tmod, nn.Linear):
                tmod.weight.data = mod.weight.data.cpu()
                tmod.bias.data = mod.weight.data.cpu()

            elif isinstance(tmod, nn.BatchNorm2d):
                tmod.weight.data =  mod.weight.data.cpu()
                tmod.bias.data = mod.bias.data.cpu()
                tmod.running_mean.data = mod.running_mean.data.cpu()
                tmod.running_var.data = mod.running_var.data.cpu()

        return activation_model

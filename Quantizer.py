# Other
import numpy as np

# PyTorch
import torch
import torch.nn as nn

class QuantizerTorch:
    def __init__(self, nmb_bits):
        self.nmb_bits = nmb_bits
        self.maxV = pow(2, self.nmb_bits-1)-1
        self.minV = -1*pow(2, self.nmb_bits-1)

    def quantize_block(self, blcknump, debug=False, alpha_calc='L2'):
        # Here we assume that blcknump will be of size:
            # [nmb_blocks, channel, blk, height, width]
        # The idea is to do that in parallel using torch instead of iteratively
        # Finding the Scaling to quantize using full range
        self.original_blck = blcknump
        blcknump = np.asarray(blcknump)
        maxPos = np.max(blcknump)
        maxNeg = np.min(blcknump)

        absmax = maxPos if abs(maxPos) > abs(maxNeg) else maxNeg
        factor = +1 if abs(maxPos) > abs(maxNeg) else -1

        sc = self.minV/absmax

        # Here we have to choose the lowest scaling because otherwise we will
        # go above the number of bits specified
        self.scaling = sc
        self.scaled_blck = np.rint(self.scaling*blcknump)

        # Here we find the alpha value that minimizes 2-norm
        if alpha_calc=='L2':
            self._finding_alpha_()
        elif alpha_calc=='KL':
            self._finding_alpha_KL_()
        else:
            print("Using L2")
            self._finding_alpha_()

        if(debug):
            print("Max and minimum values:", self.maxV, self.minV)
            print("Original block:", self.original_blck)
            print("Scaled block:", self.scaled_blck)
            print("Scaling applied:", self.scaling)
            print("Alpha calculated:", self.alpha)
            print("Resulting effective block:",self.final_blck)
            input('')

        return self.final_blck, self.scaled_blck, self.alpha
class Quantizer:

    def __init__(self, nmb_bits):
        self.nmb_bits = nmb_bits
        self.maxV = pow(2, self.nmb_bits-1)-1
        self.minV = -1*pow(2, self.nmb_bits-1)

    def quantize_block(self, blcknump, debug=False, alpha_calc='L2'):
        # Finding the Scaling to quantize using full range
        self.original_blck = blcknump
        blcknump = np.asarray(blcknump)
        maxPos = np.max(blcknump)
        maxNeg = np.min(blcknump)

        absmax = maxPos if abs(maxPos) > abs(maxNeg) else maxNeg
        factor = +1 if abs(maxPos) > abs(maxNeg) else -1

        sc = self.minV/absmax

        # Here we have to choose the lowest scaling because otherwise we will
        # go above the number of bits specified
        self.scaling = sc
        self.scaled_blck = np.rint(self.scaling*blcknump)

        # Here we find the alpha value that minimizes 2-norm
        if alpha_calc=='L2':
            self._finding_alpha_()
        elif alpha_calc=='KL':
            self._finding_alpha_KL_()
        else:
            print("Using L2")
            self._finding_alpha_()

        if(debug):
            print("Max and minimum values:", self.maxV, self.minV)
            print("Original block:", self.original_blck)
            print("Scaled block:", self.scaled_blck)
            print("Scaling applied:", self.scaling)
            print("Alpha calculated:", self.alpha)
            print("Resulting effective block:",self.final_blck)
            input('')

        return self.final_blck, self.scaled_blck, self.alpha

    def _finding_alpha_KL_(self):
        """ Finds the KDS value by minimizing KL-Divergence
        """
        torch.set_printoptions(precision=10)
        orig = torch.Tensor(self.original_blck)
        scaled = torch.Tensor(self.scaled_blck)
        alpha = torch.tensor([0.1], requires_grad=True)
        alpha_expanded = alpha.expand_as(scaled)
        log_sft = nn.LogSoftmax(dim=0)
        criteria = nn.KLDivLoss()
        sft = nn.Softmax(dim=0)
        optimizer = torch.optim.RMSprop([alpha], lr=0.0001)
        for epoch in range(1000):
            final = scaled*alpha_expanded
            optimizer.zero_grad()

            target = sft(orig)
            inp = log_sft(final)
            loss = criteria(inp, target)
            loss.backward(retain_graph=True)
            optimizer.step()

        self.alpha = alpha.detach().numpy()
        self.final_blck = self.scaled_blck*self.alpha

    def _finding_alpha_(self):
        """ Find the KDS value by minimizing L2 norm
        """
        # Applying minimum squares we can find a value of the "bonus multiply"
        # that minimizes the square distance to the original block
        numerator = np.dot(self.original_blck, self.scaled_blck)
        denominator = np.dot(self.scaled_blck, self.scaled_blck)
        self.alpha = numerator/denominator
        self.final_blck = self.scaled_blck * self.alpha

if __name__=="__main__":
    test = Quantizer(4)
    example_block = [0.3, 0.1, 0.4, -0.5, -.3, -.4, -.5, -.5, 1]
    test.quantize_block(example_block, debug = True)

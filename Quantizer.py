# PyTorch
import torch
import torch.nn as nn

class Quantizer:

    def __init__(self, nmb_bits):
        self.nmb_bits = nmb_bits
        self.minV = -1*pow(2, self.nmb_bits-1)
        self.maxV = pow(2, self.nmb_bits-1)-1

    def quantize_block(self, blcknump, debug=False, alpha_calc='L2'):
        # Finding the Scaling to quantize using full range
        # This receives the block in shape [channel, blk, height, width]

        self.original_blck = blcknump

        absblcknump = torch.abs(blcknump)
        _, indexPos = torch.max(absblcknump, dim=1)
        absmax = torch.gather(blcknump, 1, indexPos.unsqueeze(1))

        self.scaling = self.minV/absmax

        # Half LSB rounding
        self.scaled_blck = torch.round(self.scaling*blcknump)

        # In case a value was 3.8 and was rounded to 4 for 3 bit for example
        #self.scaled_blck = torch.clamp(self.scaled_blck, min =self.minV, max=self.maxV)

        # Here we find the alpha value that minimizes 2-norm
        self._finding_alpha_KL_() if alpha_calc=='KL' else self._finding_alpha_()

        if debug:
            self._report_()

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
        numerator = (self.original_blck*self.scaled_blck).sum(dim=1)
        denominator = (self.scaled_blck*self.scaled_blck).sum(dim=1)
        self.alpha = numerator/denominator
        self.final_blck = self.scaled_blck * self.alpha.unsqueeze(1)

    def _report_(self):
        print("Max absolute value:", self.minV)
        print("Original block:", self.original_blck)
        print("Scaled block:", self.scaled_blck)
        print("Scaling applied:", self.scaling)
        print("Alpha calculated:", self.alpha)
        print("Resulting effective block:",self.final_blck)
        input('Press key to continue...')


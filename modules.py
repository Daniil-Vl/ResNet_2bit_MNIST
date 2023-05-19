from typing import Any, Union
import torch
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F


class ActFn(torch.autograd.Function):
    """
    Quantization function for activations
    """
    @staticmethod
    def forward(ctx, x, alpha, bitwidth: int = 2):
        ctx.save_for_backward(x, alpha)
        y = torch.clamp(x, min=0, max=alpha.item())
        scale = (2**bitwidth - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, grad_outputs):
        x, alpha = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(
            grad_outputs * torch.ge(x, alpha).float()).view(-1)
        return grad_outputs * x_range.float(), grad_alpha, None


def quantize_k(r_i, k):
    scale = (2**k - 1)
    r_q = torch.round(scale * r_i) / scale
    return r_q


class DoReFaQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_i, bitwidth=2):
        tanh = torch.tanh(r_i).float()
        r_q = 2 * \
            quantize_k(
                tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5, bitwidth) - 1
        return r_q

    @staticmethod
    def backward(ctx, grad_outputs):
        # Straight Through Estimator (STE)
        return grad_outputs, None


class Conv2d(torch.nn.Conv2d):
    """
    Convolution layer with DoReFa quantization
    """

    def __init__(self, in_places, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False, bitwidth=2):
        super(Conv2d, self).__init__(in_places, out_planes,
                                     kernel_size, stride, padding, groups, dilation, bias)
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth

    def forward(self, x):
        quantized_weights = self.quantize(self.weight, self.bitwidth)
        y = F.conv2d(x, quantized_weights, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return y


class Linear(torch.nn.Linear):
    """
    Linear layer with DoReFa quantization
    """

    def __init__(self, in_features, out_features, bias=True, bitwidth=2):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth

    def forward(self, x):
        quantized_weights = self.quantize(self.weight, self.bitwidth)
        y = F.linear(x, quantized_weights, self.bias)
        return y

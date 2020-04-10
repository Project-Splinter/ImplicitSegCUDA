import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from implicit_seg.cuda import interp2x_boundary2d

class Interp2xBoundary2dFunction(Function):
    @staticmethod
    def forward(ctx, input, balance_value):
        output, is_boundary = \
            interp2x_boundary2d.forward(input, balance_value)
        # ctx.save_for_backward(alpha3d, alpha2d)
        return output, is_boundary

    # @staticmethod
    # def backward(ctx, grad_alpha2d):
    #     alpha3d, alpha2d = ctx.saved_tensors
    #     d_alpha3d, = softras_render_cuda.backward(
    #         grad_alpha2d.contiguous(), alpha3d.contiguous(), alpha2d.contiguous())
    #     return d_alpha3d


class Interp2xBoundary2d(nn.Module):
    def __init__(self, balance_value=0.5):
        super(Interp2xBoundary2d, self).__init__()
        self.balance_value = balance_value

    def forward(self, input):
        return Interp2xBoundary2dFunction.apply(input, self.balance_value)



            

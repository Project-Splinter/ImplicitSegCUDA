import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from implicit_seg.cuda import UpSampleBilinear2d_cuda

class UpSampleBilinear2dFunction(Function):
    @staticmethod
    def forward(ctx, input, balance_value):
        output, is_boundary = UpSampleBilinear2d_cuda.forward(input, balance_value)
        # ctx.save_for_backward(alpha3d, alpha2d)
        return output, is_boundary

    # @staticmethod
    # def backward(ctx, grad_alpha2d):
    #     alpha3d, alpha2d = ctx.saved_tensors
    #     d_alpha3d, = softras_render_cuda.backward(
    #         grad_alpha2d.contiguous(), alpha3d.contiguous(), alpha2d.contiguous())
    #     return d_alpha3d


class UpSampleBilinear2d(nn.Module):
    def __init__(self, balance_value=0.5):
        super(UpSampleBilinear2d, self).__init__()
        self.balance_value = balance_value

    def forward(self, input):
        return UpSampleBilinear2dFunction.apply(input, self.balance_value)

if __name__ == "__main__":
    import tqdm
    input = torch.Tensor([[[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ]]]).to("cuda:0").float()
    print (input.shape)
    with torch.no_grad():
            
        upsampler = UpSampleBilinear2d()
        output2, boundary2 = upsampler(input)
        print (output2)
        print (boundary2)
        
    #     output1 = F.interpolate(input, (5, 9), mode="bilinear", align_corners=True)
    #     valid1 = F.interpolate((input > 0.5).float(), (5, 9), mode="bilinear", align_corners=True)
    #     boundary1 = (valid1 > 0) & (valid1 < 1)
    #     print (boundary1)
    #     assert torch.equal(output2, output1)

    input = torch.randn(16, 128, 257, 257).cuda()
    with torch.no_grad():
        # for _ in tqdm.tqdm(range(100)):
        #     output1 = F.interpolate(input, (257*2-1, 257*2-1), mode="bilinear", align_corners=True)
        #     valid1 = F.interpolate((input > 0.5).float(), (257*2-1, 257*2-1), mode="bilinear", align_corners=True)
        #     boundary1 = (valid1 > 0) & (valid1 < 1)
        #     torch.cuda.synchronize()

        # output2, boundary2 = upsampler(input)
        # print (boundary1[0, 0, 0, 0:10])
        # print (boundary2[0, 0, 0, 0:10])

        for _ in tqdm.tqdm(range(100)):
            output2, boundary2 = upsampler(input)
            torch.cuda.synchronize()

            

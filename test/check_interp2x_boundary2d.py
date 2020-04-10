import torch
import torch.nn.functional as F

from implicit_seg.functional import Interp2xBoundary2d

if __name__ == "__main__":
    import tqdm
    input = torch.Tensor([[[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ]]]).to("cuda:0").float()
    input = torch.randn(16, 128, 257, 257).to("cuda:0").float()
    input.requires_grad = True

    upsampler = Interp2xBoundary2d(balance_value=0.5)

    # # check forward equalty
    # output, boundary = upsampler(input)

    # output0 = F.interpolate(
    #     input, (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
    # valid0 = F.interpolate(
    #     (input > 0.5).float(), (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
    # boundary0 = (valid0 > 0) & (valid0 < 1)
    
    # assert torch.allclose(output, output0, rtol=0, atol=1e-6), \
    #     "forward 'output' is not equal!"
    # print ("check equalty forward() pass!")

    # # check backward equalty
    # if input.grad is not None:
    #     input.grad.zero_()
    # output, boundary = upsampler(input)
    # output.sum().backward()
    # grad = input.grad.clone()

    # if input.grad is not None:
    #     input.grad.zero_()
    # output0 = F.interpolate(
    #     input, (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
    # valid0 = F.interpolate(
    #     (input > 0.5).float(), (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
    # boundary0 = (valid0 > 0) & (valid0 < 1)
    # output0.sum().backward()
    # grad0 = input.grad.clone()

    # assert torch.allclose(grad, grad0, rtol=0, atol=1e-6), \
    #     "backward 'grad output' is not equal!"
    # print ("check equalty backward() pass!")

    # # timeit forward
    # print ("forward timeit")
    # with torch.no_grad():
    #     for _ in tqdm.tqdm(range(100)): # 27.42it/s
    #         output0 = F.interpolate(
    #             input, (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
    #         valid0 = F.interpolate(
    #             (input > 0.5).float(), (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
    #         boundary0 = (valid0 > 0) & (valid0 < 1)
    #         torch.cuda.synchronize()

    #     for _ in tqdm.tqdm(range(100)): # 69.09it/s
    #         output, boundary = upsampler(input)
    #         torch.cuda.synchronize()

    # timeit backward
    print ("backward timeit")
    for _ in tqdm.tqdm(range(500)): # 16.97it/s
        if input.grad is not None:
            input.grad.zero_()
        output0 = F.interpolate(
            input, (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
        with torch.no_grad():
            valid0 = F.interpolate(
                (input > 0.5).float(), (input.size(2)*2-1, input.size(3)*2-1), mode="bilinear", align_corners=True)
            boundary0 = (valid0 > 0) & (valid0 < 1)
        output0.sum().backward()

    for _ in tqdm.tqdm(range(500)): # 27.54it/s
        if input.grad is not None:
            input.grad.zero_()
        output, boundary = upsampler(input)
        output.sum().backward()
    
    
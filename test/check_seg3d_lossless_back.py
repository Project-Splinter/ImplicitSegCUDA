import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Seg3dLossless
from implicit_seg.functional import plot_mask3D

resolutions = [
    (8+1, 20+1, 8+1),
    (16+1, 40+1, 16+1),
    (32+1, 80+1, 32+1),
    (64+1, 160+1, 64+1),
    (128+1, 320+1, 128+1),
]
align_corners = False

def query_func(tensor, points):
    """
    [Essential!] here align_corners should be same
    with how you process gt through interpolation.
    tensor: (bz, 1, H, W, D)
    points: [bz,] list of (N, 3)
    """
    bz = len(points)
    occupancys = [ 
        F.grid_sample(
            tensor[i].unsqueeze(0), 
            points[i].view(1, 1, 1, -1, 3),
            mode="bilinear",
            align_corners=align_corners,
            padding_mode="border", # to align with F.interpolate
        )[0, 0, 0, 0].unsqueeze(0) for i in range(bz)
    ]
    return occupancys
if __name__ == "__main__":
    import tqdm 
    import os 
    
    lr = 1e-1
    niter = 100000
    MSE = nn.MSELoss()

    # gt
    query_sdfs = torch.load(
        "./data/sdf.pth").to("cuda:0").float() # [1, 1, H, W, D]
    print ("data:", query_sdfs.shape)

    if type(resolutions[-1]) is int:
        final_W, final_H, final_D = resolutions[-1], resolutions[-1], resolutions[-1]
    else:
        final_W, final_H, final_D = resolutions[-1]
    gt = F.interpolate(
        query_sdfs, (final_D, final_H, final_W), mode="trilinear", align_corners=align_corners)
    # gt = (gt > 0.0).float()
    print ("gt:", gt.shape)

    # input
    input = torch.rand_like(gt)
    input.requires_grad = True
    print("input:", input.shape)

    # infer
    engine = Seg3dLossless(
        query_func = query_func, 
        b_min = [[-1.0, -1.0, -1.0]],
        b_max = [[1.0, 1.0, 1.0]],
        resolutions = resolutions,
        align_corners = align_corners,
        balance_value = 0.,
        device="cuda:0", 
        visualize=False
    )

    # training
    solver = torch.optim.Adam([input], lr = lr)
    pbar = tqdm.tqdm(range(niter))

    for i in pbar:
        occupancys = engine(tensor=input)

        solver.zero_grad()
        loss = MSE(occupancys, gt)
        loss.backward()
        solver.step()

        if input.grad is not None:
            input.grad.data.zero_()

        state_msg = (f'loss: {loss.item(): .6f}')
        pbar.set_description(state_msg)

        if i >20 and i % 10 == 0:
            plot_mask3D(occupancys.detach()[0, 0].to("cpu"))

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Seg3dLossless

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
    # gt
    query_sdfs = torch.load(
        "./data/sdf.pth").to("cuda:0").float() # [1, 1, H, W, D]
    query_sdfs.requires_grad = True
    print (query_sdfs.shape)

    if type(resolutions[-1]) is int:
        final_W, final_H, final_D = resolutions[-1], resolutions[-1], resolutions[-1]
    else:
        final_W, final_H, final_D = resolutions[-1]
    gt = F.interpolate(
        query_sdfs, (final_D, final_H, final_W), mode="trilinear", align_corners=align_corners)
    print ("gt:", gt.shape)

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
    os.makedirs("./data/cache/", exist_ok=True)

    with torch.no_grad():
        for _ in tqdm.tqdm(range(1)):
            sdfs = engine.forward(tensor=query_sdfs)
    print (sdfs.shape)

    for _ in tqdm.tqdm(range(1)):
        sdfs = engine(tensor=query_sdfs)

        sdfs.sum().backward()
        print (sdfs[0, 0, 0, 0, 0:10])
        print (query_sdfs.grad[0, 0, 0, 0, 0:10])

    # cv2.imwrite(
    #    "./data/cache/gen_sdf_sumz.png",
    #    np.uint8(((sdfs[0, 0]>0).sum(dim=0)>0).float().cpu().numpy() * 255)
    # )
    # cv2.imwrite(
    #    "./data/cache/gen_sdf_sumx.png",
    #    np.uint8(((sdfs[0, 0]>0).sum(dim=2)>0).float().cpu().numpy().transpose() * 255)
    # )

    # metric
    intersection = (sdfs > 0.) & (gt > 0.)
    union = (sdfs > 0.) | (gt > 0.)
    iou = intersection.sum().float() / union.sum().float()
    print (f"iou is {iou}")
    
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Seg2dTopk
from implicit_seg.functional.utils import plot_mask2D

resolutions = [(20+1, 30+1), (40+1, 60+1), (80+1, 120+1), (160+1, 240+1)]
num_points = [None, 21*31, 21*31, 21*31]
clip_mins = [None, -0.4, -0.2, -0.05]
align_corners = False

# resolutions = [28+1, 56+1, 112+1, 224+1, 448+1]
# num_points = [None, 29**2, 29**2, 29**2, 29**2]
# align_corners = False

def query_func(tensor, points):
    """
    [Essential!] here align_corners should be same
    with how you process gt through interpolation.
    tensor: (bz, 1, H, W)
    points: [bz,] list of (N, 2)
    """
    bz = len(points)
    occupancys = [ 
        F.grid_sample(
            tensor[i].unsqueeze(0), 
            points[i].view(1, 1, -1, 2),
            mode="bilinear",
            align_corners=align_corners,
            padding_mode="border", # to align with F.interpolate
        )[0, 0, 0].unsqueeze(0) for i in range(bz)
    ]
    return occupancys

if __name__ == "__main__":    
    # gt
    query_mask = torch.from_numpy(
        cv2.blur(cv2.imread("./data/image.png", cv2.IMREAD_UNCHANGED), (20, 20))[:, :, 3]
    ).unsqueeze(0).unsqueeze(0).to("cuda:0").float() / 255.0
    if type(resolutions[-1]) is int:
        final_W, final_H = resolutions[-1], resolutions[-1]
    else:
        final_W, final_H = resolutions[-1]
    gt = F.interpolate(
        query_mask, (final_H, final_W), mode="bilinear", align_corners=align_corners)
    # plot_mask2D(gt[0, 0].to("cpu"), None, title="gt")

    # infer
    engine = Seg2dTopk(
        query_func = query_func, 
        b_min = [[-1.0, -1.0]],
        b_max = [[1.0, 1.0]],
        resolutions = resolutions,
        num_points = num_points,
        align_corners = align_corners,
        clip_mins = clip_mins, 
        balance_value = 0.5,
        device="cuda:0", 
        visualize=True
    )
    occupancys = engine.forward(tensor=query_mask)
    # cv2.imwrite(
    #    "../data/test2D.png",
    #    np.uint8(occupancys[0, 0].cpu().numpy() * 255)
    # )

    # metric
    intersection = (occupancys > 0.5) & (gt > 0.5)
    union = (occupancys > 0.5) | (gt > 0.5)
    iou = intersection.sum().float() / union.sum().float()
    print (f"iou is {iou}")
    
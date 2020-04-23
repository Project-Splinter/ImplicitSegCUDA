import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import cv2
import argparse

from implicit_seg.functional import Seg2dTopk, Seg2dLossless
from implicit_seg.functional.utils import plot_mask2D

parser = argparse.ArgumentParser(description='.')
parser.add_argument(
    '--mask', type=str, default="./data/image.png")
parser.add_argument(
    '--loop', action="store_true")
parser.add_argument(
    '--vis', action="store_true")
parser.add_argument(
    '--debug', action="store_true")
parser.add_argument(
    '--use_cuda_impl', action="store_true")
parser.add_argument(
    '--mode', type=str, default="lossless", choices=["lossless", "topk"])
args = parser.parse_args()

# configs for reconstruction
resolutions = [(20+1, 30+1), (40+1, 60+1), (80+1, 120+1), (160+1, 240+1)]
align_corners = False
device = "cuda:0"

# query function for points.
def query_func(tensor, points):
    """
    [Essential!] here align_corners should be same
    with how you process gt through interpolation.
    tensor: (bz, 1, H, W)
    points: [bz,] list of (N, 2)
    return: [bz,] list of (1, N)
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

# init implicit seg engine
if args.mode == 'lossless':
    engine = Seg2dLossless(
        query_func=query_func, 
        b_min=[[-1.0, -1.0]],
        b_max=[[1.0, 1.0]],
        resolutions=resolutions,
        align_corners=align_corners,
        balance_value=0.5,
        device=device, 
        visualize=args.vis,
        debug=args.debug,
        use_cuda_impl=args.use_cuda_impl, 
    )

elif args.mode == 'topk':
    num_points = [None, 21*31, 21*31, 21*31]
    clip_mins = [None, -0.4, -0.2, -0.05]

    engine = Seg2dTopk(
        query_func=query_func, 
        b_min=[[-1.0, -1.0]],
        b_max=[[1.0, 1.0]],
        resolutions=resolutions,
        num_points=num_points,
        align_corners=align_corners,
        clip_mins=clip_mins, 
        balance_value=0.5,
        visualize=args.vis,
        device=device, 
        use_cuda_impl=args.use_cuda_impl, 
    )

# gt
query_mask = torch.from_numpy(
    cv2.blur(cv2.imread(args.mask, cv2.IMREAD_UNCHANGED), (20, 20))[:, :, -1]
).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0

# recon
if args.loop:
    engine.visualize = False
    engine.debug = False
    for _ in tqdm.tqdm(range(10000)):
        occupancys = engine.forward(tensor=query_mask)
else:
    occupancys = engine.forward(tensor=query_mask)

# metric
if type(resolutions[-1]) is int:
    final_W, final_H = resolutions[-1], resolutions[-1]
else:
    final_W, final_H = resolutions[-1]
gt = F.interpolate(
    query_mask, (final_H, final_W), mode="bilinear", align_corners=align_corners)
if args.vis or args.debug:
    plot_mask2D(gt[0, 0].to("cpu"), title="gt")

intersection = (occupancys > 0.5) & (gt > 0.5)
union = (occupancys > 0.5) | (gt > 0.5)
iou = intersection.sum().float() / union.sum().float()
print (f"iou is {iou}")
    
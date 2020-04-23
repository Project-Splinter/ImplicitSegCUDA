# ImplicitSeg

A Pytorch Segmentation module through implicit way (support 2d and 3d)

## Install

```
pip install git+https://github.com/liruilong940607/ImplicitSegCUDA --upgrade
```

## Prepare Test data
First `mkdir ./data`, then download `image.png`([here](https://drive.google.com/file/d/1OhpoGcMuk5LVyZjCYd4DXyHg4yYjbbVP/view?usp=sharing)) and `sdf.pth`([here](https://drive.google.com/file/d/1YWoctFOpe8Murzf6TvPMvaa4lcMpt3eM/view?usp=sharing)), and put them under `./data`.

## Usage

```
# 2d
python test/check_seg2d.py --mask ./data/image.png --mode lossless --use_cuda_impl --vis
# 3d
python test/check_seg3d.py --voxel ./data/sdf.pth --mode lossless --use_cuda_impl --vis
```

see also in `test/check_seg3d.py`

```
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import cv2
import argparse

from implicit_seg.functional import Seg3dTopk, Seg3dLossless
from implicit_seg.functional.utils import plot_mask3D

parser = argparse.ArgumentParser(description='.')
parser.add_argument(
    '--voxel', type=str, default="./data/sdf.pth")
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

resolutions = [
    (8+1, 20+1, 8+1),
    (16+1, 40+1, 16+1),
    (32+1, 80+1, 32+1),
    (64+1, 160+1, 64+1),
    (128+1, 320+1, 128+1),
]
align_corners = False
device = "cuda:0"

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

# init implicit seg engine
if args.mode == 'lossless':
    engine = Seg3dLossless(
        query_func=query_func, 
        b_min = [[-1.0, -1.0, -1.0]],
        b_max = [[1.0, 1.0, 1.0]],
        resolutions=resolutions,
        align_corners=align_corners,
        balance_value=0.0, # be careful
        device=device, 
        visualize=args.vis,
        debug=args.debug,
        use_cuda_impl=args.use_cuda_impl, 
    )

elif args.mode == 'topk':
    num_points = [None, 8000, 8000, 8000, 8000]
    clip_mins = [None, -1e9, -1e9, -1e9, -1e9]

    engine = Seg3dTopk(
        query_func=query_func, 
        b_min = [[-1.0, -1.0, -1.0]],
        b_max = [[1.0, 1.0, 1.0]],
        resolutions=resolutions,
        num_points=num_points,
        align_corners=align_corners,
        clip_mins=clip_mins, 
        balance_value=0.0, # be careful
        visualize=args.vis,
        device=device, 
        use_cuda_impl=args.use_cuda_impl, 
    )

# gt
query_sdfs = torch.load(args.voxel).to(device).float() # [1, 1, H, W, D]

# recon
if args.loop:
    engine.visualize = False
    engine.debug = False
    for _ in tqdm.tqdm(range(10000)):
        with torch.no_grad():
            occupancys = engine.forward(tensor=query_sdfs)
else:
    sdfs = engine.forward(tensor=query_sdfs)

# metric
if type(resolutions[-1]) is int:
    final_W, final_H, final_D = resolutions[-1], resolutions[-1], resolutions[-1]
else:
    final_W, final_H, final_D = resolutions[-1]
gt = F.interpolate(
    query_sdfs, (final_D, final_H, final_W), mode="trilinear", align_corners=align_corners)
if args.vis or args.debug:
    plot_mask3D(gt[0, 0].to("cpu"), title="gt")

intersection = (sdfs > 0.) & (gt > 0.)
union = (sdfs > 0.) | (gt > 0.)
iou = intersection.sum().float() / union.sum().float()
print (f"iou is {iou}")

```
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from implicit_seg.functional.recon2d import ImplicitSeg

torch.manual_seed(42)
device = "cuda:0"
data_file = "./data/image.png"

# test data
mask = torch.from_numpy(
    cv2.blur(cv2.imread(data_file, cv2.IMREAD_UNCHANGED), (20, 20))[:, :, 3]
).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
print (f"mask.shape: {mask.shape}; mask.range: {mask.min(), mask.max()}")

# image feature
feature = torch.randn(2, 1024, 256, 256)

# implict network: a MLP
nnet = nn.Sequential(
    nn.Conv2d(1024, 1024, 1, 1),
    nn.Conv2d(1024, 1024, 1, 1),
    nn.Conv2d(1024, 1024, 1, 1),
    nn.Conv2d(1024, 1, 1, 1),
)

# # conv output
# output_conv = nnet(feature)
# print (f"output(conv): {output_conv.shape}")

print (type(nnet))

# def implicit_func(feature):
#     return nnet(feature)

# implicit conv output
nnet_implicit = ImplicitSeg(nnet)
output_implicit = nnet_implicit(feature)
print (f"output(implicit): {output_implicit.shape}")

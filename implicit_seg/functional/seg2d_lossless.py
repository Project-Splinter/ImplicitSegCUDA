import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .utils import (
    create_grid2D,
    build_smooth_conv2D,
    plot_mask2D,
)

class Seg2dLossless(nn.Module):
    def __init__(self, 
                 query_func, b_min, b_max, resolutions,
                 channels=1, balance_value=0.5, device="cuda:0", align_corners=False, 
                 visualize=False, debug=False, use_cuda_impl=True, **kwargs):
        """
        align_corners: same with how you process gt. (grid_sample / interpolate) 
        """
        super().__init__()
        self.query_func = query_func
        self.b_min = torch.tensor(b_min).float().to(device).unsqueeze(1) #[bz, 1, 2]
        self.b_max = torch.tensor(b_max).float().to(device).unsqueeze(1) #[bz, 1, 2]
        if type(resolutions[0]) is int:
            resolutions = torch.tensor([(res, res) for res in resolutions])
        else:
            resolutions = torch.tensor(resolutions)
        self.resolutions = resolutions.to(device)
        self.device = device
        self.batchsize = self.b_min.size(0); assert self.batchsize == 1
        self.balance_value = balance_value
        self.channels = channels; assert self.channels == 1
        self.align_corners = align_corners
        self.visualize = visualize
        self.debug = debug
        self.use_cuda_impl = use_cuda_impl

        for resolution in resolutions:
            assert resolution[0] % 2 == 1 and resolution[1] % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        self.init_coords = create_grid2D(
            0, resolutions[-1]-1, steps=resolutions[0], device=self.device) #[N, 2]
        self.init_coords = self.init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 2]

        # some useful tensors
        self.calculated = torch.zeros((self.resolutions[-1][1],
                                       self.resolutions[-1][0]), 
                                       dtype=torch.bool, device=self.device)

        self.gird8_offsets = torch.stack(torch.meshgrid([
            torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
        ])).int().to(self.device).view(2, -1).t() #[9, 2]

        self.smooth_conv3x3 = build_smooth_conv2D(
            in_channels=1, out_channels=1, kernel_size=3, padding=1).to(self.device)
    
        # cuda impl
        if self.use_cuda_impl:
            from .interp2x_boundary2d import Interp2xBoundary2d
            self.upsampler = Interp2xBoundary2d()

    def batch_eval(self, coords, **kwargs):
        """
        coords: in the coordinates of last resolution
        **kwargs: for query_func
        """
        coords = coords.detach()
        # normalize coords to fit in [b_min, b_max]
        if self.align_corners:
            coords2D = coords.float() / (self.resolutions[-1] - 1)
        else:
            step = 1.0 / self.resolutions[-1].float()
            coords2D = coords.float() / self.resolutions[-1] + step / 2
        coords2D = coords2D * (self.b_max - self.b_min) + self.b_min
        # print(coords2D.shape)
        # query function
        occupancys = self.query_func(**kwargs, points=coords2D)
        if type(occupancys) is list:
            occupancys = torch.stack(occupancys) #[bz, C, N]
        # print (occupancys.shape)
        assert len(occupancys.size()) == 3, \
            "query_func should return a occupancy with shape of [bz, C, N]"
        return occupancys

    def forward(self, **kwargs):
        """
        output occupancy field would be:
        (bz, C, res, res)
        """
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]

        calculated = self.calculated.clone()
        
        for resolution in self.resolutions:
            W, H = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)
            
            # first step
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone() # torch.long 
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, H, W)

                if self.visualize:
                    self.plot(occupancys, coords, final_H, final_W)
                
                coords_accum = coords / stride
                calculated[coords[0, :, 1], coords[0, :, 0]] = True

            else:
                coords_accum *= 2

                if self.use_cuda_impl:
                    occupancys, is_boundary = self.upsampler(occupancys)

                else:
                    # here true is correct!
                    valid = F.interpolate(
                        (occupancys>0.5).float(), 
                        size=(H, W), mode="bilinear", align_corners=True)

                    # here true is correct!
                    occupancys = F.interpolate(
                        occupancys.float(), 
                        size=(H, W), mode="bilinear", align_corners=True)

                    is_boundary = (valid > 0.0) & (valid < 1.0)

                is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]
                is_boundary[coords_accum[0, :, 1], 
                            coords_accum[0, :, 0]] = False
                
                point_coords = is_boundary.permute(1, 0).nonzero().unsqueeze(0)
                point_indices = point_coords[:, :, 1] * W + point_coords[:, :, 0]

                R, C, H, W = occupancys.shape
                # interpolated value
                occupancys_interp = torch.gather(
                    occupancys.reshape(R, C, H * W), 2, point_indices.unsqueeze(1))

                # inferred value
                coords = point_coords * stride
                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)
                
                # put mask point predictions to the right places on the upsampled grid.
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, H * W)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, H, W)
                )

                # conflicts
                conflicts = (
                    (occupancys_interp - self.balance_value) *
                    (occupancys_topk - self.balance_value) < 0
                )[0, 0]
                if self.visualize:
                    self.plot(occupancys, coords, final_H, final_W)

                voxels = coords / stride
                coords_accum = torch.cat([
                    voxels, 
                    coords_accum
                ], dim=1).unique(dim=1)
                calculated[coords[0, :, 1], coords[0, :, 0]] = True

                while conflicts.sum() > 0:
                    conflicts_coords = coords[0, conflicts, :]
                    if self.debug:
                        self.plot(
                            occupancys, conflicts_coords.unsqueeze(0), 
                            final_H, final_W, title="conflicts")

                    conflicts_boundary = (
                        conflicts_coords.int() +
                        self.gird8_offsets.unsqueeze(1) * stride.int()
                    ).reshape(-1, 2).long().unique(dim=0)
                    conflicts_boundary[:, 0] = (
                        conflicts_boundary[:, 0].clamp(0, calculated.size(1) - 1))
                    conflicts_boundary[:, 1] = (
                        conflicts_boundary[:, 1].clamp(0, calculated.size(0) - 1))
                    
                    coords = conflicts_boundary[
                        calculated[conflicts_boundary[:, 1], conflicts_boundary[:, 0]] == False
                    ]
                    if self.debug:
                        self.plot(
                            occupancys, coords.unsqueeze(0), 
                            final_H, final_W, title="coords")

                    coords = coords.unsqueeze(0)
                    point_coords = coords / stride
                    point_indices = point_coords[:, :, 1] * W + point_coords[:, :, 0]
                    
                    R, C, H, W = occupancys.shape
                    # interpolated value
                    occupancys_interp = torch.gather(
                        occupancys.reshape(R, C, H * W), 2, point_indices.unsqueeze(1))

                    # inferred value
                    coords = point_coords * stride
                    if coords.size(1) == 0:
                        continue
                    occupancys_topk = self.batch_eval(coords, **kwargs)
                    
                    # conflicts
                    conflicts = (
                        (occupancys_interp - self.balance_value) *
                        (occupancys_topk - self.balance_value) < 0
                    )[0, 0]
                    
                    # put mask point predictions to the right places on the upsampled grid.
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    occupancys = (
                        occupancys.reshape(R, C, H * W)
                        .scatter_(2, point_indices, occupancys_topk)
                        .view(R, C, H, W)
                    )

                    voxels = coords / stride
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=1).unique(dim=1)
                    calculated[coords[0, :, 1], coords[0, :, 0]] = True

        return occupancys

    def plot(self, occupancys, coords, final_H, final_W, title='', **kwargs):
        final = F.interpolate(
            occupancys.float(), size=(final_H, final_W), 
            mode="bilinear", align_corners=True) # here true is correct!
        x = coords[0, :, 0].to("cpu")
        y = coords[0, :, 1].to("cpu")
        plot_mask2D(
            final[0, 0].to("cpu"), title, (x, y), **kwargs)
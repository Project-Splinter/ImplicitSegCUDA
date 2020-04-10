import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .utils import (
    create_grid3D,
    build_smooth_conv3D,
    plot_mask3D,
)

class Seg3dLossless(nn.Module):
    def __init__(self, 
                 query_func, b_min, b_max, resolutions,
                 channels=1, balance_value=0.5, device="cuda:0", align_corners=False, 
                 visualize=False):
        """
        align_corners: same with how you process gt. (grid_sample / interpolate) 
        """
        super().__init__()
        self.query_func = query_func
        self.b_min = torch.tensor(b_min).float().to(device).unsqueeze(1) #[bz, 1, 3]
        self.b_max = torch.tensor(b_max).float().to(device).unsqueeze(1) #[bz, 1, 3]
        if type(resolutions[0]) is int:
            resolutions = torch.tensor([(res, res, res) for res in resolutions])
        else:
            resolutions = torch.tensor(resolutions)
        self.resolutions = resolutions.to(device)
        self.device = device
        self.batchsize = self.b_min.size(0); assert self.batchsize == 1
        self.balance_value = balance_value
        self.channels = channels; assert self.channels == 1
        self.align_corners = align_corners
        self.visualize = visualize

        for resolution in resolutions:
            assert resolution[0] % 2 == 1 and resolution[1] % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        self.init_coords = create_grid3D(
            0, resolutions[-1]-1, steps=resolutions[0], device=self.device) #[N, 3]
        self.init_coords = self.init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 3]

        # some useful tensors
        self.calculated = torch.zeros((self.resolutions[-1][2],
                                       self.resolutions[-1][1],
                                       self.resolutions[-1][0]), 
                                       dtype=torch.bool, device=self.device)

        self.gird8_offsets = torch.stack(torch.meshgrid([
            torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
        ])).int().to(self.device).view(3, -1).t() #[27, 3]

        self.smooth_conv3x3 = build_smooth_conv3D(
            in_channels=1, out_channels=1, kernel_size=3, padding=1).to(self.device)


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
        # query function
        occupancys = self.query_func(**kwargs, points=coords2D)
        if type(occupancys) is list:
            occupancys = torch.stack(occupancys) #[bz, C, N]
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
        final_D = self.resolutions[-1][2]

        calculated = self.calculated.clone()
        
        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)
            
            # first step
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone() # torch.long 
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)

                if self.visualize:
                    final = F.interpolate(
                        occupancys.float(), size=(final_D, final_H, final_W), 
                        mode="trilinear", align_corners=True) # here true is correct!
                    x = coords[0, :, 0].to("cpu")
                    y = coords[0, :, 1].to("cpu")
                    z = coords[0, :, 2].to("cpu")
                    
                    plot_mask3D(
                        final[0, 0].to("cpu"), point_coords=(x, y, z))
                
                with torch.no_grad():
                    coords_accum = coords / stride
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

            else:
                with torch.no_grad():
                    # here true is correct!
                    valid = F.interpolate(
                        (occupancys>0.5).float(), 
                        size=(D, H, W), mode="trilinear", align_corners=True)

                # here true is correct!
                occupancys = F.interpolate(
                    occupancys.float(), 
                    size=(D, H, W), mode="trilinear", align_corners=True)
                
                with torch.no_grad():
                    coords_accum *= 2

                    is_boundary = (valid > 0.0) & (valid < 1.0)
                    is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]
                    is_boundary[coords_accum[0, :, 2],
                                coords_accum[0, :, 1], 
                                coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero().unsqueeze(0)
                    point_indices = (
                        point_coords[:, :, 2] * H * W + 
                        point_coords[:, :, 1] * W + 
                        point_coords[:, :, 0])

                    R, C, D, H, W = occupancys.shape
                    # interpolated value
                    occupancys_interp = torch.gather(
                        occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1))

                    # inferred value
                    coords = point_coords * stride

                occupancys_topk = self.batch_eval(coords, **kwargs)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, D * H * W)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, D, H, W)
                )

                with torch.no_grad():
                    # conflicts
                    conflicts = (
                        (occupancys_interp - self.balance_value) *
                        (occupancys_topk - self.balance_value) < 0
                    )[0, 0]

                    # if self.visualize:
                    #     final = F.interpolate(
                    #         occupancys.float(), size=(final_D, final_H, final_W), 
                    #         mode="trilinear", align_corners=True) # here true is correct!
                    #     x = coords[0, :, 0].to("cpu")
                    #     y = coords[0, :, 1].to("cpu")
                    #     z = coords[0, :, 2].to("cpu")
                        
                    #     plot_mask3D(
                    #         final[0, 0].to("cpu"), point_coords=(x, y, z))

                    voxels = coords / stride
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=1).unique(dim=1)
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

                while conflicts.sum() > 0:
                    with torch.no_grad():
                        conflicts_coords = coords[0, conflicts, :]

                        # if self.visualize:
                        #     final = F.interpolate(
                        #         occupancys.float(), size=(final_D, final_H, final_W), 
                        #         mode="trilinear", align_corners=True) # here true is correct!
                        #     x = conflicts_coords[:, 0].to("cpu")
                        #     y = conflicts_coords[:, 1].to("cpu")
                        #     z = conflicts_coords[:, 2].to("cpu")
                            
                        #     plot_mask3D(
                        #         final[0, 0].to("cpu"), point_coords=(x, y, z), title="conflicts")
                        
                        conflicts_boundary = (
                            conflicts_coords.int() +
                            self.gird8_offsets.unsqueeze(1) * stride.int()
                        ).reshape(-1, 3).long().unique(dim=0)
                        conflicts_boundary[:, 0] = (
                            conflicts_boundary[:, 0].clamp(0, calculated.size(2) - 1))
                        conflicts_boundary[:, 1] = (
                            conflicts_boundary[:, 1].clamp(0, calculated.size(1) - 1))
                        conflicts_boundary[:, 2] = (
                            conflicts_boundary[:, 2].clamp(0, calculated.size(0) - 1))

                        coords = conflicts_boundary[
                            calculated[conflicts_boundary[:, 2], 
                                    conflicts_boundary[:, 1], 
                                    conflicts_boundary[:, 0]] == False
                        ]

                        # if self.visualize:
                        #     final = F.interpolate(
                        #         occupancys.float(), size=(final_D, final_H, final_W), 
                        #         mode="trilinear", align_corners=True) # here true is correct!
                        #     x = coords[:, 0].to("cpu")
                        #     y = coords[:, 1].to("cpu")
                        #     z = coords[:, 2].to("cpu")
                            
                        #     plot_mask3D(
                        #         final[0, 0].to("cpu"), point_coords=(x, y, z), title="coords")

                        coords = coords.unsqueeze(0)
                        point_coords = coords / stride
                        point_indices = (
                            point_coords[:, :, 2] * H * W + 
                            point_coords[:, :, 1] * W + 
                            point_coords[:, :, 0])
                        
                        R, C, D, H, W = occupancys.shape
                        # interpolated value
                        occupancys_interp = torch.gather(
                            occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1))

                        # inferred value
                        coords = point_coords * stride

                    occupancys_topk = self.batch_eval(coords, **kwargs)
                    
                    with torch.no_grad():
                        # conflicts
                        conflicts = (
                            (occupancys_interp - self.balance_value) *
                            (occupancys_topk - self.balance_value) < 0
                        )[0, 0]
                    
                    # put mask point predictions to the right places on the upsampled grid.
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    occupancys = (
                        occupancys.reshape(R, C, D * H * W)
                        .scatter_(2, point_indices, occupancys_topk)
                        .view(R, C, D, H, W)
                    )

                    with torch.no_grad():
                        voxels = coords / stride
                        coords_accum = torch.cat([
                            voxels, 
                            coords_accum
                        ], dim=1).unique(dim=1)
                        calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

        return occupancys

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .utils import (
    create_grid2D,
    calculate_uncertainty,
    get_uncertain_point_coords_on_grid2D_faster as get_uncertain_point_coords_on_grid2D,
    plot_mask2D,
)

class Seg2dTopk(nn.Module):
    def __init__(self, 
                 query_func, b_min, b_max, resolutions, num_points, clip_mins,
                 channels=1, balance_value=0.5, device="cuda:0", align_corners=False, 
                 visualize=False):
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
        self.num_points = num_points
        self.clip_mins = clip_mins
        self.device = device
        self.batchsize = self.b_min.size(0)
        self.balance_value = balance_value
        self.channels = channels; assert channels == 1
        self.align_corners = align_corners
        self.visualize = visualize

        for resolution in resolutions:
            assert resolution[0] % 2 == 1 and resolution[1] % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        self.init_coords = create_grid2D(
            0, resolutions[-1]-1, steps=resolutions[0], device=self.device) #[N, 2]
        self.init_coords = self.init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 2]

    def batch_eval(self, coords, **kwargs):
        """
        coords: in the coordinates of last resolution
        **kwargs: for query_func
        """
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
        
        for resolution, num_pt, clip_min in zip(self.resolutions, self.num_points, self.clip_mins):
            W, H = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)
            
            # first step
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone() # torch.long 
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, H, W)

                if self.visualize:
                    final = F.interpolate(
                        occupancys.float(), size=(final_H, final_W), 
                        mode="bilinear", align_corners=True) # here true is correct!
                    x = coords[0, :, 0].to("cpu")
                    y = coords[0, :, 1].to("cpu")
                    plot_mask2D(
                        final[0, 0].to("cpu"), point_coords=(x, y))

            else:
                # here true is correct!
                occupancys = F.interpolate(
                    occupancys.float(), size=(H, W), mode="bilinear", align_corners=True)
                
                if not num_pt > 0:
                    continue

                uncertainty = calculate_uncertainty(occupancys, balance_value=self.balance_value)
                point_indices, point_coords = get_uncertain_point_coords_on_grid2D(
                    uncertainty, num_points=num_pt, clip_min=clip_min)
                
                coords = point_coords * stride
                occupancys_topk = self.batch_eval(coords, **kwargs)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, H * W)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, H, W)
                )

                if self.visualize:
                    final = F.interpolate(
                        occupancys.float(), size=(final_H, final_W), 
                        mode="bilinear", align_corners=True) # here true is correct!
                    x = coords[0, :, 0].to("cpu")
                    y = coords[0, :, 1].to("cpu")
                    plot_mask2D(
                        final[0, 0].to("cpu"), point_coords=(x, y))

        return occupancys

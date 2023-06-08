import os
import numpy as np
import torch
import torch.nn.functional as F

import config


class CanoBlendWeightVolume:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            print('# CanoBlendWeightVolume is not found from %s' % data_path)
            return
        data = np.load(data_path)

        base_weight_volume = data['weight_volume']
        base_weight_volume = base_weight_volume.transpose((3, 0, 1, 2))[None]
        self.base_weight_volume = torch.from_numpy(base_weight_volume).to(torch.float32).to(config.device)

        smpl_sdf_volume = data['sdf_volume']
        if len(smpl_sdf_volume.shape) == 3:
            smpl_sdf_volume = smpl_sdf_volume[..., None]
        smpl_sdf_volume = smpl_sdf_volume.transpose((3, 0, 1, 2))[None]
        self.smpl_sdf_volume = torch.from_numpy(smpl_sdf_volume).to(torch.float32).to(config.device)

        self.bounds = torch.from_numpy(data['bounds']).to(torch.float32).to(config.device)
        self.center = torch.from_numpy(data['center']).to(torch.float32).to(config.device)

    def forward_weight(self, pts, requires_scale = True):
        """
        :param pts: (B, N, 3)
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 24)
        """
        if requires_scale:
            pts = (pts - self.bounds[None, None, 0]) / (self.bounds[1] - self.bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        base_w = F.grid_sample(self.base_weight_volume,
                               grid,
                               padding_mode = 'border',
                               align_corners = True)
        base_w = base_w[0, :, :, 0, 0].reshape(self.base_weight_volume.shape[1], B, N).permute((1, 2, 0))
        return base_w

    def forward_sdf(self, pts, requires_scale = True):
        if requires_scale:
            pts = (pts - self.bounds[None, None, 0]) / (self.bounds[1] - self.bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        sdf = F.grid_sample(self.smpl_sdf_volume,
                            grid,
                            padding_mode = 'border',
                            align_corners = True)
        sdf = sdf[0, :, :, 0, 0].reshape(-1, B, N).permute((1, 2, 0))

        return sdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.feature2d import grid_sample


class PoseVocab(nn.Module):
    def __init__(self,
                 joint_num,
                 point_num,
                 line_size,
                 feat_dim,
                 spacial_bounds,
                 pose_points_path,
                 pose_format = 'quaternion',
                 knn = 10,
                 used_point_num = None,
                 init = 'random'):
        """
        :param point_num: P, discrete pose point number
        :param line_size: (Lx, Ly, Lz), spacial resolutions
        :param feat_dim: C, feature channel number
        :param spacial_bounds: [min_xyz, max_xyz] spacial bounds along x, y, z axes
        """
        super(PoseVocab, self).__init__()

        if isinstance(spacial_bounds, np.ndarray):
            spacial_bounds = torch.from_numpy(spacial_bounds).to(torch.float32)
        elif isinstance(spacial_bounds, torch.Tensor):
            spacial_bounds = spacial_bounds.to(torch.float32)
        elif isinstance(spacial_bounds, list):
            spacial_bounds = torch.from_numpy(np.array(spacial_bounds)).to(torch.float32)
        else:
            raise TypeError('Invalid bounds')
        self.pose_format = pose_format
        self.J = joint_num
        self.P = point_num
        self.Lx, self.Ly, self.Lz = line_size
        self.C = feat_dim
        self.out_channels = self.J * self.C * 3
        self.knn = knn

        feat_lines_x = torch.zeros((self.J, self.P, self.Lx, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_x', nn.Parameter(feat_lines_x))
        feat_lines_y = torch.zeros((self.J, self.P, self.Ly, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_y', nn.Parameter(feat_lines_y))
        feat_lines_z = torch.zeros((self.J, self.P, self.Lz, self.C), dtype = torch.float32)
        self.register_parameter('feat_lines_z', nn.Parameter(feat_lines_z))

        if init == 'random':
            nn.init.uniform_(self.feat_lines_x.data, -1e-2, 1e-2)
            nn.init.uniform_(self.feat_lines_y.data, -1e-2, 1e-2)
            nn.init.uniform_(self.feat_lines_z.data, -1e-2, 1e-2)
        elif init == 'zeros':
            nn.init.constant_(self.feat_lines_x.data, 0.)
            nn.init.constant_(self.feat_lines_y.data, 0.)
            nn.init.constant_(self.feat_lines_z.data, 0.)
        else:
            raise ValueError('Invalid init method')

        pose_points = torch.from_numpy(np.load(pose_points_path)).to(torch.float32)
        if used_point_num is not None:
            pose_points = pose_points[:, :used_point_num]
        else:
            pose_points = pose_points[:, :point_num]
        self.register_buffer('pose_points', pose_points)  # (J, P, 3) or (J, P, 4)
        self.register_buffer('spacial_bounds', spacial_bounds)

        self.pose_point_graph = None

    def sample(self, feat_lines_x, x, top_k_ids, top_k_weights):
        """
        :param feat_lines_x: (J, P, Lx, C)
        :param x: (B, N, 1)
        :param top_k_ids: (B, J, K)
        :param top_k_weights: (B, J, K)
        :return: (B, N, J, C)
        """
        J, P, Lx, C = feat_lines_x.shape
        B, N = x.shape[:2]
        K = top_k_ids.shape[2]
        feat_lines = torch.gather(feat_lines_x[None].expand(B, -1, -1, -1, -1),
                                  2,
                                  top_k_ids[..., None, None].expand(-1, -1, -1, Lx, C))  # (B, J, K, Lx, C)
        feat_lines = (top_k_weights[..., None, None] * feat_lines).sum(2)  # (B, J, Lx, C)
        if self.training:
            smooth_loss = self.smooth_loss(feat_lines)
        else:
            smooth_loss = None
        feat_lines = feat_lines.permute(0, 1, 3, 2).reshape(B, J * C, Lx, 1)
        feat = grid_sample(feat_lines,
                           torch.cat([torch.zeros_like(x), x], -1))
        feat = feat.view(B, N, J, C)
        return feat, smooth_loss

    def forward(self, query_points, query_poses):
        """
        :param query_points: (B, N, 3)
        :param query_poses: (B, J, 4)
        :return: feat: (B, N, J, 3, C)
        """

        # normalize query points to [-1, 1]
        query_points = (query_points - self.spacial_bounds[None, None, 0]) / (self.spacial_bounds[1] - self.spacial_bounds[0])[None, None]
        query_points = 2. * query_points - 1.

        with torch.no_grad():
            pose_dist = torch.abs((query_poses[:, :, None, :] * self.pose_points[None]).sum(-1))  # (B, J, P)
            k = self.knn
            topk_weight, topk_id = torch.topk(pose_dist, k, dim = 2)  # (B, J, K)
            topk_weight = F.normalize(topk_weight, dim = 2, p = 1, eps = 1e-16)

        feat_x, smooth_loss_x = self.sample(self.feat_lines_x, query_points[..., 0].unsqueeze(-1), topk_id, topk_weight)  # (B, N, J, C)
        feat_y, smooth_loss_y = self.sample(self.feat_lines_y, query_points[..., 1].unsqueeze(-1), topk_id, topk_weight)
        feat_z, smooth_loss_z = self.sample(self.feat_lines_z, query_points[..., 2].unsqueeze(-1), topk_id, topk_weight)

        if self.training:
            xyz_smooth_loss = smooth_loss_x + smooth_loss_y + smooth_loss_z
        else:
            xyz_smooth_loss = None

        return torch.stack([feat_x, feat_y, feat_z], dim = 3), xyz_smooth_loss

    @staticmethod
    def smooth_loss(feat_lines):
        """
        :param feat_lines: (B, J, K, Lx, C)
        :return:
        """
        tv = torch.square(feat_lines[..., 1:, :] - feat_lines[..., :-1, :]).mean()
        return tv

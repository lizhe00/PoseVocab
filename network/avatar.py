import torch
import torch.nn as nn
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms

import config
from network.mlp import MLPLinear, SdfMLP
from network.density import LaplaceDensity
from network.volume import CanoBlendWeightVolume
from network.posevocab import PoseVocab
from utils.embedder import get_embedder
import utils.nerf_util as nerf_util
import utils.smpl_util as smpl_util
from utils.posevocab_custom_ops.near_far_smpl import near_far_smpl


class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt

        if opt['local_pose']:
            self.atten_map = torch.from_numpy(np.loadtxt(config.PROJ_DIR + '/configs/pose_map/pose_map_scanimate.txt').transpose()).to(torch.float32).to(config.device)

        self.pos_embedder, self.pos_dim = get_embedder(opt['multires'], 3)

        # canonical blend weight volume
        self.cano_weight_volume = CanoBlendWeightVolume(config.opt['train']['data']['data_dir'] + '/cano_weight_volume.npz')

        """ PoseVocab """
        multiscale_line_sizes = self.opt['multiscale_line_sizes']
        point_nums = self.opt['point_nums']
        if 'used_point_nums' in self.opt:
            used_point_nums = self.opt['used_point_nums']
        else:
            used_point_nums = [None for point_num in point_nums]
        feat_dims = self.opt['feat_dims']
        knns = self.opt['knns']
        pose_formats = self.opt['pose_formats']
        self.pose_feat_dim = 0
        self.joint_feat_lines = nn.ModuleList()
        for line_size, point_num, used_point_num, feat_dim, knn, pose_format \
                in zip(multiscale_line_sizes, point_nums, used_point_nums, feat_dims, knns, pose_formats):
            feat_lines = PoseVocab(joint_num = 21,
                                   point_num = point_num,
                                   line_size = line_size,
                                   feat_dim = feat_dim,
                                   spacial_bounds = self.cano_weight_volume.bounds,
                                   pose_points_path = config.opt['train']['data']['data_dir'] + '/key_rotations/sorted_rotations.npy',
                                   pose_format = pose_format,
                                   knn = knn)
            self.pose_feat_dim += feat_lines.out_channels
            self.joint_feat_lines.append(feat_lines)
        if self.opt['concat_pose_vec']:
            self.pose_feat_dim += 21 * 4

        """ geometry networks """
        geo_mlp_opt = {
            'in_channels': self.pos_dim + self.pose_feat_dim,
            'out_channels': 256 + 1,
            'inter_channels': [512, 256, 256, 256, 256, 256],
            'nlactv': nn.Softplus(beta = 100),
            'res_layers': [4],
            'geometric_init': True,
            'bias': 0.7,
            'weight_norm': True
        }
        self.geo_mlp = SdfMLP(**geo_mlp_opt)

        """ texture networks """
        if self.opt['use_viewdir']:
            self.viewdir_embedder, self.viewdir_dim = get_embedder(self.opt['multires_viewdir'], 3)
        else:
            self.viewdir_embedder, self.viewdir_dim = None, 0
        tex_mlp_opt = {
            'in_channels': 256 + self.viewdir_dim,
            'out_channels': 3,
            'inter_channels': [256, 256, 256],
            'nlactv': nn.ReLU(),
            'last_op': nn.Sigmoid()
        }
        self.tex_mlp = MLPLinear(**tex_mlp_opt)

        print('# MLPs: ')
        print(self.geo_mlp)
        print(self.tex_mlp)

        # sdf2density
        self.density_func = LaplaceDensity(params_init = {'beta': 0.01})

    def forward_cano_radiance_field(self, xyz, viewdirs, pose, compute_grad = False):
        """
        :param xyz: (B, N, 3)
        :param viewdirs: (B, N, 3)
        :param pose: (B, pose_dim)
        :param compute_grad: whether computing gradient w.r.t xyz
        :return:
        """
        if compute_grad:
            xyz.requires_grad_()
        batch_size, point_num = xyz.shape[:2]
        pose = pose.reshape(batch_size, -1, 3)  # axis angle in default
        pose_quat = pytorch3d.transforms.axis_angle_to_quaternion(pose)

        if self.opt['local_pose']:
            with torch.no_grad():
                lbs_ori = self.cano_weight_volume.forward_weight(xyz, requires_scale = True)[..., :24]
                lbs_ori[..., 22:] = 0.
                lbs = torch.einsum('ij,bnj->bni', self.atten_map, lbs_ori)
                lbs = lbs[..., :-2]

        xyz_ = self.pos_embedder(xyz)
        pose_feat = []
        xyz_smooth_loss = []
        for idx in range(len(self.joint_feat_lines)):
            query_pose = pose_quat if self.joint_feat_lines[idx].pose_format == 'quaternion' else pose
            if self.training:
                pose_feat_, xyz_smooth_loss_ = self.joint_feat_lines[idx].forward(xyz.detach(), query_pose)
            else:
                query_frame = query_pose.shape[1] // 21
                pose_feat_ = []
                for i in range(query_frame):
                    pose_feat_frame = self.joint_feat_lines[idx](xyz.detach(), query_pose[:, i*21: (i+1)*21])[0]
                    pose_feat_.append(pose_feat_frame)
                pose_feat_ = torch.stack(pose_feat_, 0).mean(0)
                xyz_smooth_loss_ = None
            pose_feat.append(pose_feat_.view(batch_size, point_num, 21, -1))
            xyz_smooth_loss.append(xyz_smooth_loss_)

        if self.opt['concat_pose_vec']:
            if self.training:
                pose_feat.append(pose_quat[:, None, :, :].expand(-1, point_num, -1, -1))
            else:
                pose_feat.append(pose_quat[:, None, :, :].expand(-1, point_num, -1, -1).view(batch_size, point_num, -1, 21, 4).mean(2))
        pose_feat = torch.cat(pose_feat, -1)  # (B, N, J, C)

        if self.opt['local_pose']:
            pose_feat = (pose_feat * lbs[..., None])
        pose_feat = pose_feat.view(batch_size, point_num, -1)
        geo_feat = torch.cat([xyz_, pose_feat], -1)
        geo_feat = self.geo_mlp(geo_feat)
        sdf, geo_feat = torch.split(geo_feat, [1, geo_feat.shape[-1] - 1], -1)

        if self.viewdir_embedder is not None:
            if viewdirs is None:
                viewdirs = torch.zeros_like(xyz)
            geo_feat = torch.cat([geo_feat, self.viewdir_embedder(viewdirs)], -1)
        color = self.tex_mlp(geo_feat)

        density = self.density_func(sdf)

        ret = {
            'sdf': -sdf,  # assume outside is negative, inside is positive
            'density': density,
            'color': color
        }

        if xyz_smooth_loss[0] is not None:
            xyz_smooth_loss = torch.stack(xyz_smooth_loss, 0)
            ret.update({
                'tv_loss': xyz_smooth_loss
            })

        if compute_grad:
            d_output = torch.ones_like(sdf, requires_grad = False, device = sdf.device)
            normal = torch.autograd.grad(outputs = sdf,
                                         inputs = xyz,
                                         grad_outputs = d_output,
                                         create_graph = self.training,
                                         retain_graph = self.training,
                                         only_inputs = True)[0]
            ret.update({
                'normal': normal
            })
        return ret

    def transform_live2cano(self, posed_pts, batch, normals = None, near_thres = 0.08):
        """ live_pts -> cano_pts """
        with torch.no_grad():
            pts_w, near_flag = smpl_util.calc_blending_weight(posed_pts, batch['live_smpl_v'], batch['smpl_faces'], near_thres)

            live2cano_jnt_mats = torch.linalg.inv(batch['cano2live_jnt_mats'])  # (B, J, 4, 4)
            cano_pts = smpl_util.skinning(posed_pts, pts_w, live2cano_jnt_mats)

            if normals is not None:
                cano_normals = smpl_util.skinning_normal(normals, pts_w, live2cano_jnt_mats)
        if normals is None:
            return cano_pts, near_flag
        else:
            return cano_pts, cano_normals, near_flag

    def render(self, batch, chunk_size = 2048, depth_guided_sampling = None):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        batch_size, n_pixels = ray_o.shape[:2]

        if depth_guided_sampling['flag']:
            # update near and far by dist
            valid_dist_flag = batch['dist'] > 1e-6
            dist = batch['dist'][valid_dist_flag]
            near_dist = depth_guided_sampling['near_sur_dist']
            far_dist = depth_guided_sampling['near_sur_dist']
            near[valid_dist_flag] = dist - near_dist
            far[valid_dist_flag] = dist + far_dist
            N_ray_samples = depth_guided_sampling['N_ray_samples']
        else:
            valid_dist_flag = torch.ones_like(near, dtype = bool)
            near, far = [], []
            for b in range(batch_size):
                near_, far_ = near_far_smpl(batch['live_smpl_v'][b], ray_o[b], ray_d[b])
                near.append(near_)
                far.append(far_)
            near = torch.stack(near, 0)
            far = torch.stack(far, 0)
            N_ray_samples = 64

        output_list = []
        for i in range(0, n_pixels, chunk_size):
            near_chunk = near[:, i: i + chunk_size]
            far_chunk = far[:, i: i + chunk_size]
            ray_o_chunk = ray_o[:, i: i + chunk_size]
            ray_d_chunk = ray_d[:, i: i + chunk_size]
            valid_dist_flag_chunk = valid_dist_flag[:, i: i + chunk_size]

            # sample points on each ray
            pts, z_vals = nerf_util.sample_pts_on_rays(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk,
                                                       N_samples = N_ray_samples,
                                                       perturb = self.training,
                                                       depth_guided_mask = valid_dist_flag_chunk)

            # flat
            _, n_pixels_chunk, n_samples = pts.shape[:3]
            pts = pts.view(batch_size, n_pixels_chunk * n_samples, -1)
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, dists[..., -1:]], -1)

            # query
            cano_pts, near_flag = self.transform_live2cano(pts, batch)
            viewdirs = ray_d_chunk / torch.norm(ray_d_chunk, dim = -1, keepdim = True)
            viewdirs = viewdirs[:, :, None, :].expand(-1, -1, n_samples, -1).reshape(batch_size, n_pixels_chunk * n_samples, -1)
            # apply gaussian noise to avoid overfitting
            if self.training:
                with torch.no_grad():
                    noise = torch.randn_like(viewdirs) * 0.1
                viewdirs = viewdirs + noise
                viewdirs = viewdirs / torch.norm(viewdirs, dim = -1, keepdim = True)
            ret = self.forward_cano_radiance_field(cano_pts, viewdirs, batch['pose'], config.opt['train']['compute_grad'] and self.training)
            ret['color'] = ret['color'].view(batch_size, n_pixels_chunk, n_samples, -1)
            ret['density'] = ret['density'].view(batch_size, n_pixels_chunk, n_samples, -1)

            # integration
            alpha = 1. - torch.exp(-ret['density'] * dists[..., None])
            raw = torch.cat([ret['color'], alpha], dim = -1)
            rgb_map, disp_map, acc_map, weights, depth_map = nerf_util.raw2outputs(raw, z_vals, white_bkgd = config.white_bkgd if not self.training else False)

            output_chunk = {
                'rgb_map': rgb_map,  # (batch_size, n_pixel_chunk, 3)
                'acc_map': acc_map
            }
            if 'normal' in ret:
                output_chunk.update({
                    'normal': ret['normal'].view(batch_size, n_pixels_chunk, -1, 3)
                })
            if 'tv_loss' in ret:
                output_chunk.update({
                    'tv_loss': ret['tv_loss'].view(1, 1, -1)
                })
            output_list.append(output_chunk)

        keys = output_list[0].keys()
        output_list = {k: torch.cat([r[k] for r in output_list], dim = 1) for k in keys}

        # processing for patch-based ray sampling
        if 'mask_within_patch' in batch:
            _, ray_num = batch['mask_within_patch'].shape
            rgb_map = torch.zeros((batch_size, ray_num, 3), dtype = torch.float32, device = config.device)
            acc_map = torch.zeros((batch_size, ray_num), dtype = torch.float32, device = config.device)
            rgb_map[batch['mask_within_patch']] = output_list['rgb_map'].reshape(-1, 3)
            acc_map[batch['mask_within_patch']] = output_list['acc_map'].reshape(-1)
            batch['color_gt'][~batch['mask_within_patch']] = 0.
            batch['mask_gt'][~batch['mask_within_patch']] = 0.
            output_list['rgb_map'] = rgb_map
            output_list['acc_map'] = acc_map

        return output_list

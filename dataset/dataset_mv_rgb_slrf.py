import glob
import os
import pickle
import time

import numpy as np
import cv2 as cv
import torch
import trimesh
from torch.utils.data import Dataset
import yaml
import json
import smplx

import utils.smpl_util as smpl_util
import utils.nerf_util as nerf_util
import utils.visualize_util as visualize_util
import config


class MvRgbDataset(Dataset):
    @torch.no_grad()
    def __init__(self, data_dir, frame_range = None, training = True,
                 used_cam_ids = None,
                 load_smpl_pos_map = False,
                 load_smpl_nml_map = False,
                 frame_win = 0,
                 subject_name = None,
                 fix_head_pose = False,
                 fix_hand_pose = False):
        super(MvRgbDataset, self).__init__()

        self.data_dir = data_dir
        self.training = training
        self.subject_name = subject_name
        if self.subject_name is None:
            self.subject_name = os.path.basename(self.data_dir)

        self.load_smpl_pos_map = load_smpl_pos_map
        self.load_smpl_nml_map = load_smpl_nml_map

        cam_data = json.load(open(self.data_dir + '/calibration.json', 'r'))
        self.view_num = len(cam_data)
        self.extr_mats = []
        for view_idx in range(self.view_num):
            extr_mat = np.identity(4, np.float32)
            extr_mat[:3, :3] = np.array(cam_data['cam%02d' % view_idx]['R'], np.float32).reshape(3, 3)
            extr_mat[:3, 3] = np.array(cam_data['cam%02d' % view_idx]['T'], np.float32)
            self.extr_mats.append(extr_mat)
        self.intr_mats = [np.array(cam_data['cam%02d' % view_idx]['K'], np.float32).reshape(3, 3) for view_idx in range(self.view_num)]
        self.img_heights = [cam_data['cam%02d' % view_idx]['imgSize'][1] for view_idx in range(self.view_num)]
        self.img_widths = [cam_data['cam%02d' % view_idx]['imgSize'][0] for view_idx in range(self.view_num)]
        self.gender = 'neutral'

        smpl_data = np.load(self.data_dir + '/smpl_params.npz', allow_pickle = True)
        smpl_data = dict(smpl_data)
        self.smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}
        self.smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = self.gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)

        self.fix_head_pose = fix_head_pose
        self.fix_hand_pose = fix_hand_pose

        pose_list = list(range(self.smpl_data['body_pose'].shape[0]))
        if frame_range is not None:
            if isinstance(frame_range, list):
                if len(frame_range) == 2:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]})')
                    frame_range = range(frame_range[0], frame_range[1])
                elif len(frame_range) == 3:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
                    frame_range = range(frame_range[0], frame_range[1], frame_range[2])
            elif isinstance(frame_range, str):
                frame_range = np.loadtxt(self.data_dir + '/' + frame_range).astype(np.int).tolist()
                print(f'# Selected frame indices: {frame_range}')
            self.pose_list = list(frame_range)
        else:
            self.pose_list = pose_list

        if self.training:
            if used_cam_ids is None:
                self.used_cam_ids = list(range(self.view_num))
            else:
                self.used_cam_ids = used_cam_ids
            print('# Used camera ids: ', self.used_cam_ids)
            # self.data_list = [(self.pose_list[data_idx // self.view_num], data_idx % self.view_num) for data_idx in range(self.view_num * len(self.pose_list))]
            self.data_list = []
            for pose_idx in self.pose_list:
                for view_idx in self.used_cam_ids:
                    self.data_list.append((pose_idx, view_idx))
            # filter missing files
            self.missing_data_list = []
            with open(self.data_dir + '/missing_img_files.txt', 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                line = line.replace('\\', '/')  # considering both Windows and Ubuntu file system
                frame_idx = int(os.path.basename(line).replace('.jpg', ''))
                view_idx = int(os.path.basename(os.path.dirname(line)).replace('cam', ''))
                self.missing_data_list.append((frame_idx, view_idx))
            for missing_data_idx in self.missing_data_list:
                if missing_data_idx in self.data_list:
                    self.data_list.remove(missing_data_idx)

        print('# Dataset contains %d items' % len(self))

        # SMPL related
        smpl_util.smpl_skinning_weights = self.smpl_model.lbs_weights.clone().to(torch.float32).to(config.device)
        # ret = self.smpl_model.forward(config.cano_smpl_pose[None], self.shape[None])
        ret = self.smpl_model.forward(betas = self.smpl_data['betas'][0][None],
                                      global_orient = config.cano_smpl_global_orient[None],
                                      transl = config.cano_smpl_transl[None],
                                      body_pose = config.cano_smpl_body_pose[None])
        self.cano_smpl = {k: v[0] for k, v in ret.items() if isinstance(v, torch.Tensor)}
        self.inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        min_xyz = self.cano_smpl['vertices'].min(0)[0]
        max_xyz = self.cano_smpl['vertices'].max(0)[0]
        self.cano_smpl_center = 0.5 * (min_xyz + max_xyz)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        self.smpl_faces = self.smpl_model.faces.astype(np.int32)

        self.ray_sampling = config.opt['train']['ray_sampling']

        self.depth_dir = self.data_dir + '/depths'

        self.frame_win = int(frame_win)

    def update_ray_sampling_schedule(self, epoch_idx):
        for epoch_range, schedule in zip(self.ray_sampling['epoch_ranges'], self.ray_sampling['schedules']):
            if epoch_range[0] <= epoch_idx <= epoch_range[1]:
                self.ray_sampling['type'] = schedule
                print('# Ray sampling scheme is %s' % schedule)
                return
        raise AssertionError('Failed to choose one scheme for ray sampling.')

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index, self.training)

    @torch.no_grad()
    def getitem(self, index, training = True, **kwargs):
        # time0 = time.time()
        if training or ('eval' in kwargs and kwargs['eval'] == True):  # training or evaluation
            pose_idx, view_idx = self.data_list[index]
            pose_idx = kwargs['pose_idx'] if 'pose_idx' in kwargs else pose_idx
            view_idx = kwargs['view_idx'] if 'view_idx' in kwargs else view_idx
            data_idx = (pose_idx, view_idx)
            if not training:
            # if True:
                print('data index: (%d, %d)' % (pose_idx, view_idx))
        else:  # testing
            pose_idx = self.pose_list[index]
            data_idx = pose_idx
            print('data index: %d' % pose_idx)

        # SMPL
        with torch.no_grad():
            if training:
                left_hand_pose = self.smpl_data['left_hand_pose'][pose_idx]
                right_hand_pose = self.smpl_data['right_hand_pose'][pose_idx]
            else:
                left_hand_pose = config.left_hand_pose
                right_hand_pose = config.right_hand_pose
            live_smpl = self.smpl_model.forward(betas = self.smpl_data['betas'][0][None],
                                                global_orient = self.smpl_data['global_orient'][pose_idx][None],
                                                transl = self.smpl_data['transl'][pose_idx][None],
                                                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                                                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                                                expression = self.smpl_data['expression'][pose_idx][None],
                                                left_hand_pose = left_hand_pose[None],
                                                right_hand_pose = right_hand_pose[None])
            cano_smpl = self.smpl_model.forward(betas = self.smpl_data['betas'][0][None],
                                                global_orient = config.cano_smpl_global_orient[None],
                                                transl = config.cano_smpl_transl[None],
                                                body_pose = config.cano_smpl_body_pose[None],
                                                # jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                                                # expression = self.smpl_data['expression'][pose_idx][None],
                                                left_hand_pose = left_hand_pose[None],
                                                right_hand_pose = right_hand_pose[None])

        data_item = dict()
        if self.load_smpl_pos_map:
            smpl_pos_map = cv.imread(self.data_dir + '/smpl_pos_map/%08d.exr' % pose_idx, cv.IMREAD_UNCHANGED)
            pos_map_size = smpl_pos_map.shape[1] // 2
            smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
            smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
            data_item['smpl_pos_map'] = smpl_pos_map

        if self.load_smpl_nml_map:
            smpl_nml_map = cv.imread(self.data_dir + '/smpl_nml_map/%08d.exr' % pose_idx, cv.IMREAD_UNCHANGED)
            nml_map_size = smpl_nml_map.shape[1] // 2
            smpl_nml_map = np.concatenate([smpl_nml_map[:, :nml_map_size], smpl_nml_map[:, nml_map_size:]], 2)
            smpl_nml_map = smpl_nml_map.transpose((2, 0, 1))
            data_item['smpl_nml_map'] = smpl_nml_map

        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)

        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['time_stamp'] = np.array(pose_idx, np.float32)
        data_item['global_orient'] = self.smpl_data['global_orient'][pose_idx]
        data_item['transl'] = self.smpl_data['transl'][pose_idx]
        if self.frame_win > 0:
            total_frame_num = self.smpl_data['body_pose'].shape[0]
            data_item['pose'] = self.smpl_data['body_pose'][max(0, pose_idx - self.frame_win): min(total_frame_num, pose_idx + self.frame_win + 1)].clone()
        else:
            data_item['pose'] = self.smpl_data['body_pose'][pose_idx].clone()
        if self.fix_head_pose:
            data_item['pose'][..., 3 * 11: 3 * 11 + 3] = 0.
            data_item['pose'][..., 3 * 14: 3 * 14 + 3] = 0.
        if self.fix_hand_pose:
            data_item['pose'][..., 3 * 19: 3 * 19 + 3] = 0.
            data_item['pose'][..., 3 * 20: 3 * 20 + 3] = 0.
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], torch.linalg.inv(cano_smpl.A[0]))
        # data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], self.inv_cano_jnt_mats)
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        data_item['live_bounds'] = live_bounds

        if training:
            color_img = cv.imread(self.data_dir + '/images/cam%02d/%08d.jpg' % (view_idx, pose_idx), cv.IMREAD_UNCHANGED)
            mask_img = cv.imread(self.data_dir + '/masks/cam%02d/%08d.jpg' % (view_idx, pose_idx), cv.IMREAD_UNCHANGED)
            depth_path = self.depth_dir + '/cam%02d/%08d.png' % (view_idx, pose_idx)
            if os.path.exists(depth_path):
                depth_img = cv.imread(depth_path, cv.IMREAD_UNCHANGED)
            else:
                depth_img = np.zeros(color_img.shape[:2], np.uint16)

            # convert each image
            color_img = (color_img / 255.).astype(np.float32)
            depth_img = (depth_img / 1000.).astype(np.float32)

            boundary_mask_img, mask_img = self.get_boundary_mask(mask_img)
            depth_gradient_mask = self.get_depth_gradient_mask(depth_img)
            depth_img[depth_gradient_mask] = 0.

            # sample NeRF data
            ray_sampling = self.ray_sampling
            if ray_sampling['type'] == 'random':
                nerf_random = nerf_util.sample_randomly_for_nerf_rendering(color_img, mask_img, depth_img, self.extr_mats[view_idx], self.intr_mats[view_idx], live_bounds, unsample_region_mask = boundary_mask_img, **ray_sampling['random'])
                data_item.update({'nerf_random': nerf_random})
            elif ray_sampling['type'] == 'patch':
                nerf_patch = nerf_util.sample_patch_for_nerf_rendering(color_img, mask_img, depth_img, self.extr_mats[view_idx], self.intr_mats[view_idx], live_bounds, unsample_region_mask = boundary_mask_img, **ray_sampling['patch'])
                data_item.update({'nerf_patch': nerf_patch})
            elif ray_sampling['type'] == 'both':
                nerf_random = nerf_util.sample_randomly_for_nerf_rendering(color_img, mask_img, depth_img, self.extr_mats[view_idx], self.intr_mats[view_idx], live_bounds, unsample_region_mask = boundary_mask_img, **ray_sampling['random'])
                nerf_patch = nerf_util.sample_patch_for_nerf_rendering(color_img, mask_img, depth_img, self.extr_mats[view_idx], self.intr_mats[view_idx], live_bounds, unsample_region_mask = boundary_mask_img, **ray_sampling['patch'])
                data_item.update({'nerf_random': nerf_random,
                                  'nerf_patch': nerf_patch})
            else:
                raise NotImplemented('Invalid sampling methods!')

            # camera
            data_item.update({
                'extr': self.extr_mats[view_idx],
                'intr': self.intr_mats[view_idx]
            })

        else:
            """ synthesis config """
            img_h = 512 if 'img_h' not in kwargs else kwargs['img_h']
            img_w = 512 if 'img_w' not in kwargs else kwargs['img_w']
            intr = np.array([[550, 0, 256], [0, 550, 256], [0, 0, 1]], np.float32) if 'intr' not in kwargs else kwargs['intr']
            if 'extr' not in kwargs:
                extr = visualize_util.calc_front_mv(live_bounds.mean(0), tar_pos = np.array([0, 0, 2.5]))
            else:
                extr = kwargs['extr']

            """ training data config of view_idx """
            # view_idx = 0
            # img_h = self.img_heights[view_idx]
            # img_w = self.img_widths[view_idx]
            # intr = self.intr_mats[view_idx]
            # extr = self.extr_mats[view_idx]

            uv = self.gen_uv(img_w, img_h)
            uv = uv.reshape(-1, 2)
            ray_d, ray_o = nerf_util.get_rays(uv, extr, intr)
            near, far, mask_at_bound = nerf_util.get_near_far(live_bounds, ray_o, ray_d)
            uv = uv[mask_at_bound]
            ray_o = ray_o[mask_at_bound]
            ray_d = ray_d[mask_at_bound]

            if 'eval' in kwargs and kwargs['eval'] == True:
                if os.path.exists(self.depth_dir + '/cam%02d/%08d.png' % (view_idx, pose_idx)):
                    depth_img = cv.imread(self.depth_dir + '/cam%02d/%08d.png' % (view_idx, pose_idx), cv.IMREAD_UNCHANGED) / 1000.
                else:
                    depth_img = np.zeros((img_h, img_w), np.float32)
                depth_gt = depth_img[uv[:, 1], uv[:, 0]]
                x = (uv[:, 0] + 0.5 - intr[0, 2]) * depth_gt / intr[0, 0]
                y = (uv[:, 1] + 0.5 - intr[1, 2]) * depth_gt / intr[1, 1]
                dist = np.sqrt(x * x + y * y + depth_gt * depth_gt).astype(np.float32)
            else:
                dist = np.zeros_like(near)

            data_item.update({
                'uv': uv,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'dist': dist,
                'img_h': img_h,
                'img_w': img_w,
                'extr': extr,
                'intr': intr
            })

        return data_item

    @staticmethod
    def get_depth_gradient_mask(depth, kernel_size = 5, thres = 0.03):
        sobel_depth_x = np.abs(cv.Sobel(depth, cv.CV_32F, 1, 0)) / 4.
        sobel_depth_y = np.abs(cv.Sobel(depth, cv.CV_32F, 0, 1)) / 4.

        depth_gradient_mask = np.logical_or(sobel_depth_x > thres, sobel_depth_y > thres)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        depth_gradient_mask = cv.dilate(depth_gradient_mask.astype(np.uint8), kernel)
        # cv.imshow('depth_gradient_mask', depth_gradient_mask * 255)
        # cv.waitKey(0)

        return depth_gradient_mask > 0

    @staticmethod
    def get_boundary_mask(mask, kernel_size = 5):
        """
        :param mask: np.uint8
        :param kernel_size:
        :return:
        """
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv.erode(mask.copy(), kernel)
        mask_dilate = cv.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask,
                                      np.logical_and(mask_bk > 5, mask_bk < 250))

        # boundary_mask_resized = cv.resize(boundary_mask.astype(np.uint8), (0, 0), fx = 0.5, fy = 0.5)
        # cv.imshow('boundary_mask', boundary_mask_resized.astype(np.uint8) * 255)
        # cv.waitKey(0)

        return boundary_mask, mask == 1

    @staticmethod
    def gen_uv(img_w, img_h):
        x, y = np.meshgrid(np.linspace(0, img_w - 1, img_w, dtype = np.int),
                           np.linspace(0, img_h - 1, img_h, dtype = np.int))
        uv = np.stack([x, y], axis = -1)
        return uv

import glob
import os
import pickle

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


class PoseDataset(Dataset):
    @torch.no_grad()
    def __init__(self, data_path, frame_range = None, smpl_shape = None, gender = 'neutral', frame_win = 0, fix_head_pose = True, fix_hand_pose = True, denoise = False):
        super(PoseDataset, self).__init__()

        self.data_path = data_path
        self.training = False

        self.gender = gender

        self.seq_name, ext = os.path.splitext(os.path.basename(data_path))
        if ext == '.pkl':
            smpl_data = pickle.load(open(data_path, 'rb'))
            smpl_data = dict(smpl_data)
            self.body_poses = torch.from_numpy(smpl_data['smpl_poses']).to(torch.float32)
            self.transl = torch.from_numpy(smpl_data['smpl_trans']).to(torch.float32) * 1e-3
            self.dataset_name = 'aist++'
        elif ext == '.npz':
            if os.path.basename(data_path).startswith('pose'):
                self.dataset_name = 'thuman4'
            else:
                self.dataset_name = 'aist++_smplx'
            smpl_data = np.load(data_path)
            smpl_data = dict(smpl_data)
            smpl_data = {k: torch.from_numpy(v).to(torch.float32) for k, v in smpl_data.items()}
            frame_num = smpl_data['body_pose'].shape[0]
            self.body_poses = torch.zeros((frame_num, 72), dtype = torch.float32)
            self.body_poses[:, :3] = smpl_data['global_orient']
            self.body_poses[:, 3:3+21*3] = smpl_data['body_pose']
            self.transl = smpl_data['transl']

            data_dir = os.path.dirname(data_path)
            calib_path = os.path.basename(data_path).replace('.npz', '.json').replace('pose', 'calibration')
            calib_path = data_dir + '/' + calib_path
            if os.path.exists(calib_path):
                cam_data = json.load(open(calib_path, 'r'))
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
        else:
            raise AssertionError('Invalid data_path!')

        self.fix_head_pose = fix_head_pose
        self.fix_hand_pose = fix_hand_pose

        self.smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = self.gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)

        pose_list = list(range(self.body_poses.shape[0]))
        if frame_range is not None:
            if isinstance(frame_range, list):
                if len(frame_range) == 2:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]})')
                    frame_range = range(frame_range[0], frame_range[1])
                elif len(frame_range) == 3:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
                    frame_range = range(frame_range[0], frame_range[1], frame_range[2])
            self.pose_list = list(frame_range)
        else:
            self.pose_list = pose_list

        print('# Dataset contains %d items' % len(self))

        # SMPL related
        smpl_util.smpl_skinning_weights = self.smpl_model.lbs_weights.clone().to(torch.float32).to(config.device)
        self.smpl_shape = smpl_shape.to(torch.float32) if smpl_shape is not None else torch.zeros(10, dtype = torch.float32)
        ret = self.smpl_model.forward(betas = self.smpl_shape[None],
                                      global_orient = config.cano_smpl_global_orient[None],
                                      transl = config.cano_smpl_transl[None],
                                      body_pose = config.cano_smpl_body_pose[None],
                                      left_hand_pose = config.left_hand_pose[None],
                                      right_hand_pose = config.right_hand_pose[None])
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

        self.frame_win = int(frame_win)
        self.denoise = denoise
        if self.denoise:
            win_size = 1
            body_poses_clone = self.body_poses.clone()
            transl_clone = self.transl.clone()
            frame_num = body_poses_clone.shape[0]
            self.body_poses[win_size: frame_num-win_size] = 0
            self.transl[win_size: frame_num-win_size] = 0
            for i in range(-win_size, win_size + 1):
                self.body_poses[win_size: frame_num-win_size] += body_poses_clone[win_size+i: frame_num-win_size+i]
                self.transl[win_size: frame_num-win_size] += transl_clone[win_size+i: frame_num-win_size+i]
            self.body_poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
            self.transl[win_size: frame_num-win_size] /= (2 * win_size + 1)

    def __len__(self):
        return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index)

    @torch.no_grad()
    def getitem(self, index, **kwargs):
        pose_idx = self.pose_list[index]
        data_idx = pose_idx
        print('data index: %d' % pose_idx)

        # SMPL
        live_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
                                            global_orient = self.body_poses[pose_idx, :3][None],
                                            transl = self.transl[pose_idx][None],
                                            body_pose = self.body_poses[pose_idx, 3: 66][None],
                                            left_hand_pose = config.left_hand_pose[None],
                                            right_hand_pose = config.right_hand_pose[None])

        data_item = dict()
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['global_orient'] = self.body_poses[pose_idx, :3]
        data_item['transl'] = self.transl[pose_idx]
        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['pose_1st'] = self.body_poses[0, 3: 66]
        if self.frame_win > 0:
            total_frame_num = len(self.pose_list)
            selected_frames = self.pose_list[max(0, index - self.frame_win): min(total_frame_num, index + self.frame_win + 1)]
            data_item['pose'] = self.body_poses[selected_frames, 3: 66].clone()
        else:
            data_item['pose'] = self.body_poses[pose_idx, 3: 66].clone()

        if self.fix_head_pose:
            data_item['pose'][..., 3 * 11: 3 * 11 + 3] = 0.
            data_item['pose'][..., 3 * 14: 3 * 14 + 3] = 0.
        if self.fix_hand_pose:
            data_item['pose'][..., 3 * 19: 3 * 19 + 3] = self.body_poses[0, 3 * 20: 3 * 20 + 3]
            data_item['pose'][..., 3 * 20: 3 * 20 + 3] = self.body_poses[0, 3 * 21: 3 * 21 + 3]
        data_item['time_stamp'] = np.array(pose_idx, np.float32)
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], self.inv_cano_jnt_mats)
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        data_item['live_bounds'] = live_bounds

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

        data_item.update({
            'uv': uv,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'dist': np.zeros_like(near),
            'img_h': img_h,
            'img_w': img_w,
            'extr': extr,
            'intr': intr
        })

        return data_item

    @staticmethod
    def gen_uv(img_w, img_h):
        x, y = np.meshgrid(np.linspace(0, img_w - 1, img_w, dtype = np.int),
                           np.linspace(0, img_h - 1, img_h, dtype = np.int))
        uv = np.stack([x, y], axis = -1)
        return uv

    @torch.no_grad()
    def visualize(self, interval = 1):
        from utils.renderer import Renderer, gl_perspective_projection_matrix
        renderer = Renderer(512, 512, shader_name = 'phong_geometry')
        extr = None
        proj_mat = gl_perspective_projection_matrix(550, 550, 256, 256, 512, 512)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(os.path.join(os.path.dirname(self.data_path), os.path.basename(self.data_path).replace('.pkl', '.mp4')), fourcc, 30, (512, 512))

        for index in range(0, len(self), interval):
            live_smpl = self.smpl_model.forward(betas = self.smpl_shape[None],
                                                global_orient = self.body_poses[index, :3][None],
                                                transl = self.transl[index][None] * 1e-3,
                                                body_pose = self.body_poses[index, 3: 66][None],
                                                left_hand_pose = config.left_hand_pose[None],
                                                right_hand_pose = config.right_hand_pose[None])

            # import trimesh
            live_smpl_mesh = trimesh.Trimesh(live_smpl.vertices[0].numpy(),
                                             self.smpl_faces)
            # live_smpl_mesh.export('../debug/live_smpl_mesh.obj')
            # print(self.transl[index])
            # print(self.body_poses[index, :3])
            # exit(1)

            if extr is None:
                object_center = 0.5 * (live_smpl.vertices[0].min(0)[0] + live_smpl.vertices[0].max(0)[0])
                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = 0.,
                                                   # global_orient = cv.Rodrigues(self.body_poses[index, :3].numpy())[0])
                                                   global_orient = None)

            real2gl = np.identity(4, np.float32)
            real2gl[:3, :3] = cv.Rodrigues(np.array([np.pi, 0, 0], np.float32))[0]
            renderer.set_mv_mat(real2gl @ extr)
            renderer.set_mvp_mat(proj_mat @ extr)
            renderer.set_model(live_smpl_mesh.vertices[live_smpl_mesh.faces.reshape(-1)].astype(np.float32),
                               live_smpl_mesh.vertex_normals[live_smpl_mesh.faces.reshape(-1)].astype(np.float32))
            img = renderer.render()[:, :, :3]
            img = (img * 255).astype(np.uint8)
            # cv.imshow('smpl_x', img)
            # key = cv.waitKey(1)
            # if key == 27:
            #     return

            # cv.imwrite(output_dir + '/%d.jpg' % index, (img * 255).astype(np.uint8))
            writer.write(img)

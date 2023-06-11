import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import torch
import numpy as np

from base_trainer import BaseTrainer
import config
from network.avatar import AvatarNet
from network.lpips import LPIPS

from dataset.dataset_pose import PoseDataset
import utils.lr_schedule as lr_schedule
import utils.net_util as net_util
import utils.recon_util as recon_util
import utils.visualize_util as visualize_util
from utils.nerf_util import get_rays


class AvatarTrainer(BaseTrainer):
    def __init__(self, opt):
        super(AvatarTrainer, self).__init__(opt)

    def update_config_before_epoch(self, epoch_idx):
        # update ray sampling scheme
        self.dataset.update_ray_sampling_schedule(epoch_idx)

        if epoch_idx > 5:
            config.opt['train']['compute_grad'] = False
        else:
            config.opt['train']['compute_grad'] = True

        if self.dataset.ray_sampling['type'] == 'patch':
            self.loss_weight['tv'] = 1e2

    def forward_one_pass(self, items):
        total_loss = 0
        batch_losses = {}

        """ random sampling """
        if 'nerf_random' in items:
            items.update(items['nerf_random'])
            render_output = self.network.render(items, depth_guided_sampling = self.opt['train']['depth_guided_sampling'])

            # color loss
            if 'rgb_map' in render_output:
                color_loss = torch.nn.L1Loss()(render_output['rgb_map'], items['color_gt'])
                total_loss += self.loss_weight['color'] * color_loss
                batch_losses.update({
                    'color_loss_random': color_loss.item()
                })

            # mask loss
            if 'acc_map' in render_output:
                mask_loss = torch.nn.L1Loss()(render_output['acc_map'], items['mask_gt'])
                total_loss += self.loss_weight['mask'] * mask_loss
                batch_losses.update({
                    'mask_loss_random': mask_loss.item()
                })

            # eikonal loss
            if 'normal' in render_output:
                eikonal_loss = ((torch.linalg.norm(render_output['normal'], dim = -1) - 1.) ** 2).mean()
                total_loss += self.loss_weight['eikonal'] * eikonal_loss
                batch_losses.update({
                    'eikonal_loss': eikonal_loss.item()
                })

            """ regularization """
            if self.loss_weight['tv'] > 0.:
                if 'tv_loss' in render_output:
                    tv_loss = render_output['tv_loss'].mean()
                    total_loss += self.loss_weight['tv'] * tv_loss
                    batch_losses.update({
                        'tv_loss_random': tv_loss.item()
                    })

        """ patch sampling """
        if 'nerf_patch' in items:
            items.update(items['nerf_patch'])
            render_output = self.network.render(items, depth_guided_sampling = self.opt['train']['depth_guided_sampling'])

            # color loss
            if 'rgb_map' in render_output:
                color_loss = torch.nn.L1Loss()(render_output['rgb_map'], items['color_gt'])
                total_loss += self.loss_weight['color'] * color_loss
                batch_losses.update({
                    'color_loss_patch': color_loss.item()
                })

                if self.loss_weight['lpips'] > 0.:
                    patch_num = self.opt['train']['ray_sampling']['patch']['patch_num']
                    patch_size = self.opt['train']['ray_sampling']['patch']['patch_size']
                    rgb_map = render_output['rgb_map'].reshape(-1, patch_size, patch_size, 3)
                    rgb_map_gt = items['color_gt'].reshape(-1, patch_size, patch_size, 3)

                    # convert to rgb
                    rgb_map = rgb_map[..., [2, 1, 0]]
                    rgb_map_gt = rgb_map_gt[..., [2, 1, 0]]
                    lpips_loss = self.lpips.forward(rgb_map.permute(0, 3, 1, 2),
                                                    rgb_map_gt.permute(0, 3, 1, 2),
                                                    normalize = True).mean()
                    total_loss += 0.1 * self.loss_weight['lpips'] * lpips_loss
                    batch_losses.update({
                        'lpips_loss': lpips_loss.item()
                    })

            # mask loss
            if 'acc_map' in render_output:
                mask_loss = torch.nn.L1Loss()(render_output['acc_map'], items['mask_gt'])
                total_loss += self.loss_weight['mask'] * mask_loss
                batch_losses.update({
                    'mask_loss_patch': mask_loss.item()
                })

            """ regularization """
            if self.loss_weight['tv'] > 0.:
                if 'tv_loss' in render_output:
                    tv_loss = render_output['tv_loss'].mean()
                    total_loss += self.loss_weight['tv'] * tv_loss
                    batch_losses.update({
                        'tv_loss_patch': tv_loss.item()
                    })

        return total_loss, batch_losses

    def run(self):
        MvRgbDataset = __import__(self.opt['train'].get('dataset', 'dataset.dataset_mv_rgb_slrf'), fromlist = ['MvRgbDataset']).MvRgbDataset
        self.set_dataset(MvRgbDataset(**self.opt['train']['data']))
        self.set_network(AvatarNet(self.opt['model']).to(config.device))
        self.set_net_dict({
            'network': self.network
        })
        self.set_optm_dict({
            'network': torch.optim.Adam(self.network.parameters(), lr = 1e-3)
        })
        self.set_lr_schedule_dict({
            'network': lr_schedule.get_learning_rate_schedules(**self.opt['train']['lr']['network'])
        })
        self.set_update_keys(['network'])

        if 'lpips' in self.opt['train']['loss_weight']:
            self.lpips = LPIPS(net = 'vgg').to(config.device)
            for p in self.lpips.parameters():
                p.requires_grad = False

        self.train()

    def test_geometry(self, items, space = 'live', testing_res = (128, 128, 128)):
        if space == 'live':
            bounds = items['live_bounds'][0]
        else:
            bounds = items['cano_bounds'][0]
        vol_pts = net_util.generate_volume_points(bounds, testing_res)
        chunk_size = 256 * 256 * 4
        sdf_list = []
        for i in range(0, vol_pts.shape[0], chunk_size):
            vol_pts_chunk = vol_pts[i: i + chunk_size][None]
            if space == 'live':
                cano_pts_chunk, near_flag = self.network.transform_live2cano(vol_pts_chunk, items, near_thres = 0.1)
            else:
                cano_pts_chunk = vol_pts_chunk
                near_flag = torch.ones(cano_pts_chunk.shape[:2], dtype = torch.bool)
            sdf_chunk = torch.zeros(cano_pts_chunk.shape[1]).to(cano_pts_chunk)
            if near_flag.sum() > 0:
                ret = self.network.forward_cano_radiance_field(cano_pts_chunk[near_flag][None], None, items['pose'])
                sdf_chunk[near_flag[0]] = ret['sdf'][0, :, 0]
            sdf_list.append(sdf_chunk)
        sdf_list = torch.cat(sdf_list, 0)
        vertices, faces, normals = recon_util.recon_mesh(sdf_list, testing_res, bounds, iso_value = 0.)
        return vertices, faces, normals

    @torch.no_grad()
    def test(self):
        from utils.renderer import Renderer, gl_perspective_projection_matrix
        from utils.net_util import to_cuda
        from utils.obj_io import save_mesh_as_ply
        import cv2 as cv

        MvRgbDataset = __import__(self.opt['test'].get('dataset', 'dataset.dataset_mv_rgb_slrf'), fromlist = ['MvRgbDataset']).MvRgbDataset
        training_dataset = MvRgbDataset(**self.opt['test']['data'], training = False)
        if 'pose_data' in self.opt['test']:
            testing_dataset = PoseDataset(**self.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0])
            dataset_name = testing_dataset.dataset_name
            seq_name = testing_dataset.seq_name
        else:
            testing_dataset = training_dataset
            dataset_name = 'training'
            seq_name = ''

        self.set_dataset(testing_dataset)
        self.set_network(AvatarNet(self.opt['model']).to(config.device))
        self.network.eval()
        self.set_net_dict({
            'network': self.network
        })
        self.load_ckpt(self.opt['test']['prev_ckpt'], False)

        output_dir = self.opt['test'].get('output_dir', None)
        if output_dir is None:
            view_setting = config.opt['test'].get('view_setting', 'free')
            if view_setting == 'free':
                view_folder = 'free_view'
            elif view_setting == 'camera':
                view_folder = '%d_view' % config.opt['test']['render_view_idx']
            else:
                raise ValueError('Invalid view setting for animation!')
            output_dir = './test_results/{}/{}/{}/{}'.format(training_dataset.subject_name, dataset_name, seq_name, view_folder)
        print('# Output dir: %s' % output_dir)
        os.makedirs(output_dir + '/live_geometry', exist_ok = True)
        os.makedirs(output_dir + '/live_geometry/rendered_geometry', exist_ok = True)
        os.makedirs(output_dir + '/acc_map', exist_ok = True)
        os.makedirs(output_dir + '/live_skeleton', exist_ok = True)
        os.makedirs(output_dir + '/rgb_map', exist_ok = True)

        pos_renderer = None
        geo_renderer = None
        phong_renderer = None
        item_0 = self.dataset.getitem(0, training = False)
        object_center = item_0['live_bounds'].mean(0)
        global_orient = item_0['global_orient'].numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
        global_orient = cv.Rodrigues(global_orient)[0]

        data_num = len(self.dataset)
        for idx in range(data_num):
            time_ani_start = time.time()

            img_scale = self.opt['test'].get('img_scale', 1.0)
            view_setting = config.opt['test'].get('view_setting', 'free')
            if view_setting == 'camera':
                # training view setting
                cam_id = config.opt['test']['render_view_idx']
                intr = self.dataset.intr_mats[cam_id].copy()
                intr[:2] *= img_scale
                item = self.dataset.getitem(idx,
                                            training = False,
                                            extr = self.dataset.extr_mats[cam_id],
                                            intr = intr,
                                            img_w = int(img_scale * self.dataset.img_widths[cam_id]),
                                            img_h = int(img_scale * self.dataset.img_heights[cam_id]))
            elif view_setting == 'free':
                # free view setting
                frame_num_per_circle = 216
                rot_Y = (idx % frame_num_per_circle) / float(frame_num_per_circle) * 2 * np.pi

                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = rot_Y,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
                item = self.dataset.getitem(idx, training = False, extr = extr,
                                            intr = intr, img_w = img_w, img_h = img_h)
            else:
                raise ValueError('Invalid view setting for animation!')

            items = to_cuda(item, add_batch = True)

            if self.opt['test']['depth_guided_sampling']['flag']:
                vertices, faces, normals = self.test_geometry(items, 'live', testing_res = self.opt['test']['vol_res'])
                if self.opt['test']['save_mesh']:
                    save_mesh_as_ply(output_dir + '/live_geometry/%s.ply' % item['data_idx'],
                                     vertices, faces, normals)

                # render geometry
                if geo_renderer is None:
                    geo_renderer = Renderer(item['img_w'], item['img_h'], shader_name = 'phong_geometry', bg_color = (1, 1, 1))
                extr, intr = item['extr'], item['intr']
                proj_mat = gl_perspective_projection_matrix(intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], item['img_w'], item['img_h'])
                geo_renderer.set_mvp_mat(proj_mat @ extr)
                extr_gl = extr.copy()
                extr_gl[1:3, :3] *= -1
                geo_renderer.set_mv_mat(extr_gl)
                geo_renderer.set_model(vertices[faces.reshape(-1)].astype(np.float32), normals[faces.reshape(-1)].astype(np.float32))
                geo_img = geo_renderer.render()[:, :, :3]
                geo_img = (geo_img * 255).astype(np.uint8)
                cv.imwrite(output_dir + '/live_geometry/rendered_geometry/%s.jpg' % item['data_idx'], geo_img)

            if self.opt['test'].get('render_skeleton', False):
                import trimesh
                from utils.visualize_skeletons import construct_skeletons
                skel_vertices, skel_faces = construct_skeletons(items['joints'][0].cpu().numpy(), items['kin_parent'][0].cpu().numpy())
                skel_mesh = trimesh.Trimesh(skel_vertices, skel_faces, process = False)

                if phong_renderer is None:
                    phong_renderer = Renderer(item['img_w'], item['img_h'], shader_name = 'phong_geometry', bg_color = (1, 1, 1))
                extr, intr = item['extr'], item['intr']
                proj_mat = gl_perspective_projection_matrix(intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], item['img_w'], item['img_h'])
                phong_renderer.set_mvp_mat(proj_mat @ extr)
                extr_gl = extr.copy()
                extr_gl[1:3, :3] *= -1
                phong_renderer.set_mv_mat(extr_gl)
                phong_renderer.set_model(skel_vertices[skel_faces.reshape(-1)], skel_mesh.vertex_normals.astype(np.float32)[skel_faces.reshape(-1)])
                skel_img = phong_renderer.render()[:, :, :3]
                skel_img = (skel_img * 255).astype(np.uint8)

                cv.imwrite(output_dir + '/live_skeleton/%s.jpg' % item['data_idx'], skel_img)

            if not self.opt['test']['infer_rgb']:
                time_ani_end = time.time()
                print('Animating one frame costs %f secs' % (time_ani_end - time_ani_start))
                torch.cuda.empty_cache()
                continue

            if self.opt['test']['depth_guided_sampling']['flag']:
                if pos_renderer is None:
                    pos_renderer = Renderer(item['img_w'], item['img_h'], shader_name = 'position')
                extr, intr = item['extr'], item['intr']
                proj_mat = gl_perspective_projection_matrix(intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], item['img_w'], item['img_h'])
                pos_renderer.set_mvp_mat(proj_mat @ extr)
                pos_renderer.set_model(vertices[faces.reshape(-1)].astype(np.float32))
                pos_map = pos_renderer.render()[..., :3]
                nonzero_flag = np.linalg.norm(pos_map, axis = -1) > 1e-6
                pos_map[nonzero_flag] = np.einsum('ij,vj->vi', extr[:3, :3], pos_map[nonzero_flag]) + extr[:3, 3]
                dist_map = np.linalg.norm(pos_map, axis = -1)

                infer_mask = cv.dilate(nonzero_flag.astype(np.uint8), np.ones((5, 5), np.uint8))
                uv = np.argwhere(infer_mask > 0)[:, [1, 0]].astype(np.int64)
                near = np.zeros(uv.shape[0], np.float32)
                far = np.zeros(uv.shape[0], np.float32)
                ray_d, ray_o = get_rays(uv, item['extr'], item['intr'])
                dist = dist_map[uv[:, 1], uv[:, 0]]
                items['uv'] = torch.from_numpy(uv).to(torch.long).to(config.device).unsqueeze(0)
                items['near'] = torch.from_numpy(near).to(torch.float32).to(config.device).unsqueeze(0)
                items['far'] = torch.from_numpy(far).to(torch.float32).to(config.device).unsqueeze(0)
                items['ray_o'] = torch.from_numpy(ray_o).to(torch.float32).to(config.device).unsqueeze(0)
                items['ray_d'] = torch.from_numpy(ray_d).to(torch.float32).to(config.device).unsqueeze(0)
                items['dist'] = torch.from_numpy(dist).to(torch.float32).to(config.device).unsqueeze(0)

            output = self.network.render(items,
                                         depth_guided_sampling = self.opt['test']['depth_guided_sampling'])

            # re-infer for output['acc_map'] < 0.99, because the rendered depth is not so accurate on boundaries of self-occluded regions
            uv = items['uv'][0].to(torch.long)
            reinfer_mask = output['acc_map'] < 0.99
            for k in ['uv', 'near', 'far', 'ray_o', 'ray_d', 'dist']:
                items[k] = items[k][reinfer_mask].unsqueeze(0)
            reinfer_output = self.network.render(items, depth_guided_sampling = {'flag': False})
            output['rgb_map'][0, reinfer_mask[0]] = reinfer_output['rgb_map'][0]
            output['acc_map'][0, reinfer_mask[0]] = reinfer_output['acc_map'][0]

            # save rgb_map & acc_map
            if 'rgb_map' in output:
                rgb_map = torch.zeros((item['img_h'], item['img_w'], 3), dtype = torch.float32, device = config.device).fill_(config.bg_color)
                rgb_map[uv[:, 1], uv[:, 0]] = output['rgb_map'][0]
                rgb_map.clip_(0., 1.)
                rgb_map = (rgb_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + '/rgb_map/%s.png' % item['data_idx'], (rgb_map.cpu().numpy()).astype(np.uint8))

            if 'acc_map' in output:
                acc_map = torch.zeros((item['img_h'], item['img_w']), dtype = torch.float32, device = config.device)
                acc_map[uv[:, 1], uv[:, 0]] = output['acc_map'][0]
                acc_map.clip_(0., 1.)
                cv.imwrite(output_dir + '/acc_map/%s.png' % item['data_idx'], (acc_map.cpu().numpy() * 255).astype(np.uint8))

            time_ani_end = time.time()
            print('Animating one frame costs %f secs' % (time_ani_end - time_ani_start))

            torch.cuda.empty_cache()

    @torch.no_grad()
    def mini_test(self):
        import cv2 as cv
        self.network.eval()

        # training data
        pose_idx, view_idx = self.opt['train'].get('eval_training_ids', (310, 19))
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = False,
                                    eval = True,
                                    img_h = self.dataset.img_heights[view_idx],
                                    img_w = self.dataset.img_widths[view_idx],
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = self.dataset.intr_mats[view_idx])
        items = net_util.to_cuda(item, add_batch = True)
        output = self.network.render(items, depth_guided_sampling = {'flag': True, 'near_sur_dist': 0.05, 'N_ray_samples': 32})

        if 'rgb_map' in output:
            rgb_map = torch.zeros((item['img_h'], item['img_w'], 3), dtype = torch.float32, device = config.device)
            acc_map = torch.zeros((item['img_h'], item['img_w']), dtype = torch.float32, device = config.device)
            uv = items['uv'][0].to(torch.long)
            rgb_map[uv[:, 1], uv[:, 0]] = output['rgb_map'][0]
            acc_map[uv[:, 1], uv[:, 0]] = output['acc_map'][0]
            rgb_map.clip_(0., 1.)
            acc_map.clip_(0., 1.)
            # cv.imshow('rgb_map', rgb_map.cpu().numpy())
            # cv.imshow('acc_map', acc_map.cpu().numpy())
            # cv.waitKey(0)
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval/training'
            os.makedirs(output_dir, exist_ok = True)
            cv.imwrite(output_dir + '/nerf_batch_%d.jpg' % self.iter_idx, (rgb_map.cpu().numpy() * 255).astype(np.uint8))

        # testing data
        pose_idx, view_idx = self.opt['train'].get('eval_testing_ids', (2012, 21))
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = False,
                                    eval = True,
                                    img_h = self.dataset.img_heights[view_idx],
                                    img_w = self.dataset.img_widths[view_idx],
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = self.dataset.intr_mats[view_idx])
        items = net_util.to_cuda(item, add_batch = True)
        output = self.network.render(items, depth_guided_sampling = {'flag': True, 'near_sur_dist': 0.05, 'N_ray_samples': 32})

        if 'rgb_map' in output:
            rgb_map = torch.zeros((item['img_h'], item['img_w'], 3), dtype = torch.float32, device = config.device)
            acc_map = torch.zeros((item['img_h'], item['img_w']), dtype = torch.float32, device = config.device)
            uv = items['uv'][0].to(torch.long)
            rgb_map[uv[:, 1], uv[:, 0]] = output['rgb_map'][0]
            acc_map[uv[:, 1], uv[:, 0]] = output['acc_map'][0]
            rgb_map.clip_(0., 1.)
            acc_map.clip_(0., 1.)
            # cv.imshow('rgb_map', rgb_map.cpu().numpy())
            # cv.imshow('acc_map', acc_map.cpu().numpy())
            # cv.waitKey(0)
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval/testing'
            os.makedirs(output_dir, exist_ok = True)
            cv.imwrite(output_dir + '/nerf_batch_%d.jpg' % self.iter_idx, (rgb_map.cpu().numpy() * 255).astype(np.uint8))

        self.set_train()

    @torch.no_grad()
    def render_depth_sequences(self):
        from utils.renderer import Renderer, gl_perspective_projection_matrix
        from utils.net_util import to_cuda
        from utils.obj_io import save_mesh_as_ply
        import cv2 as cv

        MvRgbDataset = __import__(self.opt['train'].get('dataset', 'dataset.dataset_mv_rgb_slrf'), fromlist = ['MvRgbDataset']).MvRgbDataset
        training_dataset = MvRgbDataset(**self.opt['train']['data'], training = False)
        renderers = [Renderer(training_dataset.img_widths[i], training_dataset.img_heights[i], shader_name = 'position') for i in range(training_dataset.view_num)]

        self.set_dataset(training_dataset)
        self.set_network(AvatarNet(self.opt['model']).to(config.device))
        self.network.eval()
        self.set_net_dict({
            'network': self.network
        })
        self.load_ckpt(self.opt['train']['net_ckpt_dir'] + '/epoch_latest', False)

        for view_idx in range(training_dataset.view_num):
            os.makedirs(training_dataset.data_dir + '/depths/cam%02d' % view_idx, exist_ok = True)

        for idx in range(len(training_dataset)):
            item = training_dataset.getitem(idx, training = False)
            items = to_cuda(item, add_batch = True)

            vertices, faces, normals = self.test_geometry(items, 'live', testing_res = (256, 256, 256))

            # # debug
            # save_mesh_as_ply('./debug/live_geometry_%s.ply' % item['data_idx'], vertices, faces, normals)
            # exit(1)

            vertices = vertices.astype(np.float32)
            vertices = vertices[faces.reshape(-1)]

            for view_idx in range(training_dataset.view_num):
                renderer = renderers[view_idx]
                intr = training_dataset.intr_mats[view_idx]
                extr = training_dataset.extr_mats[view_idx]
                proj_mat = gl_perspective_projection_matrix(intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], renderer.img_w, renderer.img_h)
                renderer.set_mvp_mat(proj_mat @ extr)
                renderer.set_model(vertices)
                pos_map = renderer.render()[..., :3]
                mask = np.linalg.norm(pos_map, axis = -1) > 1e-6
                pos_map[mask] = np.einsum('ij,vj->vi', extr[:3, :3], pos_map[mask]) + extr[:3, 3]
                depth_map = (pos_map[:, :, 2] * 1000).astype(np.uint16)

                cv.imwrite(training_dataset.data_dir + '/depths/cam%02d/%08d.png' % (view_idx, int(item['data_idx'])), depth_map)


if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)

    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-m', '--mode', type = str, help = 'Running mode.', choices = ['train', 'test', 'render_depth_sequences', None], default = None)
    args = arg_parser.parse_args()

    config.load_global_opt(args.config_path)
    if args.mode is not None:
        config.opt['mode'] = args.mode

    trainer = AvatarTrainer(config.opt)
    if config.opt['mode'] == 'train':
        trainer.run()
    elif config.opt['mode'] == 'test':
        trainer.test()
    elif config.opt['mode'] == 'render_depth_sequences':
        trainer.render_depth_sequences()
    else:
        raise NotImplementedError('Invalid running mode!')

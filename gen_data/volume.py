import os

import numpy as np
import igl
import smplx
import torch
import trimesh
import config


@torch.no_grad()
def calc_cano_weight_volume(data_dir, gender = 'neutral'):
    smpl_params = np.load(data_dir + '/smpl_params.npz')
    smpl_shape = torch.from_numpy(smpl_params['betas'][0]).to(torch.float32)
    smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)

    def get_grid_points(xyz):
        min_xyz = np.min(xyz, axis = 0)
        max_xyz = np.max(xyz, axis = 0)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        bounds = np.stack([min_xyz, max_xyz], axis = 0)
        center = 0.5 * (min_xyz + max_xyz)
        vsize = 0.01
        voxel_size = [vsize, vsize, vsize]
        x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
        y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
        z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing = 'ij'), axis = -1)
        return pts, bounds, center

    if isinstance(smpl_model, smplx.SMPLX):
        cano_smpl = smpl_model.forward(betas = smpl_shape[None],
                                       global_orient = config.cano_smpl_global_orient[None],
                                       transl = config.cano_smpl_transl[None],
                                       body_pose = config.cano_smpl_body_pose[None])
    elif isinstance(smpl_model, smplx.SMPL):
        cano_smpl = smpl_model.forward(betas = smpl_shape[None],
                                       global_orient = config.cano_smpl_global_orient[None],
                                       transl = config.cano_smpl_transl[None],
                                       body_pose = config.cano_smpl_pose[6:][None])
    else:
        raise TypeError('Not support this SMPL type.')
    cano_smpl.vertices = cano_smpl.vertices[0]

    # generate volume pts
    pts, bounds, center = get_grid_points(cano_smpl.vertices.numpy())
    X, Y, Z, _ = pts.shape
    print('Volume resolution: (%d, %d, %d)' % (X, Y, Z))
    pts = pts.reshape(-1, 3)

    # barycentric
    dists, face_id, closest_pts = igl.signed_distance(pts, cano_smpl.vertices.numpy(), smpl_model.faces.astype(np.int32))
    triangles = cano_smpl.vertices.numpy()[smpl_model.faces[face_id]]
    weights = smpl_model.lbs_weights.numpy()[smpl_model.faces[face_id]]
    barycentric_weight = trimesh.triangles.points_to_barycentric(triangles, closest_pts)
    weights = (barycentric_weight[:, :, None] * weights).sum(1)
    # weights[dists > 0.08] = 0.
    dists = dists.reshape(X, Y, Z).astype(np.float32)

    weights = weights.reshape(X, Y, Z, -1).astype(np.float32)
    # return weights, -dists
    np.savez(data_dir + '/cano_weight_volume.npz',
             weight_volume = weights,
             sdf_volume = -dists,
             bounds = bounds,
             center = center)

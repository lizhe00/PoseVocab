import torch
import pytorch3d.ops
from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
from utils.knn import knn_gather


def calc_blending_weight(query_pts, smpl_v, smpl_f, near_thres = 0.08, method = 'barycentric'):
    """
    :param query_pts: (B, N, 3)
    :param smpl_v: (B, M, 3)
    :param smpl_f: (B, F, 3)
    :param near_thres:
    :param method: 'NN' or 'barycentric'
    :return:
    """
    assert (query_pts.shape[0] == smpl_v.shape[0] == smpl_f.shape[0])
    batch_size = query_pts.shape[0]
    if method == 'NN':
        # NN
        dists_to_smpl, indices, _ = pytorch3d.ops.knn_points(query_pts, smpl_v, K = 1)
        near_flag = dists_to_smpl[:, :, 0] < near_thres ** 2
        pts_w = pytorch3d.ops.knn_gather(smpl_skinning_weights[None].expand(batch_size, -1, -1), indices)
        pts_w = pts_w[:, :, 0]
    else:
        dists_to_smpl, face_indices, bc_coords = nearest_face_pytorch3d(query_pts, smpl_v, smpl_f[0])

        face_vertex_ids = torch.gather(smpl_f.long(), 1, face_indices[:, :, None].long().expand(-1, -1, 3))  # (B, N, 3)
        face_lbs = knn_gather(smpl_skinning_weights[None].expand(batch_size, -1, -1), face_vertex_ids)
        pts_w = (bc_coords[..., None] * face_lbs).sum(2)
        near_flag = dists_to_smpl < near_thres
    return pts_w, near_flag


def skinning(points, lbs, jnt_mats, return_pt_mats = False):
    """
    forward skinning
    :param points: (B, N, 3)
    :param lbs: (B, N, 24)
    :param jnt_mats: (B, 24, 4, 4)
    :return:
    """
    # lbs
    pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, jnt_mats)

    live_pts = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], points) + pt_mats[..., :3, 3]

    if return_pt_mats:
        return live_pts, pt_mats
    else:
        return live_pts


def skinning_normal(normals, lbs, jnt_mats):
    # lbs
    pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, jnt_mats)

    live_normals = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], normals)
    return live_normals


smpl_skinning_weights = None  # should be initialized

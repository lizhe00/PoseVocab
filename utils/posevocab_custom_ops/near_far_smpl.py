import torch

import posevocab_custom_ops


def near_far_smpl(vertices, ray_o, ray_d, radius = 0.05):
    vertices = vertices.contiguous().to(torch.float32)
    ray_o = ray_o.contiguous().to(torch.float32)
    ray_d = ray_d.contiguous().to(torch.float32)
    ray_num = ray_o.shape[0]
    near = torch.cuda.FloatTensor(ray_num).fill_(0.0).contiguous()
    far = torch.cuda.FloatTensor(ray_num).fill_(0.0).contiguous()
    posevocab_custom_ops.near_far_smpl(vertices, ray_o, ray_d, near, far, radius)
    return near, far




import torch

import posevocab_custom_ops


def nearest_face(vertices, faces, query_points):
    vertices = vertices.contiguous().to(torch.float32)
    faces = faces.contiguous().to(torch.int32)
    query_points = query_points.contiguous().to(torch.float32)
    query_num = query_points.size(0)
    dist = torch.cuda.FloatTensor(query_num).fill_(0.0).contiguous()
    face_ids = torch.cuda.IntTensor(query_num).fill_(-1).contiguous()
    nearest_pts = torch.cuda.FloatTensor(query_num, 3).fill_(0.0).contiguous()
    posevocab_custom_ops.nearest_face(vertices, faces, query_points, dist, face_ids, nearest_pts)
    return dist, face_ids, nearest_pts


def nearest_face_pytorch3d(points, vertices, faces):
    """
    :param points: (B, N, 3)
    :param vertices: (B, M, 3)
    :param faces: (F, 3)
    :return dists (B, N), indices (B, N), bc_coords (B, N, 3)
    """
    B, N = points.shape[:2]
    F = faces.shape[0]
    dists, indices, bc_coords = [], [], []
    points = points.contiguous()
    for b in range(B):
        triangles = vertices[b, faces.reshape(-1).to(torch.long)].reshape(F, 3, 3)
        triangles = triangles.contiguous()

        l_idx = torch.tensor([0, ]).to(torch.long).to(points.device)
        dist, index, w0, w1, w2 = posevocab_custom_ops.nearest_face_pytorch3d(
            points[b],
            l_idx,
            triangles,
            l_idx,
            N
        )
        dists.append(torch.sqrt(dist))
        indices.append(index)
        bc_coords.append(torch.stack([w0, w1, w2], 1))

    dists = torch.stack(dists, 0)
    indices = torch.stack(indices, 0)
    bc_coords = torch.stack(bc_coords, 0)

    return dists, indices, bc_coords



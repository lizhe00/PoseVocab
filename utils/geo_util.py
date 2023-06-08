import torch
import torch.nn.functional as F


def barycentric_coordinate(pts, face_vertices):
    """
    :param pts: (B, N, 3)
    :param face_vertices: (B, N, 3, 3)
    :return bc_coords: (B, N, 3)
    """
    vec0 = face_vertices[:, :, 0] - pts
    vec1 = face_vertices[:, :, 1] - pts
    vec2 = face_vertices[:, :, 2] - pts
    area0 = torch.linalg.norm(torch.cross(vec1, vec2), dim = -1)
    area1 = torch.linalg.norm(torch.cross(vec2, vec0), dim = -1)
    area2 = torch.linalg.norm(torch.cross(vec0, vec1), dim = -1)
    bc_coord = torch.stack([area0, area1, area2], -1)
    bc_coord = F.normalize(bc_coord, p = 1, dim = -1, eps = 1e-16)
    return bc_coord

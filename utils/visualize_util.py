import math
import numpy as np
import torch
import cv2 as cv


def to_HSV(c: torch.Tensor):
    """
    :param c: (N, 1) or (N,)
    :return: (N, 3)
    """
    h = (1 - c) * 240. / 60.
    x = 1 - torch.abs(h.to(torch.int64) % 2 + h - h.to(torch.int64) - 1.)

    rgb = torch.zeros((c.shape[0], 3)).to(c).to(torch.int64)

    cond_1 = torch.logical_and(h >= 0, h < 1)
    rgb[cond_1, 0] = 255
    rgb[cond_1, 1] = (x[cond_1] * 255).to(torch.int64)

    cond_2 = torch.logical_and(h >= 1, h < 2)
    rgb[cond_2, 0] = (x[cond_2] * 255).to(torch.int64)
    rgb[cond_2, 1] = 255

    cond_3 = torch.logical_and(h >= 2, h < 3)
    rgb[cond_3, 1] = 255
    rgb[cond_3, 2] = (x[cond_3] * 255).to(torch.int64)

    cond_4 = h >= 3
    rgb[cond_4, 1] = (x[cond_4] * 255).to(torch.int64)
    rgb[cond_4, 2] = 255

    rgb.clip_(0, 255)

    return rgb.to(torch.uint8)


# def calc_back_mv(dist):
#     rot_center = np.array([0, 0, dist], np.float32)
#     trans_mat = np.identity(4, np.float32)
#     trans_mat[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
#     trans_mat[:3, 3] = (np.identity(3) - trans_mat[:3, :3]) @ rot_center
#
#     return trans_mat


def calc_front_mv(object_center, tar_pos = np.array([0, 0, 2.0])):
    """
    calculate an extrinsic matrix for rendering the front of a 3D object
    under the assumption of fx,fy=550, cx,cy=256, img_h,img_w=512
    :param object_center: np.ndarray, (3,): the original center of the 3D object
    :param tar_pos: np.ndarray, (3,): the target center of the 3D object
    :return: extr_mat: np.ndarray, (4, 4)
    """
    extr_mat = np.identity(4, np.float32)
    extr_mat[:3, 3] = tar_pos - object_center
    return extr_mat


def calc_back_mv(object_center, tar_pos = np.array([0, 0, 2.0])):
    """
    calculate an extrinsic matrix for rendering the back of a 3D object
    under the assumption of fx,fy=550, cx,cy=256, img_h,img_w=512
    :param object_center: np.ndarray, (3,): the original center of the 3D object
    :param tar_pos: np.ndarray, (3,): the target center of the 3D object
    :return: extr_mat: np.ndarray, (4, 4)
    """
    mat_2origin = np.identity(4, np.float32)
    mat_2origin[:3, 3] = -object_center

    mat_rotY = np.identity(4, np.float32)
    mat_rotY[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]

    mat_2tarPos = np.identity(4, np.float32)
    mat_2tarPos[:3, 3] = tar_pos

    extr_mat = mat_2tarPos @ mat_rotY @ mat_2origin
    return extr_mat


def calc_free_mv(object_center, tar_pos = np.array([0, 0, 2.0]), rot_Y = 0.1, rot_X = 0., global_orient = None):
    """
    calculate an extrinsic matrix for rendering the back of a 3D object
    under the assumption of fx,fy=550, cx,cy=256, img_h,img_w=512
    :param object_center: np.ndarray, (3,): the original center of the 3D object
    :param tar_pos: np.ndarray, (3,): the target center of the 3D object
    :param rot_Y: float, rotation angle along Y axis
    :param global_orient: np.ndarray, global orientation of the 3D object
    :return: extr_mat: np.ndarray, (4, 4)
    """
    mat_2origin = np.identity(4, np.float32)
    mat_2origin[:3, 3] = -object_center

    mat_invGlobalOrient = np.identity(4, np.float32)
    if global_orient is not None:
        mat_invGlobalOrient[:3, :3] = cv.Rodrigues(np.array([math.pi, 0., 0.]))[0] @ np.linalg.inv(global_orient)
    else:
        mat_invGlobalOrient[:3, :3] = cv.Rodrigues(np.array([math.pi, 0., 0.]))[0]

    mat_rotY = np.identity(4, np.float32)
    mat_rotY[:3, :3] = cv.Rodrigues(np.array([0, rot_Y, 0]))[0]

    mat_2tarPos = np.identity(4, np.float32)
    mat_2tarPos[:3, 3] = tar_pos

    extr_mat = mat_2tarPos @ mat_rotY @ mat_invGlobalOrient @ mat_2origin
    return extr_mat

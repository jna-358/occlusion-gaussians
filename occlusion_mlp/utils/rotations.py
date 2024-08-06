import numpy as np
import torch


def get_rotaion_matrix_np(angle, dim):
    if dim == 0:
        return get_rotation_matrix_x_np(angle)
    elif dim == 1:
        return get_rotation_matrix_y_np(angle)
    else:
        return get_rotation_matrix_z_np(angle)


def get_rotation_matrix_z_np(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_matrix_y_np(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_matrix_x_np(angle):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_matrix_z_torch(angle):
    T_res = torch.eye(4)
    T_res[0, 0] = torch.cos(angle)
    T_res[0, 1] = -torch.sin(angle)
    T_res[1, 0] = torch.sin(angle)
    T_res[1, 1] = torch.cos(angle)
    return T_res


def get_rotation_matrix_y_torch(angle):
    T_res = torch.eye(4)
    T_res[0, 0] = torch.cos(angle)
    T_res[0, 2] = torch.sin(angle)
    T_res[2, 0] = -torch.sin(angle)
    T_res[2, 2] = torch.cos(angle)
    return T_res


def get_rotation_matrix_x_torch(angle):
    T_res = torch.eye(4)
    T_res[1, 1] = torch.cos(angle)
    T_res[1, 2] = -torch.sin(angle)
    T_res[2, 1] = torch.sin(angle)
    T_res[2, 2] = torch.cos(angle)
    return T_res


def get_rotation_matrix_torch(angle, dim):
    if dim == 0:
        return get_rotation_matrix_x_torch(angle)
    elif dim == 1:
        return get_rotation_matrix_y_torch(angle)
    else:
        return get_rotation_matrix_z_torch(angle)

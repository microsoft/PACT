# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from numba import njit
import cv2
import os


@njit(fastmath=True)
def gen_bbox_ij(s, height, xmin, ymin, scale):
    L = 0.5
    W = 0.3
    ds = np.array(
        [
            [
                [0.0, 0.0],
                [L / 2.0, W / 2.0],
                [L / 2.0, -W / 2.0],
                [-L / 2.0, W / 2.0],
                [-L / 2.0, -W / 2.0],
            ]
        ]
    )
    cos = np.cos(s[:, :, 2])
    sin = np.sin(s[:, :, 2])
    new_x = s[:, :, 0] + ds[:, :, 0] * cos - ds[:, :, 1] * sin
    new_y = s[:, :, 1] + ds[:, :, 0] * sin + ds[:, :, 1] * cos

    # return new_x, new_y
    i = height - (new_y - ymin) / scale
    j = (new_x - xmin) / scale
    return i, j


@njit(fastmath=True)
def compute_simple_dynamic(x, y, v, th, dt):
    new_x_sm = x + v * np.cos(th) * dt
    new_y_sm = y + v * np.sin(th) * dt
    new_th_sm = th
    return new_x_sm, new_y_sm, new_th_sm


@njit(fastmath=True)
def compute_complex_dynamic(x, y, v, th, beta, L, dt):
    sin_beta = np.sin(beta)
    new_th_big = th + 2 * v / L * sin_beta * dt
    new_x_big = x + L / (2 * sin_beta) * (np.sin(new_th_big + beta) - np.sin(th + beta))
    new_y_big = y + L / (2 * sin_beta) * (
        -np.cos(new_th_big + beta) + np.cos(th + beta)
    )
    return new_x_big, new_y_big, new_th_big


@njit(fastmath=True)
def merge_dynamic(
    mask, new_x_sm, new_y_sm, new_th_sm, new_th_big, new_x_big, new_y_big
):
    new_x = mask * new_x_sm + (1 - mask) * new_x_big
    new_y = mask * new_y_sm + (1 - mask) * new_y_big
    new_th = mask * new_th_sm + (1 - mask) * new_th_big
    return new_x, new_y, new_th


@njit(fastmath=True)
def world_to_map(x, y, height, width, xmin, ymin, scale):
    i = height - (y - ymin) / scale
    j = (x - xmin) / scale
    return i, j


@njit(fastmath=True)
def map_to_world_func(ij, height, width, xmin, ymin, scale):
    x = ij[..., 1] * scale + xmin
    y = (height - ij[..., 0]) * scale + ymin
    return x, y


@njit(fastmath=True)
def map_to_world_func(ij, height, width, xmin, ymin, scale):
    x = ij[..., 1] * scale + xmin
    y = (height - ij[..., 0]) * scale + ymin
    return x, y


def map_to_world(ij, height, width, xmin, ymin, scale):
    x, y = map_to_world_func(ij, height, width, xmin, ymin, scale)
    return np.stack((x, y), axis=-1)


@njit(fastmath=True)
def compute_min(cost):
    idx = np.argmin(cost)
    val = cost[idx]

    return [idx], [val]


@njit(fastmath=True)
def transform_center_to_minmax(h, w, cx, cy, scale):
    xmin = cx - w // 2 * scale
    xmax = cx + w // 2 * scale
    ymin = cy - h // 2 * scale
    ymax = cy + h // 2 * scale
    return xmin, xmax, ymin, ymax


@njit(fastmath=True)
def conv(img, kernel):
    h, w = img.shape
    m, n = kernel.shape
    pad_img = np.zeros((h + m - 1, w + n - 1))
    pad_img[m // 2 : m // 2 + h, n // 2 : n // 2 + w] = img
    new_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = np.sum(pad_img[i : i + m, j : j + n] * kernel)
    return new_img


@njit(fastmath=True)
def blur(img, inflate_factor):
    occ_mask = img == 0
    kernel = np.ones((inflate_factor * 2 + 1, inflate_factor * 2 + 1))
    new_occ_mask = conv(occ_mask, kernel)
    new_img = np.array(img)
    new_img[new_occ_mask > 0] = 0
    return new_img


def read_map(map_path, inflate_factor=0):
    real_map = cv2.imread(map_path, -1)
    if inflate_factor != 0:
        blur_img_path = map_path.replace(".png", "_blur%d.png" % inflate_factor)
        if os.path.exists(blur_img_path):
            real_map = cv2.imread(blur_img_path, -1)
        else:
            real_map = blur(real_map, inflate_factor=inflate_factor)
            cv2.imwrite(blur_img_path, real_map)

    occ_map = (real_map != 254).astype(dtype=np.float32)
    return real_map, occ_map


def randomly_choose_from(IJ_list):
    I_list, J_list = IJ_list
    N = I_list.shape[0]
    idx = np.random.choice(N, 1)[0].item()
    point = np.array([[I_list[idx], J_list[idx]]])
    return point  # shape (1, 2)

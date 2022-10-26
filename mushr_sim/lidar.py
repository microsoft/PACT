# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw
import range_libc


def polar_to_xy(radius, angle, shift=[0, 0]):
    x = shift[0] + radius * np.cos(angle)
    y = shift[1] + radius * np.sin(angle)
    return np.stack((x, y), axis=-1)


class Lidar:
    def __init__(
        self,
        map,
        xmin,
        xmax,
        ymin,
        ymax,
        n_readings,
        scan_method,
        d_max,
        img_dmax,
        theta_disc=None,
    ):
        self.n_readings = n_readings
        self.ranges = np.zeros(n_readings, dtype=np.float32)
        self.range_ones = np.ones(n_readings)
        self.theta_range = np.linspace(0, 2 * np.pi, n_readings, endpoint=False)
        self.theta_sym = np.linspace(-np.pi, np.pi, n_readings, endpoint=False)

        self.img_h, img_w = map.shape[:2]
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.img_scale = (self.ymax - self.ymin) / self.img_h
        self.d_max = d_max
        self.img_dmax = img_dmax
        self.buffer = np.zeros(n_readings, dtype=np.float32)

        libc_map = range_libc.PyOMap(map)
        if scan_method == "bl":
            self.algorithm = range_libc.PyBresenhamsLine(libc_map, max_range=d_max)
        elif scan_method == "rm":
            self.algorithm = range_libc.PyRayMarching(libc_map, max_range=d_max)
        elif scan_method == "rmgpu":
            self.algorithm = range_libc.PyRayMarchingGPU(libc_map, max_range=d_max)
        elif scan_method == "cddt":
            self.algorithm = range_libc.PyCDDTCast(
                libc_map, max_range=d_max, theta_disc=theta_disc
            )
        elif scan_method == "glt":
            self.algorithm = range_libc.PyGiantLUTCast(
                libc_map, max_range=d_max, theta_disc=theta_disc
            )

    def get_bev(self, scan, pcl_ego, render_type="pil", use_point=True):
        dmax = self.img_dmax
        valid = np.where(scan < dmax)[0]
        points = pcl_ego[valid]
        scale = 10
        w = 2 * dmax * scale
        h = 2 * dmax * scale
        xy = np.stack(
            (points[..., 0] * scale + h // 2, h - points[..., 1] * scale - w // 2),
            axis=-1,
        )

        if render_type == "pil":
            img = Image.new(mode="L", size=(w, h))
            draw = ImageDraw.Draw(img)
            if use_point:
                draw.point(list(xy.flatten()), fill=255)
            else:
                draw.polygon(list(xy.flatten()), fill=255)
            raw_img = np.array(img)
            # img.save("tmp-bev.png")
        elif render_type == "cv":
            xy = xy.astype(dtype=np.int32)
            img = np.zeros((h, w))
            if use_point:
                for i in range(xy.shape[0]):
                    cv2.circle(
                        img, (xy[i, 0], xy[i, 1]), radius=1, color=(255, 255, 255)
                    )
            else:
                cv2.fillPoly(img, pts=[xy], color=(255, 255, 255))
            cv2.imwrite("tmp-bev.png", img)
        elif render_type == "libc":
            self.algorithm.saveTrace(b"tmp-bev.png")
        else:
            fig = plt.Figure(figsize=(1, 1))
            ax = plt.gca()
            if use_point:
                plt.scatter(points[..., 0], points[..., 1])
            else:
                ax.add_patch(
                    Rectangle((-dmax, -dmax), 2 * dmax, 2 * dmax, color="gray")
                )
                ax.add_patch(
                    Polygon(points, facecolor="white", edgecolor="white", alpha=1)
                )
            ax.axis("off")
            plt.axis("scaled")
            ax.set_xlim(-dmax, dmax)
            ax.set_ylim(-dmax, dmax)
            fig.tight_layout()
            plt.savefig("tmp-bev.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
        # im = cv2.imread("tmp-bev.png", 0)
        im = raw_img
        return im

    def get_scan(self, pos, load_bev):
        # real coords -> pixel space
        Is = self.img_h - (pos[1] - self.ymin) / self.img_scale * self.range_ones
        Js = (pos[0] - self.xmin) / self.img_scale * self.range_ones
        theta = pos[2] + np.pi / 2 + self.theta_range
        input_data = np.stack((Is, Js, theta), axis=-1).astype(dtype=np.float32)

        self.algorithm.calc_range_many(input_data, self.buffer)
        scan = self.buffer * self.img_scale
        points = polar_to_xy(scan, theta - np.pi / 2, pos)
        pcl_ego = polar_to_xy(scan, self.theta_sym - np.pi / 2)

        # rasterization
        if load_bev:
            bev = self.get_bev(scan, pcl_ego)
        else:
            bev = None
        return scan, theta, points, bev, pcl_ego  # a series of readings

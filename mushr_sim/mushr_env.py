# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

from matplotlib.patches import Polygon
from scipy import ndimage

from mushr_sim.lidar import Lidar
from mushr_sim.planner import Planner
from mushr_sim.data_writer import DatasetWriter
from mushr_sim.sim_utils import (
    compute_complex_dynamic,
    compute_simple_dynamic,
    gen_bbox_ij,
    map_to_world,
    merge_dynamic,
    randomly_choose_from,
    read_map,
    transform_center_to_minmax,
)


class MushrSim:
    def __init__(self, args):
        np.random.seed(args.seed)
        self.args = args
        self.epi = 0
        self.tidx = 0
        self.history = []

        L = 0.5
        W = 0.3

        self.ds = np.array(
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

        # bloat to be square
        scale = 0.05
        cx = -32.925
        cy = -37.3

        print("map_path = %s" % args.map_path)

        # print all directories in a path python
        # rootdir = '/mnt/data'
        # for file in os.listdir(rootdir):
        #     d = os.path.join(rootdir, file)
        #     print(d)

        # rootdir = '/mnt/output'
        # for file in os.listdir(rootdir):
        #     d = os.path.join(rootdir, file)
        #     print(d)

        self.real_map, self.occ_map = read_map(
            args.map_path, inflate_factor=args.inflate_factor
        )
        self.height, self.width = self.real_map.shape

        self.xmin, self.xmax, self.ymin, self.ymax = transform_center_to_minmax(
            self.height, self.width, cx, cy, scale
        )

        self.scale = (self.ymax - self.ymin) / self.height

        # lidar configs
        self.lidar = Lidar(
            self.occ_map,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            n_readings=self.args.n_readings,
            scan_method=self.args.libc_method,
            d_max=self.args.d_max / self.scale,
            img_dmax=self.args.d_max,
            theta_disc=self.args.theta_disc,
        )

        # free indices configs
        self.dist_field = ndimage.distance_transform_edt(
            1 - self.occ_map, sampling=0.05
        )
        self.free_indices = np.where(self.dist_field > args.free_dist_thres)

        # basic info
        self.epi = 0
        self.tidx = 0
        self.history = []

    def reset(self, **kwargs):
        self.epi += 1
        self.tidx = 0
        self.history = []
        self.prev_observation = None
        self.prev_label = None

        # TODO make sure the sampled point is always in free space
        self.init_point_map = randomly_choose_from(self.free_indices)
        self.init_point_world = map_to_world(
            self.init_point_map,
            self.height,
            self.width,
            self.xmin,
            self.ymin,
            self.scale,
        )
        th_init = np.random.uniform(-np.pi, np.pi, (1, 1))
        state = np.append(self.init_point_world, th_init, axis=-1)

        i = 0
        while (
            i == 0 or np.linalg.norm(self.goal_point_world - self.init_point_world) < 5
        ):
            i += 1
            self.goal_point_map = randomly_choose_from(self.free_indices)
            self.goal_point_world = map_to_world(
                self.goal_point_map,
                self.height,
                self.width,
                self.xmin,
                self.ymin,
                self.scale,
            )

        # TODO update states
        self.start_state = np.array(state)
        self.state = state
        self.observation = self.get_observation(state)
        self.label = (
            np.min(self.observation["scan"], axis=-1) >= self.args.safe_threshold
        )
        return self.observation

    def dynamic(self, s, u):
        x = s[..., 0:1]
        y = s[..., 1:2]
        th = s[..., 2:3]
        v = u[..., 0:1]
        delta = u[..., 1:2]
        # https://github.com/prl-mushr/mushr_base/blob/d20b3d096a13f0e6c4120aed83159d716bbaaca7/mushr_base/src/racecar_state.py#L46
        L = 0.33

        mask = np.abs(delta) < 1e-2
        new_x_sm, new_y_sm, new_th_sm = compute_simple_dynamic(
            x, y, v, th, self.args.dt
        )
        beta = np.arctan(0.5 * np.tan(delta))
        beta[beta == 0] = 1e-2
        new_x_big, new_y_big, new_th_big = compute_complex_dynamic(
            x, y, v, th, beta, L, self.args.dt
        )
        new_x, new_y, new_th = merge_dynamic(
            mask, new_x_sm, new_y_sm, new_th_sm, new_th_big, new_x_big, new_y_big
        )
        new_s = np.concatenate((new_x, new_y, new_th), axis=-1)
        return new_s

    def get_observation(self, s):
        obs = {"scan": [], "theta": [], "points": [], "bev": [], "pcl": []}
        for i in range(s.shape[0]):
            scan, theta, points, bev, pcl = self.lidar.get_scan(
                s[i], load_bev=self.args.load_bev
            )
            obs["scan"].append(scan)
            obs["theta"].append(theta)
            obs["points"].append(points)
            obs["bev"].append(bev)
            obs["pcl"].append(pcl)

        for key in obs:
            obs[key] = np.stack(obs[key])
        return obs

    def step(self, action, state=None, **kwargs):
        if state is not None:
            old_state = state
        else:
            old_state = self.state
        new_state = self.dynamic(old_state, action)
        self.prev_observation = self.observation
        self.observation = self.get_observation(new_state)
        if "history_less" in kwargs and kwargs["history_less"] == True:
            self.hisotry = []
        else:
            self.history.append(old_state)
        self.state = new_state
        self.prev_label = self.label
        label = np.min(self.observation["scan"], axis=-1) >= self.args.safe_threshold
        self.label = label
        info = {"state": old_state, "new_state": new_state}
        if np.all(label == False):
            info["status"] = "crash"
        elif np.all(
            np.linalg.norm(old_state[:, :2] - self.goal_point_world, axis=-1)
            <= self.args.goal_reach_thres
        ):
            info["status"] = "reach"
        else:
            info["status"] = "normal"
        done = info["status"] in ["crash", "reach"]
        self.tidx += 1

        return self.observation, label, done, info

    def visualization(self, writer):
        print(self.epi, self.tidx)

        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(16, 16))

        plt.imshow(
            self.real_map,
            cmap="gray",
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
        )
        plt.plot(
            [x[0, 0] for x in self.history] + [self.state[0, 0]],
            [x[0, 1] for x in self.history] + [self.state[0, 1]],
            color="red",
            label="trajs",
        )
        plt.scatter(self.state[0, 0], self.state[0, 1], color="blue", label="ego", s=72)

        self.observation["scan"][0]
        x = self.state[0, 0]
        y = self.state[0, 1]
        points = self.observation["points"][0]

        plt.plot([x, points[0, 0]], [y, points[0, 1]], color="green", label="heading")

        plt.scatter(
            self.goal_point_world[0, 0],
            self.goal_point_world[0, 1],
            color="green",
            marker="v",
            s=72,
            label="goal_point",
        )
        plt.scatter(
            self.end_i_world[0], self.end_i_world[1], color="green", s=72, label="end_i"
        )
        plt.scatter(
            self.start_i_world[0, 0],
            self.start_i_world[0, 1],
            color="brown",
            s=72,
            label="start_i",
        )
        plt.scatter(
            self.reach_world[0, 0],
            self.reach_world[0, 1],
            color="brown",
            marker="v",
            s=72,
            label="reach_point",
        )

        # scan
        patch = Polygon(points, color="green", label="scan", alpha=0.2)
        ax = plt.gca()
        ax.add_patch(patch)
        plt.legend()

        viz_dir = args.root_dir
        viz_path = os.path.join(viz_dir, writer.header, "viz")
        os.makedirs(viz_path, exist_ok=True)

        plt.savefig(
            "%s/e%05d_t%05d.png" % (viz_path, self.epi, self.tidx),
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

    def check_collision_map(self, s):  # x in shape (M, 3)
        new_i, new_j = gen_bbox_ij(
            s[..., None, :], self.height, self.xmin, self.ymin, self.scale
        )
        new_i, new_j = new_i.astype(dtype=np.int32), new_j.astype(dtype=np.int32)
        new_i = np.clip(new_i, 0, self.height - 1)
        new_j = np.clip(new_j, 0, self.width - 1)
        return np.any(self.occ_map[new_i, new_j] == 1, axis=-1, keepdims=True)

    def world_to_map(self, x, y):
        i = self.height - (y - self.ymin) / self.scale
        j = (x - self.xmin) / self.scale
        return i, j

    def world_to_map_ij(self, xy):
        ij = np.array(xy)
        ij[..., 0] = self.height - (xy[..., 1] - self.ymin) / self.scale
        ij[..., 1] = (xy[..., 0] - self.xmin) / self.scale
        return ij

    def map_to_world(self, ij):
        x = ij[..., 1] * self.scale + self.xmin
        y = (self.height - ij[..., 0]) * self.scale + self.ymin
        return np.stack((x, y), axis=-1)


# input (N, N) -> edge distance from i to j (0, 1)
# return (N, N) -> to-go distance from i to j (0, ...., N)


def main():

    print("start")
    mushr_sim = MushrSim(args)

    obs = mushr_sim.reset()

    planner = Planner(mushr_sim, args)
    planner.reset()

    dbg_t1 = time.time()

    root_dir = args.root_dir
    ann_file_name = "test.json"

    len_sum = 0
    safe_sum = 0

    writer = DatasetWriter(
        root_dir,
        ann_file_name,
        args.dt,
        args.nt,
        mushr_sim.height,
        mushr_sim.width,
        mushr_sim.xmin,
        mushr_sim.xmax,
        mushr_sim.ymin,
        mushr_sim.ymax,
    )

    for ep_idx in range(args.n_episodes):
        if ep_idx != 0:
            obs = mushr_sim.reset()
            planner.reset()

        trajs = [obs]
        head_t = 0
        # storing data

        writer.initialize_episode(
            start=mushr_sim.start_state[0], goal=mushr_sim.goal_point_world[0]
        )

        done = False

        t1 = time.time()

        for i in range(args.nt):
            if done:
                break
            t1 = time.time()

            action = planner.plan(
                mushr_sim.state, obs, True, writer.header, args.root_dir
            )
            obs, label, done, info = mushr_sim.step(action)
            trajs.append(obs)

            pose = mushr_sim.history[-1][0]
            label_tmp = label.astype(dtype=float)[0]
            label = mushr_sim.prev_label.astype(dtype=float)[0]

            bev = mushr_sim.prev_observation["bev"]
            pcl = mushr_sim.prev_observation["pcl"][0]

            writer.record_timestep_data(action, pose, label, info, bev, pcl)

            # display the image with matplotlib plt for 0.1 sec of delay
            # if args.viz_bev:
            #     plt.imshow(bev[0,:,:])
            #     plt.show(block=False)
            #     # plt.show()
            #     # plt.pause(1.0)
            #     plt.close()

            if args.viz_last == False:
                if i > 0 and i % args.viz_freq == 0:
                    mushr_sim.visualization(writer)
                    print(i)
                    print(i % args.viz_freq)

            t3 = time.time()
            if i == 0:
                head_t = t3 - t1

        if args.viz_last:
            mushr_sim.visualization(writer)

        writer.finalize_episode()
        len_sum += writer.len_list[-1] - 1
        if label_tmp == 1:
            safe_sum += 1

    dbg_t2 = time.time()

    print(
        f"Average len: ({np.mean(writer.len_list):7.3f} meters / from a max of {args.nt:7.3f} timesteps cut-off)"
    )
    print(f"Safety stats:  ({safe_sum}/{args.n_episodes}) trajectories are safe")
    print(
        "Finished in %.4f seconds | average:%.4f  %.1fFPS"
        % (
            dbg_t2 - dbg_t1,
            (dbg_t2 - dbg_t1 - head_t) / (len_sum + 0.0001),
            (len_sum + 0.0001) / (dbg_t2 - dbg_t1 - head_t),
        )
    )

    writer.write_to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default="./maps/bravern_floor.png")
    parser.add_argument("--root_dir", type=str, default="./data")
    # parser.add_argument("--viz_dir", type=str, default="./viz")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--viz_freq", type=int, default=10000)
    parser.add_argument("--viz_last", action="store_true", default=False)
    # parser.add_argument("--viz_bev", action="store_true", default=False)
    parser.add_argument("--exp_name", type=str, default="")

    # Map configs
    parser.add_argument("--inflate_factor", type=int, default=2)
    parser.add_argument("--free_dist_thres", type=float, default=1.2)
    parser.add_argument("--goal_reach_thres", type=float, default=1.5)

    # Lidar configs
    parser.add_argument("--libc_method", type=str, default="rmgpu")
    parser.add_argument("--d_max", type=float, default=10)
    parser.add_argument("--n_readings", type=int, default=720)
    parser.add_argument("--theta_disc", type=int, default=100)
    parser.add_argument("--safe_threshold", type=float, default=0.05)

    # Planner configs
    parser.add_argument("--dist_map_path", type=str, default=None)
    parser.add_argument("--sampled_points", type=int, default=500)
    parser.add_argument("--k_neighbor", type=int, default=10)

    parser.add_argument("--n_segs", type=int, default=3)
    parser.add_argument("--seg_len", type=int, default=5)
    parser.add_argument("--branches", type=int, default=9)
    parser.add_argument("--vs", type=float, nargs="+", default=[2.5, 2.5])
    parser.add_argument("--n_vs", type=int, default=1)
    parser.add_argument("--deltas", type=float, nargs="+", default=[-0.34, 0.34])
    parser.add_argument("--n_deltas", type=int, default=9)

    parser.add_argument("--n_episodes", type=int, default=5)

    parser.add_argument("--sdf_thres", type=float, default=0.3)
    parser.add_argument("--goal_tol_dist", type=float, default=0.5)
    parser.add_argument("--dist_factor", type=float, default=10000)
    parser.add_argument("--coll_factor", type=float, default=100000.0)
    parser.add_argument("--goal_factor", type=float, default=200.0)
    parser.add_argument("--effort_factor", type=float, default=1.0)

    parser.add_argument("--suffix", type=str, default="")

    parser.add_argument("--load_bev", action="store_true", default=True)
    parser.add_argument("--numba", action="store_true", default=False)
    args = parser.parse_args()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))

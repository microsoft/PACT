# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import spatial
import matplotlib.tri as tri

from mushr_sim.sim_utils import world_to_map, map_to_world


def floyd_warshall(graph):
    dist = np.asarray(graph)
    np.fill_diagonal(dist, 0)
    for k in range(graph.shape[0]):
        dist = np.minimum(dist, dist[None, k, :] + dist[:, k, None])
    return dist


class Planner:
    def __init__(self, sim, args):
        self.args = args
        self.sim = sim
        self.init_map_data()

        self.n_seg = self.args.n_segs  # how many segs
        self.t_len = self.args.seg_len  # 28  # where each seg length
        self.branches = (
            self.args.n_vs * self.args.n_deltas
        )  # 1 for accel, and 9 for angle

        u_veloc = np.linspace(self.args.vs[0], self.args.vs[1], self.args.n_vs)
        u_delta = np.linspace(
            self.args.deltas[0], self.args.deltas[1], self.args.n_deltas
        )
        self.u_cmd = np.stack(np.meshgrid(u_veloc, u_delta), axis=-1).reshape(
            (1, 1, self.branches, 2)
        )

    def reset(self):
        self.m_end_idx = np.argmin(
            (self.map_points[:, 0] - self.sim.goal_point_map[0, 0]) ** 2
            + (self.map_points[:, 1] - self.sim.goal_point_map[0, 1]) ** 2,
            axis=-1,
        )

    # TODO generate the cost map
    def init_map_data(self):
        if self.args.dist_map_path is not None and self.args.dist_map_path != "None":
            # load from the data
            data = np.load(
                os.path.join(self.args.root_dir, self.args.dist_map_path),
                allow_pickle=True,
            )
            points = data["points"]
            map_dist = data["dist"]

            # left-over (fix this later)
            l2_dist = np.linalg.norm(points[None, :, :] - points[:, None, :], axis=-1)
            ind = np.argsort(l2_dist, axis=-1)[:, : self.args.k_neighbor]
            neighbor = points[ind]
        else:
            # sampled points
            safe_indices = np.where(self.sim.occ_map == 0)
            point_idx = np.random.choice(
                range(safe_indices[0].shape[0]), size=self.args.sampled_points
            )
            points = np.stack(
                (safe_indices[0][point_idx], safe_indices[1][point_idx]), axis=-1
            )

            # find neighbors
            l2_dist = np.linalg.norm(points[None, :, :] - points[:, None, :], axis=-1)
            ind = np.argsort(l2_dist, axis=-1)[:, : self.args.k_neighbor]
            neighbor = points[ind]

            # init neighbor dist
            graph = np.zeros((self.args.sampled_points, self.args.sampled_points))
            graph[
                np.array(
                    [
                        [i] * self.args.k_neighbor
                        for i in range(self.args.sampled_points)
                    ]
                ).flatten(),
                ind.flatten(),
            ] = 1
            graph[
                ind.flatten(),
                np.array(
                    [
                        [i] * self.args.k_neighbor
                        for i in range(self.args.sampled_points)
                    ]
                ).flatten(),
            ] = 1
            graph[graph != 1] = 65535

            # compute shortest path dist
            map_dist = floyd_warshall(graph)

        self.map_points = points
        self.map_neighbor = neighbor
        self.map_dist = map_dist
        self.kd_tree = spatial.KDTree(self.map_points)

        # (1300 * 1300, 2)
        self.hash_closest_points = np.zeros((self.sim.height, self.sim.width))
        all_is, all_js = np.meshgrid(
            np.linspace(0, self.sim.height - 1, self.sim.height),
            np.linspace(0, self.sim.width - 1, self.sim.width),
        )
        all_ijs = np.stack((all_is, all_js), axis=-1).reshape(
            (self.sim.height * self.sim.width, 2)
        )
        _, all_idx = self.kd_tree.query(all_ijs)
        self.hash_closest_points[
            all_ijs[:, 0].astype(dtype=np.int32), all_ijs[:, 1].astype(dtype=np.int32)
        ] = all_idx
        self.hash_closest_points = self.hash_closest_points.astype(dtype=np.int32)

    def viz_dist_map(self):
        # viz the map dist contour by first doing interpolation
        ngridx = 1000
        ngridy = 1000
        xi = np.linspace(0, self.sim.width, ngridx)
        yi = np.linspace(0, self.sim.height, ngridy)
        triang = tri.Triangulation(self.map_points[:, 1], self.map_points[:, 0])
        interpolator = tri.LinearTriInterpolator(triang, self.map_dist[:, 0])
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        fig = plt.figure(figsize=(12, 12))
        plt.imshow(self.sim.real_map, cmap="gray")  # plot the layout map
        plt.scatter(
            self.map_points[:, 1], self.map_points[:, 0], s=4
        )  # plot the sampled points

        ax = plt.gca()
        ax.contour(
            xi, yi, zi, levels=10, linewidths=0.5, colors="k"
        )  # plot the contour lines
        cntr = ax.contourf(xi, yi, zi, levels=10, alpha=0.3)  # plot the contour shades
        fig.colorbar(cntr, ax=ax)

        # plot the neighbors
        for ci, c in enumerate(["red", "blue", "yellow", "gray", "orange", "pink"]):
            plt.scatter(
                self.map_neighbor[ci, :, 1], self.map_neighbor[ci, :, 0], s=12, color=c
            )
        plt.scatter(self.map_points[0, 1], self.map_points[0, 0], color="red", s=20)
        plt.scatter(
            self.map_points[:, 1], self.map_points[:, 0], s=4, c=self.map_dist[:, 0]
        )
        plt.savefig(
            "/datadrive/viz/debug_mapdist0.png", bbox_inches="tight", pad_inches=0.1
        )
        plt.close()

    # TODO MPC-based RHC
    def plan(self, states, obses, normal=True, writer_header=None, viz_dir=None):
        time.time()

        goal_point_world = self.sim.goal_point_world

        s = np.array(states)[:, None]  # bs, 1, 3
        s_list = [
            np.tile(s, (1, self.branches**self.n_seg, 1))[:, :, None]
        ]  # [(bs, 9**3, 1, 3)]
        u_list = []
        bs, _, ndim = s.shape
        udim = 2

        # generate trajs
        for seg_i in range(self.n_seg):
            loc_s_list = []
            s = s[:, :, None]  # bs, #cand, 1, 3
            for ti in range(self.t_len):
                new_s = self.sim.dynamic(s, self.u_cmd)  # bs, branches, 3
                loc_s_list.append(new_s.reshape(bs, -1, ndim))
                s = new_s

            s = loc_s_list[-1]
            u_list.append(
                np.tile(
                    self.u_cmd[:, :, :, None, :],
                    (
                        bs,
                        self.branches ** (seg_i),
                        1,
                        self.branches ** (self.n_seg - 1 - seg_i),
                        1,
                    ),
                ).reshape((bs, self.branches**self.n_seg, udim))
            )
            loc_s_list = np.stack(loc_s_list, axis=2)  # (bs, #cand, T, 3)
            loc_s_list = np.tile(
                loc_s_list[:, :, None],
                (1, 1, self.branches ** (self.n_seg - 1 - seg_i), 1, 1),
            ).reshape((bs, -1, self.t_len, ndim))
            s_list.append(loc_s_list)

        s_list = np.concatenate(s_list, axis=2)  # (bs, #cand, T, 3)
        u_list = np.stack(u_list, axis=2)  # (bs, #cand, nsegs, 2)

        # COLLISION COST
        n_cands = s_list.shape[1]
        T = s_list.shape[2]
        si_list, sj_list = world_to_map(
            s_list[..., 0],
            s_list[..., 1],
            self.sim.height,
            self.sim.width,
            self.sim.xmin,
            self.sim.ymin,
            self.sim.scale,
        )
        si_list = np.clip(si_list, 0, self.sim.width - 1)
        sj_list = np.clip(sj_list, 0, self.sim.height - 1)
        dist_cost = self.sim.dist_field[
            si_list.astype(dtype=np.uint32), sj_list.astype(dtype=np.uint32)
        ]

        dist_cost = (
            np.any(dist_cost < self.args.sdf_thres, axis=-1) * self.args.dist_factor
        )
        coll = self.sim.check_collision_map(s_list.reshape((-1, ndim))).reshape(
            (bs, n_cands, T)
        )
        coll_cost = np.any(coll, axis=-1)
        coll_cost = coll_cost * self.args.coll_factor + dist_cost

        # GOAL COST
        time.time()
        s_reach = s_list[:, :, -1, :2]
        s_reach_i, s_reach_j = world_to_map(
            s_reach[..., 0],
            s_reach[..., 1],
            self.sim.height,
            self.sim.width,
            self.sim.xmin,
            self.sim.ymin,
            self.sim.scale,
        )

        s_reach_i = np.clip(s_reach_i, 0, self.sim.width - 1)
        s_reach_j = np.clip(s_reach_j, 0, self.sim.height - 1)

        self.sim.goal_point_map[0, 0]
        self.sim.goal_point_map[0, 1]

        _, m_start_idx = self.kd_tree.query(np.stack((s_reach_i, s_reach_j), axis=-1))

        m_end_idx = self.m_end_idx

        start_to_end = self.map_dist[m_start_idx, m_end_idx]
        s_to_d_world = np.linalg.norm(s_reach - goal_point_world[0], axis=-1)
        s_to_d_world[s_to_d_world < self.args.goal_tol_dist] = 0
        s_to_d_world /= 20

        goal_cost = start_to_end * (start_to_end > 1) + s_to_d_world * (
            start_to_end <= 1
        )
        goal_cost = goal_cost * self.args.goal_factor

        time.time()

        # MIN CONTROL EFFORT COST
        effort_cost = np.sum(np.linalg.norm(u_list[..., 1:2], axis=-1), axis=-1)
        effort_cost = effort_cost * self.args.effort_factor

        # TOTAL COST
        cost = coll_cost + goal_cost + effort_cost

        # pick the one that fits  (bs, 1)
        if normal:
            min_idx = np.argmin(cost, axis=1)
        else:
            min_idx = np.random.choice(cost.shape[1], 1)

        if self.args.viz_last == False or self.sim.tidx >= self.args.nt:
            if self.sim.tidx % self.args.viz_freq == 0 and self.sim.tidx > 0:
                plt.figure(figsize=(16, 16))
                plt.imshow(
                    self.sim.real_map,
                    cmap="gray",
                    extent=[self.sim.xmin, self.sim.xmax, self.sim.ymin, self.sim.ymax],
                )
                plt.plot(
                    [x[0, 0] for x in self.sim.history] + [self.sim.state[0, 0]],
                    [x[0, 1] for x in self.sim.history] + [self.sim.state[0, 1]],
                    color="red",
                    label="trajs",
                )
                for i in range(s_list.shape[1]):
                    plt.plot(
                        s_list[0, i, :, 0],
                        s_list[0, i, :, 1],
                        color="purple",
                        label="rhc" if i == 0 else None,
                        linewidth=0.5,
                    )
                plt.plot(
                    s_list[0, min_idx[0], :, 0],
                    s_list[0, min_idx[0], :, 1],
                    color="blue",
                    label="planned",
                )
                viz_path = os.path.join(viz_dir, writer_header, "viz")
                os.makedirs(viz_path, exist_ok=True)
                plt.savefig(
                    "%s/debug_%05d.png" % (viz_path, self.sim.tidx),
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                plt.close()

        time.time()

        # use only the first one
        v = u_list[:, min_idx, 0, 0]
        delta = u_list[:, min_idx, 0, 1]

        # for visualization
        # self.sim.goal_point_world = goal_point_world
        self.sim.end_i_world = map_to_world(
            self.map_points[m_end_idx],
            self.sim.height,
            self.sim.width,
            self.sim.xmin,
            self.sim.ymin,
            self.sim.scale,
        )
        self.sim.start_i_world = map_to_world(
            self.map_points[m_start_idx[0, min_idx]],
            self.sim.height,
            self.sim.width,
            self.sim.xmin,
            self.sim.ymin,
            self.sim.scale,
        )
        self.sim.reach_world = s_reach[0, min_idx]

        time.time()

        # print("PLAN init:%.5f rollout:%.5f coll:%.5f goal:%.5f eff:%.5f pick:%.5f print:%.5f assign:%.5f"%(ttt2-ttt1, ttt3-ttt2, ttt4-ttt3, ttt5-ttt4, ttt6-ttt5, ttt7-ttt6, ttt8-ttt7, ttt9-ttt8))

        return np.concatenate((v, delta), axis=-1)

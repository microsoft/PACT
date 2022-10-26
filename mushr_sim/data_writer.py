# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from datetime import datetime, timedelta
import os
import json
import random


class DatasetWriter:
    def __init__(
        self, root_dir, ann_file_name, dt, nt, height, width, xmin, xmax, ymin, ymax
    ):
        self.dt = dt
        self.nt = nt

        self.ep_idx = 0
        self.ep_step = 0
        self.video = {}
        self.len_list = []

        self.height = height
        self.width = width
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.root_dir = root_dir
        self.ann_file_name = ann_file_name

        self.time_base = datetime(1970, 1, 1)
        self.time_now = datetime.now()
        self.header = str((self.time_now - self.time_base).total_seconds())

    def li(self, data):
        return data.tolist()

    def initialize_episode(self, start, goal):
        self.ep_idx += 1
        self.ep_step = 0
        event_id = self.header + "_%d" % (self.ep_idx)
        self.video[event_id] = {"data": [], "metadata": {}}

        img_path = "%s/processed_images_occupancy2/%d/all_images.npy" % (
            self.header,
            self.ep_idx,
        )
        pcl_path = "%s/processed_images_occupancy2/%d/all_pcls.npy" % (
            self.header,
            self.ep_idx,
        )
        img_full_path = os.path.join(self.root_dir, img_path)
        pcl_full_path = os.path.join(self.root_dir, pcl_path)

        self.video[event_id]["metadata"]["all_images_path"] = img_path
        self.video[event_id]["metadata"]["all_pcls_path"] = pcl_path
        self.all_data_path = "%s/processed_images_occupancy2/%d/all_data.npz" % (
            self.header,
            self.ep_idx,
        )
        self.video[event_id]["metadata"]["video_len"] = self.nt

        os.makedirs(os.path.dirname(img_full_path), exist_ok=True)
        os.makedirs(os.path.dirname(pcl_full_path), exist_ok=True)

        # initialize empty list with the data
        self.image_list = []
        self.pcl_list = []
        self.total_lines = []

        self.actions = []
        self.poses = []
        self.labels = []

        self.start = self.li(start[0])
        self.goal = self.li(goal[0])

        self.event_id = event_id

    def record_timestep_data(self, action, pose, label, info, bev, pcl):
        frame = {}

        frame["dir_name"] = self.header
        frame["episode_number"] = self.ep_idx
        frame["img_number"] = self.ep_step
        frame["start"] = self.start
        frame["goal"] = self.goal
        frame["time"] = str(
            (
                self.time_now
                + timedelta(seconds=self.dt * self.ep_step)
                - self.time_base
            ).total_seconds()
        )

        frame["action"] = action[0, 1]
        frame["action_full"] = self.li(action[0])
        frame["pose"] = self.li(pose)

        # frame["img_bev_rel_path"] = "%s/bev/%d/%d.png" % (
        #     frame["dir_name"],
        #     frame["episode_number"],
        #     frame["img_number"],
        # )
        # frame["pcl_rel_path"] = "%s/pcl/%d/%d.npy" % (
        #     frame["dir_name"],
        #     frame["episode_number"],
        #     frame["img_number"],
        # )
        frame["label"] = self.li(label)

        self.video[self.event_id]["data"].append(frame)

        self.actions.append(frame["action"])
        self.poses.append(frame["pose"])
        self.labels.append(frame["label"])

        # img_path = os.path.join(self.root_dir, frame["img_bev_rel_path"])
        # os.makedirs(os.path.dirname(img_path), exist_ok=True)

        bev_tmp = np.transpose(bev, (1, 2, 0))
        self.image_list.append(bev_tmp)
        self.pcl_list.append(pcl)

        self.label = frame["label"]
        self.ep_step += 1

    def finalize_episode(self):
        video_len = len(self.video[self.event_id]["data"])
        self.video[self.event_id]["metadata"]["video_len"] = video_len

        self.len_list.append(video_len)
        # print("video len: ", video_len)

        image_list = np.stack(self.image_list, axis=0)
        pcl_list = np.stack(self.pcl_list, axis=0)
        np.save(
            os.path.join(
                self.root_dir, self.video[self.event_id]["metadata"]["all_images_path"]
            ),
            image_list.astype(np.float32),
        )
        np.save(
            os.path.join(
                self.root_dir, self.video[self.event_id]["metadata"]["all_pcls_path"]
            ),
            pcl_list.astype(np.float32),
        )

        actions = np.array(self.actions)[:, None]
        poses = np.array(self.poses)
        labels = np.array(self.labels)[:, None]
        np.savez_compressed(
            os.path.join(self.root_dir, self.all_data_path),
            actions=actions,
            poses=poses,
            labels=labels,
        )

        self.total_lines.append(
            "%s %d %d %.1f\n" % (self.header, self.ep_idx, video_len, self.label)
        )

    def write_to_file(self):
        video_keys = list(self.video.keys())
        vkey_to_i = {k: ki for ki, k in enumerate(video_keys)}
        random.shuffle(video_keys)
        # whole_ann = {"type": "mushr_sim_pretrain", "ann": video}
        split = int(len(video_keys) * 0.8)
        train_ann = {
            "type": "mushr_sim_pretrain",
            "ann": {k: self.video[k] for k in sorted(video_keys[:split])},
        }
        val_ann = {
            "type": "mushr_sim_pretrain",
            "ann": {k: self.video[k] for k in sorted(video_keys[split:])},
        }
        # train_lines = [self.total_lines[vkey_to_i[k]] for k in sorted(video_keys[:split])]
        # val_lines = [self.total_lines[vkey_to_i[k]] for k in sorted(video_keys[split:])]
        # val_dang_ratio = sum(val_dang_list) / len(val_dang_list)
        with open(os.path.join(self.root_dir, self.header, "train_ann.json"), "w") as f:
            json.dump(train_ann, f, indent=2)

        with open(os.path.join(self.root_dir, self.header, "val_ann.json"), "w") as f:
            json.dump(val_ann, f, indent=2)

        # with open(os.path.join(self.root_dir, self.header, "train_ann.txt"), "w") as f:
        #     for line in train_lines:
        #         f.write(line)

        # with open(os.path.join(self.root_dir, self.header, "val_ann.txt"), "w") as f:
        #     for line in val_lines:
        #         f.write(line)

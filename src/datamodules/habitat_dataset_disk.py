# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import json
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from utils.geom_utils import calc_relative_pose


class HabitatVideoDataset(torch.utils.data.Dataset):
    """Imitation learning (IL) video dataset in mushr lidar."""

    def __init__(
        self,
        dataset_dir,
        dataset_type,
        ann_file_name,
        transform,
        state_type="hits",
        clip_len=8,
        flatten_img=False,
        load_gt_map=False,
        rebalance_samples=False,
        num_bins=15,
        map_recon_dim=64,
        dataset_fraction=1.0,
    ):
        self.dataset_dir = dataset_dir
        self.clip_len = clip_len
        self.flatten_img = flatten_img
        self.state_type = state_type
        self.local_map_size_px = 100  # self.local_map_size_m/self.map_res
        self.map_recon_dim = map_recon_dim
        self.dataset_fraction = dataset_fraction
        self.transform = transform
        self.dataset_type = dataset_type
        self.maps = {}

        self.load_gt_map = load_gt_map

        # Load annotation file.
        # Format:
        # {
        #     'type': 'video',
        #     'ann': {
        #         'video name 1': [
        #             {'timestamp': timestamp, 'img_rel_path': img_rel_path, 'flow_rel_path': flow_rel_path, 'vel': vel},
        #             ...
        #         ]
        #         ...
        #     }
        # }

        ann_path = os.path.join(dataset_dir, ann_file_name)
        with open(ann_path) as f:
            self.ann = json.load(f)
        assert self.ann["type"] == "habitat_pointnav"

        self.datadir_prefix = "pointnav_hm3d_train_10_percent"

        max_num_videos = int(dataset_fraction * len(self.ann["ann"]))

        # Generate clip indices. Format: (video name, start frame index).
        self.clip_indices = []
        self.clip_actions = []
        self.video_names = []

        if self.dataset_type == "train":
            data_start_idx = 0
            data_end_idx = int(0.8 * max_num_videos)
            print(data_start_idx, data_end_idx)

        elif self.dataset_type == "test":
            data_start_idx = int(0.8 * max_num_videos)
            data_end_idx = max_num_videos
            print(data_start_idx, data_end_idx)

        import itertools

        self.ann_sliced = dict(
            itertools.islice(self.ann["ann"].items(), data_start_idx, data_end_idx)
        )

        for video_idx, video_name in enumerate(self.ann_sliced):
            if video_idx > data_end_idx - data_start_idx:
                break

            video = self.ann_sliced[video_name]

            self.video_names.append(video_name)
            dir_name = video["data"][0]["dir_name"]
            episode_num = video["data"][0]["episode_number"]

            # episode_dir_name = os.path.join(dataset_dir, 'pointnav_hm3d_train_10_percent', dir_name, 'rgb_fpv_obs', str(episode_num))
            # print("Episode dir name: ", episode_dir_name)

            # HACK to avoid bad episodes in data
            if dir_name == "pointnav_hm3d_2050_2100":
                if episode_num in [43, 44, 45, 48]:
                    continue

            if len(video["data"]) >= clip_len:
                # print(len(video['data']))

                # for i in range(0, len(video['data']) - clip_len + 1):
                #    rgb_path = os.path.join(dataset_dir, 'pointnav_hm3d_train_10_percent', dir_name, 'rgb_fpv_obs', str(episode_num), str(i) + '.png')
                #    if not os.path.exists(rgb_path):
                #        print("RGB path does not exist: {}".format(rgb_path))

                for start_frame_index in range(len(video["data"]) - clip_len + 1):
                    self.clip_indices.append((video_name, start_frame_index))
                    self.clip_actions.append(
                        video["data"][start_frame_index + clip_len - 1]["action"]
                    )

            if self.load_gt_map:
                # TODO: append GT map to the main dictionary
                gt_map_file_name = os.path.join(
                    dataset_dir,
                    "pointnav_hm3d_train_10_percent",
                    dir_name,
                    "all_rgb",
                    str(episode_num),
                    "world_map.png",
                )
                # gt_map_name = os.path.join(dataset_dir, gt_map_file_name)
                orig_map = cv2.imread(gt_map_file_name, 0)
                orig_map[
                    orig_map == 150
                ] = 0  # make the gray area become black, as occupied space

                orig_map = (orig_map / 255.0 - 1.0).astype(
                    np.float32
                )  # invert colors and normalize to -1,1 range
                # change the color scheme to 0 free to 1 occupied
                orig_map[orig_map > -1] = 1

                self.maps[video_name] = orig_map

        # Other settings.
        self.num_bins = num_bins
        # create sampling classes if needed
        self.rebalance_samples = rebalance_samples
        if self.rebalance_samples:
            self.clip_actions = np.array(self.clip_actions)
            # create bins for classes
            bins = np.linspace(
                self.clip_actions.min(), self.clip_actions.max(), num=self.num_bins
            )
            # insert indices in each bin
            binplace = np.digitize(self.clip_actions, bins, right=True)
            self.binned_indices = []
            sum = 0
            for i in range(self.num_bins):
                self.binned_indices.append(np.where(binplace == i)[0])
                print(len(self.binned_indices[i]))
                sum += len(self.binned_indices[i])
            print(sum)

    def __len__(self):
        return len(self.clip_indices)

    def crop_map_new(self, center_pose, current_map):
        cx = center_pose[0]
        cy = center_pose[1]

        rect = (
            (cx, cy),
            (self.local_map_size_px, self.local_map_size_px),
            90 - math.degrees(center_pose[2]),
        )
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(current_map, M, (width, height))
        return warped

    def process_relative_poses_deltas_incremental(self, poses):
        num_poses = poses.shape[0]
        new_poses = np.zeros(shape=(num_poses - 1, 3))
        # iterate but skip the first point (because it outputs a zero delta)
        for i in range(num_poses - 1):
            new_poses[i, :] = calc_relative_pose(poses[i], poses[i + 1])
        return new_poses

    def get_full_eval_trajectory(self, traj_index=None):
        # process the index, cut off at the max length of trajectories
        vid_len = 0

        while vid_len < 2 * self.clip_len:
            if traj_index is None:
                traj_index = torch.randint(len(self.video_names), (1,)).item()
            else:
                traj_index = min(len(self.video_names) - 1, traj_index)
            # get all the clip indices for that video
            video_name = self.video_names[traj_index]
            video = self.ann_sliced[video_name]["data"]
            vid_len = len(video)
            traj_index = None

        items = []
        for start_frame_index in range(len(video) - self.clip_len + 1):
            items.append(self.get_item_internal(video_name, start_frame_index))

        return items

    def __getitem__(self, index):

        # Transform the index into a weighted sample
        if self.rebalance_samples:
            idx_bin = np.random.randint(len(self.binned_indices))  # choose a bin
            idx_clip = np.random.randint(
                len(self.binned_indices[idx_bin])
            )  # choose random element from bin
            index = self.binned_indices[idx_bin][idx_clip]

        # Get annotation.
        video_name, start_frame_index = self.clip_indices[index]

        return self.get_item_internal(video_name, start_frame_index)

    def get_item_internal(self, video_name, start_frame_index):

        # start_time = time.time()

        states = []
        acts = []
        poses = []
        for frame_index in range(start_frame_index, start_frame_index + self.clip_len):
            frame_ann = self.ann_sliced[video_name]["data"][frame_index]

            # load the state info
            # Load image.
            img_rel_path = frame_ann["rgb_obs_rel_path"]
            img_rel_path = self.datadir_prefix + "/" + img_rel_path
            img_path = os.path.join(self.dataset_dir, img_rel_path)

            if self.flatten_img:
                img = cv2.imread(img_path)[:, :, 0]
                img = 2.0 * img / 254 - 1
                img = np.array(img, dtype=np.float32).flatten()
                states.append(img)
            else:
                img = Image.open(img_path)
                states.append(self.transform(img))

            act = frame_ann["action"]
            act = np.array(act, dtype=int)
            acts.append(act)

            pose = frame_ann["pose"]

            new_pose = np.zeros(3)
            new_pose[0] = pose[0]
            new_pose[1] = pose[2]
            new_pose[-1] = pose[4]
            pose = np.array(new_pose, dtype=np.float32)
            poses.append(pose)

            # get the middle pose
            # subtract 1 from clip len because had to add one for localization
            if frame_index == start_frame_index + int((self.clip_len - 1) / 2) - 1:
                pass

        # reading_time = time.time()
        # print(f"reading_time after entire loop = {reading_time-start_time}")

        # process the state information
        # if self.state_type == "hits" or self.state_type == "occupancy":
        if self.state_type == "rgb":
            if self.flatten_img:
                img_seq = [torch.from_numpy(item) for item in states]
                states = torch.stack(img_seq, 0)  # shape: [C] -> [D, C]
            else:
                states = torch.stack(states, dim=0)  # Shape: [C,H,W] -> [N,C,H,W].
        elif self.state_type == "pcl":
            states = [torch.from_numpy(item) for item in states]
            states = torch.stack(states, dim=0)  # Shape: [C,2] -> [N,C,2].
        else:
            raise RuntimeError("Data type not supported!")

        # torch_processing_time = time.time()
        # print(f"torch_processing_time = {torch_processing_time-reading_time}")

        # process the action information
        act_seq = [torch.from_numpy(item) for item in acts]
        act_seq = torch.stack(act_seq, dim=0).long()  # actions are discrete

        poses = np.asarray(poses)
        poses = self.process_relative_poses_deltas_incremental(poses)

        pose_seq = torch.from_numpy(poses).float()

        # crop the GT map from the larger bravern map and add it to the dict
        if self.load_gt_map:
            col_x, row_y, ang = frame_ann["grid_pose"]
            # curr_map = self.ann['ann'][video_name]['meta_data']['gt_map_img']

            gt_map = self.crop_map_new([row_y, col_x, ang], self.maps[video_name])
            # gt_map = self.crop_map([col_x, row_y, middle_pose[2]]) # returned in the -1 (free) to 1 (occ) range
            gt_map = resize(gt_map, (self.map_recon_dim, self.map_recon_dim))
            # gt_map[gt_map>-0.99] = 1
            # gt_map[gt_map<0] = -1
            gt_map = torch.tensor(gt_map).float()
            # gt_map = (gt_map+1.0)/2.0  # go to the 0-1 scale to match the lidar images
        else:
            gt_map = 0  # some useless value just to have something because we can't pass None types later

        # mount the dict
        # not used now: [:-1] means that we return all elements in order, but drop the last one because we had one extra from seq length
        # pose_seq will have one element less because it's the deltas btw states
        item = {"state": states, "action": act_seq, "pose": pose_seq, "gt_map": gt_map}
        return item

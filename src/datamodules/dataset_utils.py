# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from utils.geom_utils import calc_relative_pose


def norm_angle(angle):
    # normalize all actions
    act_max = 0.38
    act_min = -0.38
    return 2.0 * (angle - act_min) / (act_max - act_min) - 1.0


def process_relative_poses_deltas_incremental(poses):
    num_poses = poses.shape[0]
    new_poses = np.zeros(shape=(num_poses - 1, 3))
    # iterate but skip the first point (because it outputs a zero delta)
    for i in range(num_poses - 1):
        new_poses[i, :] = calc_relative_pose(poses[i], poses[i + 1])
    return new_poses

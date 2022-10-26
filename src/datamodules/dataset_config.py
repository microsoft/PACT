# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# =================== from old dataloader===============================
from dataclasses import dataclass


@dataclass
class FileParams:
    dataset_dir: str
    train_ann_file_name: str
    val_ann_file_name: str


@dataclass
class MapParams:
    local_map_size_m: float
    map_center: tuple
    map_res: float
    gt_map_file_name: str
    load_gt_map: bool
    map_recon_dim: int


@dataclass
class TrainParams:
    clip_len: int
    rebalance_samples: bool
    img_dim: int
    num_bins: int
    flatten_img: bool
    state_tokenizer: str
    load_gt_map: False

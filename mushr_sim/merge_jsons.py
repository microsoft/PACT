# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import json
from os.path import join as ospj
import time

WRITE_JSON = True


print("Start")

t1 = time.time()

root = "./data"

print("Root is", root)
sub_ds = sorted(os.listdir(root))
print("Sub_directories are", sub_ds)

big_data = {
    "train": {"type": "mushr_sim_pretrain", "ann": {}},
    "val": {"type": "mushr_sim_pretrain", "ann": {}},
}

if not WRITE_JSON:
    train_f = open(ospj(root, "train_list.txt"), "w")
    val_f = open(ospj(root, "val_list.txt"), "w")

cnt = 0
file_list = {"train": [], "val": []}

len_total_train = 0
len_total_val = 0

for sub_d in sub_ds:
    if os.path.isdir(ospj(root, sub_d)):
        print("Open", cnt, ospj(root, sub_d))
        cnt += 1
        if WRITE_JSON:
            for split in ["train", "val"]:
                file_path = os.path.join(root, sub_d, split + "_ann.json")
                if os.path.exists(file_path):
                    with open(file_path) as f:
                        data = json.load(f)
                    for video in data["ann"]:
                        big_data[split]["ann"][video] = {
                            "data": data["ann"][video]["data"],
                            "metadata": data["ann"][video]["metadata"],
                        }
                        if split == "train":
                            len_total_train += data["ann"][video]["metadata"][
                                "video_len"
                            ]
                        else:
                            len_total_val += data["ann"][video]["metadata"]["video_len"]
        else:
            file_path_train = os.path.join(root, sub_d, "train_ann.json")
            file_path_val = os.path.join(root, sub_d, "train_ann.json")
            if os.path.exists(file_path_train) and os.path.exists(file_path_val):
                train_f.write(ospj(root, sub_d, "train_ann.json") + "\n")
                val_f.write(ospj(root, sub_d, "val_ann.json") + "\n")

if WRITE_JSON:
    print("Finished storing to the dict object")
    for split in ["train", "val"]:
        with open(ospj(root, split + "_ann_merged.json"), "w") as f:
            json.dump(big_data[split], f, indent=1)
    print("Finished writing to json file")
else:
    train_f.close()
    val_f.close()

t2 = time.time()

print(
    "Finished in %.4f seconds, with %d train and %d val episodes"
    % (t2 - t1, len(big_data["train"]["ann"]), len(big_data["val"]["ann"]))
)

print(
    "Contains %d train and %d val state-action pairs" % (len_total_train, len_total_val)
)

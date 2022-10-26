# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from src.datamodules.mushr_dataset_memory import MushrVideoDatasetPreload
from torch.utils.data import DataLoader


class MushrDataModule(LightningDataModule):
    # Parameters data.xxx from the config yml get mapped here
    def __init__(
        self, file_params, map_params, train_params, batch_size, num_workers, pin_memory
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.file_params = file_params
        self.train_params = train_params
        self.map_params = map_params

        if self.train_params["state_tokenizer"] == "conv2D":
            self.train_transform = T.Compose(
                [T.Resize(self.hparams["img_dim"]), T.ToTensor()]
            )  # Only resize.
        elif self.train_params["state_tokenizer"].startswith("resnet"):
            self.train_transform = T.Compose([T.ToTensor()])
        elif self.train_params["state_tokenizer"] == "pointnet":
            self.train_transform = None
            self.state_type = "pcl"
        else:
            print("Not supported!")

    def setup(self, stage=None):
        print(self.train_params)
        self.train_dataset = MushrVideoDatasetPreload(
            dataset_dir=self.file_params["dataset_dir"],
            ann_file_name=self.file_params["train_ann_file_name"],
            transform=self.train_transform,
            gt_map_file_name=self.map_params["gt_map_file_name"],
            local_map_size_m=self.map_params["local_map_size_m"],
            map_center=self.map_params["map_center"],
            map_res=self.map_params["map_res"],
            state_type=self.state_type,
            clip_len=self.train_params["clip_len"],
            flatten_img=self.train_params["flatten_img"],
            load_gt_map=self.map_params["load_gt_map"],
            rebalance_samples=self.train_params["rebalance_samples"],
            num_bins=self.train_params["num_bins"],
            map_recon_dim=self.map_params["map_recon_dim"],
            dataset_fraction=self.train_params["train_dataset_fraction"],
        )
        self.val_dataset = MushrVideoDatasetPreload(
            dataset_dir=self.file_params["dataset_dir"],
            ann_file_name=self.file_params["val_ann_file_name"],
            transform=self.train_transform,
            gt_map_file_name=self.map_params["gt_map_file_name"],
            local_map_size_m=self.map_params["local_map_size_m"],
            map_center=self.map_params["map_center"],
            map_res=self.map_params["map_res"],
            state_type=self.state_type,
            clip_len=self.train_params["clip_len"],
            flatten_img=self.train_params["flatten_img"],
            load_gt_map=self.map_params["load_gt_map"],
            rebalance_samples=self.train_params["rebalance_samples"],
            num_bins=self.train_params["num_bins"],
            map_recon_dim=self.map_params["map_recon_dim"],
            dataset_fraction=self.train_params["val_dataset_fraction"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

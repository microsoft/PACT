# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from src.datamodules.habitat_dataset_disk import HabitatVideoDataset
from torch.utils.data import DataLoader


class HabitatDataModule(LightningDataModule):
    # Parameters data.xxx from the config yml get mapped here
    def __init__(
        self,
        file_params,
        map_params,
        train_params,
        num_workers,
        batch_size,
        pin_memory,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.file_params = file_params
        self.train_params = train_params
        self.map_params = map_params

        # ! check with sai
        self.state_type = "rgb"

        if self.train_params["state_tokenizer"] == "conv2D":
            self.train_transform = T.Compose(
                [T.Resize(self.train_params["img_dim"]), T.ToTensor()]
            )  # Only resize.
        elif self.train_params["state_tokenizer"].startswith("resnet"):
            self.train_transform = T.Compose([T.ToTensor()])
        elif self.train_params["state_tokenizer"] == "pointnet":
            self.train_transform = None
            self.state_type = "pcl"
        else:
            print("Not supported!")

        # train/val transform
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(self.train_params["img_dim"], scale=(0.8, 1.0)),
                # transforms.RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, consistent=False)], p=0.8),
                T.ToTensor(),
                normalize,
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize(
                    self.train_params["img_dim"]
                ),  # Note: It is different from the ImageNet example, which resizes image to 256x256,
                T.ToTensor(),  #       and then center crops it to 224x224.
                normalize,
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = HabitatVideoDataset(
            dataset_dir=self.file_params["dataset_dir"],
            dataset_type="train",
            ann_file_name=self.file_params["train_ann_file_name"],
            transform=self.train_transform,
            state_type=self.state_type,
            clip_len=self.train_params["clip_len"],
            flatten_img=self.train_params["flatten_img"],
            load_gt_map=self.map_params["load_gt_map"],
            rebalance_samples=self.train_params["rebalance_samples"],
            num_bins=self.train_params["num_bins"],
            map_recon_dim=self.map_params["map_recon_dim"],
            dataset_fraction=self.train_params["train_dataset_fraction"],
        )

        self.val_dataset = HabitatVideoDataset(
            dataset_dir=self.file_params["dataset_dir"],
            dataset_type="test",
            ann_file_name=self.file_params["train_ann_file_name"],
            transform=self.val_transform,
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
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import io
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
import PIL
from torchvision.transforms import ToTensor
from utils.geom_utils import transform_poses


@CALLBACK_REGISTRY
class LocalizationEvaluator(Callback):
    def __init__(self):
        super().__init__()

    def plot_trajectory(self, current_epoch, pl_module, poses_gt, poses_est):
        figure = plt.figure(figsize=(8, 8))
        plt.plot(
            poses_gt[:, 0], poses_gt[:, 1], "b", poses_est[:, 0], poses_est[:, 1], "r"
        )
        plt.axis("equal")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")

        plt.close(figure)
        buf.seek(0)

        # decode the array into an image
        traj_img = PIL.Image.open(buf)
        traj_img = ToTensor()(traj_img)  # .unsqueeze(0)
        tensorboard = pl_module.logger.experiment
        tensorboard.add_image(
            "loc_traj_images",
            traj_img,
            global_step=current_epoch,
        )

    def evaluate_trajectory(self, pl_module, eval_traj):
        poses_gt = np.zeros((len(eval_traj) + 1, 3))
        poses_est = np.zeros((len(eval_traj) + 1, 3))
        for pose_idx, item in enumerate(eval_traj):
            item["state"] = item["state"].unsqueeze(dim=0).to(pl_module.device)
            item["action"] = item["action"].unsqueeze(dim=0).to(pl_module.device)
            item["pose"] = item["pose"].unsqueeze(dim=0).to(pl_module.device)
            _, pos_recon0 = pl_module.step(item)
            # get delta values
            delta_pose_gt = item["pose"][0, -1, :].cpu().numpy()
            delta_pose_pred = pos_recon0[0, -1, :].cpu().numpy()
            # calculate and store the world frame positions
            new_pose_world_frame_gt = transform_poses(poses_gt[pose_idx], delta_pose_gt)
            poses_gt[pose_idx + 1, :] = new_pose_world_frame_gt
            new_pose_world_frame_pred = transform_poses(
                poses_est[pose_idx], delta_pose_pred
            )
            poses_est[pose_idx + 1, :] = new_pose_world_frame_pred

        return poses_gt, poses_est

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        traj_index = 4
        eval_traj = trainer.datamodule.train_dataset.get_full_eval_trajectory(
            traj_index=traj_index
        )
        with torch.set_grad_enabled(False):
            poses_gt, poses_est = self.evaluate_trajectory(
                pl_module, eval_traj=eval_traj
            )
            self.plot_trajectory(trainer.current_epoch, pl_module, poses_gt, poses_est)

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        num_trajs_to_eval = 5
        total_sum = 0
        total_n = 0
        with torch.set_grad_enabled(False):
            for traj_index in range(num_trajs_to_eval):
                eval_traj = trainer.datamodule.val_dataset.get_full_eval_trajectory(
                    traj_index=traj_index
                )
                poses_gt, poses_est = self.evaluate_trajectory(pl_module, eval_traj)
                total_sum += np.sum(np.linalg.norm(poses_gt - poses_est, axis=1))
                total_n += poses_gt.shape[0]

        ate_metric = total_sum / total_n

        pl_module.log("loc/val/ATE", ate_metric)

        traj_index = 2
        eval_traj = trainer.datamodule.val_dataset.get_full_eval_trajectory(
            traj_index=traj_index
        )

        poses_gt, poses_est = self.evaluate_trajectory(pl_module, eval_traj=eval_traj)
        self.plot_trajectory(trainer.current_epoch, pl_module, poses_gt, poses_est)

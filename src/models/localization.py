# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from typing import Dict, List, Union, Any

import torch
from src.models.modules.modeling_pact import PACTBase, PACTTaskBase
from torch import nn as nn


class PACTLocalization(PACTTaskBase):
    """PACT Lightning module for localization task.

    Args:
        PACTTask (lightningmodule): Base class for PACT Task modules
    """

    def __init__(
        self,
        pretrain_config: Dict[str, Any],
        head_config: Dict[str, Union[int, str, float, list]],
        **kwargs,
    ) -> None:
        # call PACTTaskBase.__init__
        super().__init__(**kwargs)
        self.criterion = torch.nn.MSELoss(reduction="none")

        self.save_hyperparameters("pretrain_config", "head_config")
        # * how to determine the tensor float type if we are using fp16?
        self.pose_mse_weight: List[float] = head_config["pose_mse_weight"]

        self.pose_mse_weight = torch.tensor(
            self.pose_mse_weight, device=self.device, requires_grad=False
        )

        if not pretrain_config["from_pretrained"]:
            # from_pretrained is not None or empty string
            self.base_model = PACTBase(self.gpt_config, self.input_config)
        else:
            self.base_model = self.load_base_from_ckpt(pretrain_config)

        if pretrain_config["freeze_base"]:
            self.base_model.freeze()

        self.head = PACTLocalizationHead(head_config)

    def step(self, batch):
        out_embd, _ = self.base_model(batch)
        pred_pose = self.head(out_embd)

        # size of batch["pose"] = b,t, 3
        loss = self.criterion(batch["pose"], pred_pose).sum(dim=(0, 1))
        return loss, pred_pose

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, pred_pose = self.step(batch=batch)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        weighted_loss = torch.dot(loss, self.pose_mse_weight.type_as(loss))
        values = [
            ("train/loss", weighted_loss),
            ("train/loss_trans_x", loss[0]),
            ("train/loss_trans_y", loss[1]),
            ("train/loss_angle", loss[2]),
        ]
        self.epoch_log(values)

        return {"loss": weighted_loss, "pred_pose": pred_pose}

    def train_epoch_end(self, training_step_outputs: List[torch.Tensor]) -> None:
        pass

    def validation_step(self, batch: Dict[str, dict], batch_idx: int):
        loss, _ = self.step(batch=batch)
        weighted_loss = torch.dot(loss, self.pose_mse_weight.type_as(loss))
        return torch.tensor(
            [weighted_loss, loss[0], loss[1], loss[2]], requires_grad=False
        )

    def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
        loss = torch.stack(validation_step_outputs, dim=1)

        values = [
            ("loc/val/loss", loss[0].mean()),
            ("loc/val/loss_trans_x", loss[1].mean()),
            ("loc/val/loss_trans_y", loss[2].mean()),
            ("loc/val/loss_angle", loss[3].mean()),
        ]
        self.epoch_log(values)


class PACTLocalizationHead(nn.Module):
    def __init__(self, head_config: Dict[str, Union[int, str, list]]) -> None:
        super().__init__()
        # a decoder head (originally called PosePredictorJoint) that predicts from tuple (a_{t-1}, s_{t}, s_{t})
        # * the input to head should be organized as predicted (s_{t-1}, a_{t-1}, s_{t}) at time step t
        self.pose_head = nn.Sequential(
            nn.Linear(3 * head_config["n_embd"], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.apply(self.__init_weights)

    def __init_weights(self, module: nn.Module):
        """Initialize weights in head module."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, out_embd: torch.Tensor) -> torch.Tensor:
        """Predict pose Delta{x}_t given predicted tuple (hat{s}_{t-1}, hat{a}_{t-1}, hat{s}_t).

        Args:
            out_embd (torch.Tensor): output of the forward method of class PACTBase.

        Returns:
            torch.Tensor: pose embedding
        """
        b, t, c = out_embd.size()
        # out_embd of size b*2t*d: output embdding of: [s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}]
        # doc of pytorch unfold: https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
        out_embd_sta_act_sta = (
            out_embd[:, :-1, :]
            .unfold(dimension=1, size=3, step=2)  # size = b * t * c * 3
            .transpose(2, 3)  # size = b * t * 3 * c
            .contiguous()
            .reshape(b, t // 2 - 1, -1)  # size = b * (t//2-1) * 3c
        )

        pose_pred = self.pose_head(out_embd_sta_act_sta)
        return pose_pred

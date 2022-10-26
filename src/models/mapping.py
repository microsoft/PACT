# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import io
from typing import Dict, List, Union, Any

import matplotlib.pyplot as plt
import PIL.Image
import torch
from src.models.modules.head_utils import MapDecoder_2x_Deconv, Reshape
from src.models.modules.modeling_pact import PACTBase, PACTTaskBase
from torch import nn as nn
from torchvision.transforms import ToTensor


class PACTMapping(PACTTaskBase):
    """PACT Lightning module for mapping task.

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
        self.criterion = torch.nn.MSELoss()

        self.save_hyperparameters("pretrain_config", "head_config")
        if not pretrain_config["from_pretrained"]:
            # from_pretrained is not None or empty string
            self.base_model = PACTBase(self.gpt_config, self.input_config)
        else:
            self.base_model = self.load_base_from_ckpt(pretrain_config)

        if pretrain_config["freeze_base"]:
            self.base_model.freeze()

        self.head = PACTMappingHead(head_config)

    def step(self, batch):
        out_embd, _ = self.base_model(batch)
        pred_map = self.head(out_embd)
        img_x, img_y = batch["gt_map"].size(1), batch["gt_map"].size(2)
        loss = self.criterion(
            batch["gt_map"],
            pred_map.view(-1, img_x, img_y),
        )

        return (
            loss,
            batch["gt_map"][0],
            pred_map[0].view(img_x, img_y),
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, gt_map, pred_map = self.step(batch=batch)
        self.epoch_log([("map/train/loss", loss)])
        return {"loss": loss, "gt_map": gt_map, "pred_map": pred_map}

    def training_epoch_end(self, training_step_outputs: List[torch.Tensor]) -> None:
        gt_map = training_step_outputs[-1]["gt_map"]
        pred_map = training_step_outputs[-1]["pred_map"]

        self.plot_map(gt_map, pred_map)

    def validation_step(self, batch: Dict[str, dict], batch_idx: int):
        loss, gt_map, pred_map = self.step(batch=batch)
        self.epoch_log([("map/val/loss", loss)])
        return {"loss": loss, "gt_map": gt_map, "pred_map": pred_map}

    def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
        gt_map = validation_step_outputs[-1]["gt_map"]
        pred_map = validation_step_outputs[-1]["pred_map"]

        self.plot_map(gt_map, pred_map)

    def plot_map(self, gt_map, pred_map):
        gt_map = (gt_map + 1.0) * 255.0 / 2.0
        pred_map = (pred_map + 1.0) * 255.0 / 2.0
        figure = plt.figure(figsize=(8, 16))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_map.detach().cpu())
        plt.subplot(1, 2, 2)
        plt.imshow(pred_map.detach().cpu())
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)

        # decode the array into an image
        map = PIL.Image.open(buf)
        map = ToTensor()(map)
        tensorboard = self.logger.experiment
        tensorboard.add_image(
            "gt_map and pred_map",
            map,
            global_step=self.trainer.current_epoch,
            dataformats="CHW",
        )


class PACTMappingHead(nn.Module):
    def __init__(self, head_config: Dict[str, int]) -> None:
        super().__init__()
        self.map_head = nn.Sequential(
            nn.Linear(2 * head_config["n_embd"] * head_config["seq_len"], 4096),
            nn.ReLU(),
            Reshape(16, 16, 16),
            MapDecoder_2x_Deconv(16),
        )
        self.apply(self.__init_weights)

    def __init_weights(self, module: nn.Module):
        """Initialize weights in head module."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight, 1.0, 0.02)
            torch.nn.init.zeros_(module.bias)

    def forward(self, out_embd: torch.Tensor) -> torch.Tensor:
        """Predict map given the whole (state, action) embedding sequence. For convenice, it keeps
        the last action embedding.

        Args:
            out_embd (torch.Tensor): output of the forward method of class PACTBase.

        Returns:
            torch.Tensor: map embeddings
        """
        b = out_embd.size(0)
        map_pred = self.map_head(out_embd.view(b, -1))
        return map_pred

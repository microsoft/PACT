# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from typing import Any, Dict, List, Tuple, Union

import torch
from src.models.modules.modeling_pact import PACTBase, PACTTaskBase
from torch import nn as nn


class PACTPretrain(PACTTaskBase):
    """PACT Lightning module for pretraining task.

    Args:
        PACTTask (lightningmodule): Base class for PACT Task modules
    """

    def __init__(
        self,
        head_config: Dict[str, Union[int, list, str, float]],
        **kwargs,
    ):
        # call PACTTaskBase.__init__
        super().__init__(**kwargs)
        self.state_criterion = torch.nn.MSELoss()
        self.action_criterion = (
            torch.nn.MSELoss()
            if self.input_config["action"]["input_type"] == "continuous"
            else torch.nn.CrossEntropyLoss()
        )

        self.save_hyperparameters(
            "head_config",
        )

        self.model = PACTBase(self.gpt_config, self.input_config)
        self.head = PACTPretrainHead(head_config, self.input_config)

    def step(self, batch: Dict[str, torch.Tensor]):
        out_embd, state_embd = self.model(batch)
        state_pred, action_pred = self.head(out_embd)
        # it would not work for multiple dimensional
        if isinstance(self.action_criterion, torch.nn.CrossEntropyLoss):
            # one dimensional discrete action
            b, t, c = action_pred.size()
            loss_action = self.action_criterion(
                action_pred.view(b * t, c), batch["action"].view(-1)
            )
        else:
            loss_action = self.action_criterion(action_pred, batch["action"])

        loss_state = self.state_criterion(state_embd[:, 1:, :], state_pred)
        loss = loss_action + loss_state

        return loss, loss_action, loss_state

    def forward(self, batch: Dict[str, torch.Tensor]):
        out_embd, state_embd = self.model(batch)
        state_pred, action_pred = self.head(out_embd)
        return state_pred, action_pred

    def training_step(self, batch, batch_idx: int):
        loss, loss_act, loss_state = self.step(batch)
        values = [
            ("train/loss", loss),
            ("train/loss_act", loss_act),
            ("train/loss_state", loss_state),
        ]
        self.epoch_log(values)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        loss, loss_act, loss_state = self.step(batch)
        return torch.tensor([loss, loss_act, loss_state], requires_grad=False)

    def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
        losses = torch.stack(validation_step_outputs, dim=1)
        values = [
            ("val/loss", losses[0].mean()),
            ("val/loss_act", losses[1].mean()),
            ("val/loss_state", losses[2].mean()),
        ]
        self.epoch_log(values)


class PACTPretrainHead(nn.Module):
    def __init__(
        self, head_config: Dict[str, int], input_config: Dict[str, Any]
    ) -> None:
        super().__init__()

        n_embd = head_config["n_embd"]
        # the numbers '2', '1' will need to be set up in future
        self.state_head = nn.Sequential(nn.Linear(2 * n_embd, n_embd))

        if input_config["action"]["input_type"] == "continuous":
            self.action_head = nn.Sequential(nn.Linear(n_embd, 1))
        else:
            self.action_head = nn.Sequential(
                nn.Linear(
                    n_embd,
                    input_config["action"]["tokenizer_kwargs"]["action_dim"],
                )
            )
        self.apply(self.__init_weights)

    def __init_weights(self, module: nn.Module):
        """Initialize weights in head module."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, out_embd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = out_embd.size()
        # from the tokens of s_t, predict next action a_t (like a policy)
        action_preds = self.action_head(out_embd[:, ::2, :])
        # from the output embeddings of s_t and a_t, predict embedding of next state s_{t+1}
        # -1 to ignore the last s_t, a_t pair because we will not predict s_t+1 at the very end
        state_action_out_embd = out_embd.reshape(b, t // 2, -1)[:, :-1, :]
        state_preds = self.state_head(state_action_out_embd)

        return state_preds, action_preds

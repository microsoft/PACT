# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import yaml
import os
from abc import ABC
from typing import Dict, List, Tuple, TypeVar, Union

import einops
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from src.models.modules.minGPT import GPT, GPTConfig
from src.models.modules.tokenizer_pact import PACTTokenizer
from torch import nn as nn
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR


class PACTTaskBase(ABC, LightningModule):
    """Abstract PACT lightning modules for tasks.

    Notes:
        1. weights initialization is done by each submodular component itself, unless those weights (nn.parameters()) are declared inside a module.
    Args:
        ABC (abc.ABC): python abstract class
    """

    def __init__(
        self,
        optimizer_config: Dict[str, Union[float, str, list]],
        scheduler_config: Dict[str, Union[float, str, list]],
        gpt_config: Dict[str, Union[int, float]],
        input_config: Dict[str, dict],
    ) -> None:
        super().__init__()
        self.input_config = input_config
        self.gpt_config = gpt_config
        self.save_hyperparameters(
            "optimizer_config",
            "scheduler_config",
            "gpt_config",
            "input_config",
        )

    def load_base_from_ckpt(self, pretrain_config) -> "PACTBase":
        ckpt = torch.load(pretrain_config["load_ckpt_path"])
        model = PACTBase(
            gpt_config=ckpt["hyper_parameters"]["gpt_config"],
            input_config=ckpt["hyper_parameters"]["input_config"],
        )
        # 'model' is the name of the attribute of PACTBase in lightning module
        state_dict = {
            key.partition("model.")[2]: value
            for key, value in ckpt["state_dict"].items()
            if key.startswith("model.")
        }

        print("Loading PACTBase from checkpoint...")
        model.load_state_dict(state_dict=state_dict)
        return model

    def save_pretrained_base(self, basemodel_dir) -> None:
        """Save pretrained pactbase."""
        self.model.save_pretrained(basemodel_dir)

    def epoch_log(self, values: List[Tuple[str, torch.Tensor]]) -> None:
        """Add variables to logger in the epoch level.

        Notes:
            Stats shown in logger is only from rank=0 process in distributed settings

        Args:
            values (List[Tuple[str, torch.Tensor]]): values to put to logger
        """
        for key, value in values:
            self.log(
                key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        """This long function is unfortunately doing something very simple and is being very
        defensive: We are separating out all parameters of the model into two buckets: those that
        will experience weight decay for regularization and those that won't (biases, and
        layernorm/embedding weights).

        notes:
            For modules experiencing decays, parameters in GPT Blocks and the rest of the transformer have different
            decay parameters.
        We are then returning the PyTorch optimizer object.

        Code based on minGPT: https://github.com/karpathy/minGPT
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay_gpt = set()
        decay_rest = set()
        no_decay = set()

        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
        )
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm3d,
            torch.nn.BatchNorm2d,
        )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    if "blocks" in pn:
                        decay_gpt.add(fpn)
                    else:
                        decay_rest.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = (
            (decay_gpt & no_decay) | (decay_rest & no_decay) | (decay_rest & decay_gpt)
        )
        union_params = decay_gpt | decay_rest | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters {} were not separated into either decay/no_decay set!".format(
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay_gpt))],
                "weight_decay": self.hparams.optimizer_config["weight_decay_gpt"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(decay_rest))],
                "weight_decay": self.hparams.optimizer_config["weight_decay_rest"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.optimizer_config["lr"],
            betas=self.hparams.optimizer_config["betas"],
        )

        max_epochs = self.trainer.max_epochs
        warmup_epochs = (
            self.trainer.max_epochs * self.hparams.scheduler_config["warmup_ratio"]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=self.hparams.scheduler_config["warmup_start_lr"],
            eta_min=self.hparams.scheduler_config["min_lr"],
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


P = TypeVar("P", bound="PACTBase")
MODEL_config = "model.yaml"
MODEL_DICT = "state_dict.pt"


class PACTBase(ABC, nn.Module):
    def __init__(
        self, gpt_config: Dict[str, Union[int, float]], input_config: Dict[str, dict]
    ):
        super().__init__()

        self.model_config = {"gpt_config": gpt_config, "input_config": input_config}
        self.gpt_config = GPTConfig(**gpt_config)
        # position embedding is defined here, instead of in GPT, is because in future, we may have other elements than (state, action) to put in there.
        # In that case, we need to change the __init__ as well as the forward function
        self.pos_embd_global = nn.Embedding(
            self.gpt_config.block_size, self.gpt_config.n_embd
        )
        self.pos_embd_local = nn.Embedding(
            self.gpt_config.seq_len, self.gpt_config.n_embd
        )

        self.tokenizer = PACTTokenizer(input_config, self.gpt_config.n_embd)
        self.gpt = GPT(self.gpt_config)

        # initialize positional embeddings
        # initalizations of tokenizer and gpt are taken care of by the two submodules themselves
        torch.nn.init.normal_(self.pos_embd_global.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pos_embd_local.weight, mean=0.0, std=0.02)

    def freeze(self) -> None:
        """Keep parameters in PACTBase frozen.

        This disables dropout and makes BN layers use statistics learning during training.
        This implementation is what lighting does with lightningModule.freeze().
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return

    def unfreeze(self) -> None:
        """Unfreeze the PACTBase models' parameters, they will be updated by optimizers."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        return

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Dict[str, torch.Tensor]): Dictionary of "state", "action", "pose" batches

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (output embedding sequence, and state embedding sequence (input))
        """
        # get arbrtary data from batch dict
        batch_state = batch["action"]
        b, t = batch_state.size(0), batch_state.size(1)

        # * current action is scalar, delete this line if action itself is a vector
        if batch["action"].dim() == 2:
            batch["action"].unsqueeze_(2)

        # calculate global positional embedding (unique for each token)
        pos_local = torch.arange(
            0, t, dtype=torch.long, device=batch_state.device
        ).unsqueeze(
            0
        )  # shape=(1,t)
        # * magic number 2 here denotes number of inputs we are feeding into attention block: (state, action) pair with 2 elements in here.
        pos_global = torch.arange(
            0, t * 2, dtype=torch.long, device=batch_state.device
        ).unsqueeze(
            0
        )  # shape (1, 2*t)

        pos_embd_global = self.pos_embd_global(pos_global)

        # calculate local positional embedding (unique for each state-action pair)
        pos_embd_local = self.pos_embd_local(pos_local)
        pos_embd_local = torch.repeat_interleave(pos_embd_local, 2, dim=1)
        # position_embeddings_local = position_embeddings_local.repeat(B,1,1)
        tok_embd_dict = self.tokenizer(
            {"state": batch["state"], "action": batch["action"]}
        )
        # currently, keys: "state", "action" are hard-coded, but can be extended in the future
        embd_list = [tok_embd_dict[input_type] for input_type in ["state", "action"]]

        tok_embd = einops.rearrange(
            torch.stack(embd_list, dim=2),
            "b n s l -> b (n s) l",
        )
        out_embd = self.gpt(tok_embd + pos_embd_global + pos_embd_local)
        return out_embd, tok_embd_dict["state"]

    @classmethod
    def from_pretrained(cls, model_dir) -> P:
        """Given model config and model weights, construct a model with the configuration and
        weights. Set the model's mode to models.eval().

        Args:
            model_dir (str): directory that contains json file that specifies the network config
            and dict file that stores state-dict of network
        """
        print(f"Loading from pretrained stored at {model_dir}!")
        with open(os.path.join(model_dir, MODEL_config)) as f:
            model_config = yaml.safe_load(f)

        model = cls(model_config["gpt_config"], model_config["input_config"])
        model.load_state_dict(torch.load(os.path.join(model_dir, MODEL_DICT)))
        model.eval()

        return model

    def save_pretrained(self, model_dir) -> None:
        """Save config, state_dict of the current model.

        Args:
            model_dir (str): directory that contains json file that specifies the network config
            and dict file that stores state-dict of network
        Returns:
            None
        """
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, MODEL_config), mode="w") as f:
            yaml.safe_dump(self.model_config, f)

        torch.save(self.state_dict(), os.path.join(model_dir, MODEL_DICT))

        return None

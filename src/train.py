# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from omegaconf import OmegaConf
from utils import pl_utils


def main(cfg):
    trainer = pl_utils.instantiate_trainer(cfg)
    model = pl_utils.instantiate_class(cfg["model"])
    datamodule = pl_utils.instantiate_class(cfg["data"])

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    cfg = OmegaConf.from_cli()

    if "base" in cfg:
        basecfg = OmegaConf.load(cfg.base)
        del cfg.base
        cfg = OmegaConf.merge(basecfg, cfg)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        main(cfg)
    else:
        raise SystemExit("Base configuration file not specified! Exiting.")

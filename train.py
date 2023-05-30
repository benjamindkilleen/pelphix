# !/usr/bin/env python3
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
from time import time

from typing import Optional, List, Any, Tuple
from shutil import rmtree
from scipy.spatial import KDTree
import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
from rich.traceback import install
from torch.utils.data import DataLoader
import scienceplots
import imageio.v3 as iio
import imageio

# Nosort
import deepdrr
import pelphix
from pelphix.utils.onedrive_utils import OneDrive
from pelphix.ssm import ssm_build
from pelphix.sims import PelphixSim
from perphix.data import (
    PerphixSequenceDataset,
)


np.set_printoptions(precision=3, suppress=True, threshold=10000)

torch.set_float32_matmul_precision("medium")

# Requires latex
plt.style.use(["science", "ieee"])

install(show_locals=False)

os.environ["HYDRA_FULL_ERROR"] = "1"

# Use agg backend for plotting when no graphical display available.
mpl.use("agg")

plt.rcParams["font.family"] = "serif"  # "Times New Roman"

log = logging.getLogger("pelphix")
log.setLevel(logging.DEBUG)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def train(cfg):
    from pelphix.modules.seq import PelphixModule

    mp.set_start_method("spawn", force=True)
    cfg = cfg.experiment

    train_dataset = PerphixSequenceDataset.from_configs(**cfg.sequences_train)
    # counts = train_dataset.get_sequence_counts()
    counts = None
    val_dataset = PerphixSequenceDataset.from_configs(**cfg.sequences_val)

    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
    trainer = pl.Trainer(**cfg.trainer)

    log.info(f"Training on {len(train_dataset)} sequences.")
    if cfg.weights_only:
        # For resuming training after changing the scheduler or something
        module = PelphixModule.load_from_checkpoint(
            cfg.ckpt, **cfg.unet_module, sequence_counts=counts
        )
        trainer.fit(module, train_dataloader, val_dataloader)
    else:
        # Normal training
        module = PelphixModule(**cfg.unet_module, sequence_counts=counts)
        trainer.fit(module, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt)


if __name__ == "__main__":
    train()

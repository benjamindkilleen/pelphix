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
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.traceback import install
from torch.utils.data import DataLoader
import scienceplots

# Nosort
import deepdrr
import pelphix
from pelphix.utils.onedrive_utils import OneDrive
from pelphix.ssm import ssm_build
from pelphix.sim import PelphixSim
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


@pelphix.register_experiment
def ssm(cfg):
    """Build the SSM and propagate annotations.

    Not for the faint of heart.

    """
    ssm_build(cfg)


@pelphix.register_experiment
def pregenerate():
    """Generate a dataset for pretraining."""
    raise NotImplementedError("TODO: view-invariant pretraining of image features.")


@pelphix.register_experiment
def pretrain():
    """Pretrain the model on the view-invariant dataset."""
    pass


@pelphix.register_experiment
def generate(cfg):
    """Generate the sequence dataset."""
    mp.set_start_method("spawn", force=True)
    # Check that the CTs/annotations are downloaded
    onedrive = OneDrive(cfg.onedrive_dir)
    nmdid_dir = Path(cfg.nmdid_dir).expanduser()
    for d in ["nifti", "TotalSegmentator", "TotalSegmentator_mesh", cfg.pelvis_annotations_dir]:
        onedrive.download(nmdid_dir / d, skip=cfg.skip_download)

    # Generate the images
    pelphix_sim = PelphixSim(train=True, **cfg.sim)
    pelphix_sim.generate()

    pelphix_val = PelphixSim(train=False, **cfg.sim)
    pelphix_val.generate()


@pelphix.register_experiment
def train(cfg):
    from pelphix.modules.seq import PelphixModule

    train_dataset = PerphixSequenceDataset.from_configs(**cfg.sequences_train)
    counts = train_dataset.get_sequence_counts()
    val_dataset = PerphixSequenceDataset.from_configs(**cfg.sequences_val)

    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
    trainer = pl.Trainer(**cfg.trainer)

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


@pelphix.register_experiment
def vis(cfg):
    """Visualize the training set."""
    dataset = PerphixSequenceDataset.from_configs(**cfg.sequences_train)


@pelphix.register_experiment
def test(cfg):
    from pelphix.modules.seq import PelphixModule

    # onedrive = OneDrive(cfg.onedrive_dir)
    # onedrive.download(cfg.liverpool.root_in_onedrive, skip=cfg.skip_download)

    if cfg.ckpt is None:
        cfg.ckpt = os.path.join(
            cfg.results_dir, "lightning_logs", "version_0", "checkpoints", "last.ckpt"
        )

    dataset = PerphixSequenceDataset.from_configs(**cfg.sequences_test)
    # TODO: flip test-set X-ray images horizontallyk

    dataloader = DataLoader(dataset, **cfg.dataloader, shuffle=False)
    module = PelphixModule.load_from_checkpoint(
        cfg.ckpt,
        results_dir=cfg.results_dir,
        test_dataset=dataset,
        strict=False,
        **cfg.unet_module,
    )
    trainer = pl.Trainer(**cfg.trainer)
    trainer.test(module, dataloaders=dataloader)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    pelphix.run(cfg)


if __name__ == "__main__":
    main()

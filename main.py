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
    ssm_build(cfg)


@pelphix.register_experiment
def generate(cfg):
    mp.set_start_method("spawn", force=True)
    # Check that the CTs/annotations are downloaded
    onedrive = OneDrive(cfg.onedrive_dir)
    onedrive.download(cfg.nmdid_dir, skip=cfg.skip_download)

    # Generate the images
    pelvic_workflows = PelvicWorkflowsSimulation(train=True, **cfg.pelvic_workflows)
    pelvic_workflows.generate()

    pelvic_workflows_val = PelvicWorkflowsSimulation(train=False, **cfg.pelvic_workflows)
    pelvic_workflows_val.generate()


@pelphix.register_experiment
def train(cfg):
    from pelphix.train_detector import run

    run(cfg.train)


@pelphix.register_experiment
def transformer(cfg):
    from pelphix.train_transformer import RecognitionModule
    from pelphix.detector import EmbeddingsDataset

    train_dataset = EmbeddingsDataset(**cfg.embeddings_dataset, train=True)
    counts = train_dataset.dataset.get_sequence_counts()

    val_dataset = EmbeddingsDataset(**cfg.embeddings_dataset, train=False)

    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader)

    module = RecognitionModule(
        d_input=train_dataset.d_input, sequence_counts=counts, **cfg.recognition_module
    )
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(module, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt)

    # TODO: test the model on the cadaver and patient datasets


@pelphix.register_experiment
def test(cfg):
    # TODO: write out the prediction results. Might need train dataset if only for correct mapping, but should match.
    # 1. Get the predictions
    # 2. Map back to phase names using the dataset
    # 3. Write out the results to the original output dir of the transformer model, based on ckpt
    # 4. If write_images is true, write out the images with label names on them, to make a GIF
    from pelphix.train_transformer import RecognitionModule
    from pelphix.detector import EmbeddingsDataset

    onedrive = OneDrive(cfg.onedrive_dir)
    onedrive.download(cfg.liverpool.root_in_onedrive, skip=cfg.skip_download)

    if cfg.ckpt is None:
        cfg.ckpt = os.path.join(
            cfg.results_dir, "lightning_logs", "version_0", "checkpoints", "last.ckpt"
        )

    embeddings_dataset = EmbeddingsDataset(**cfg.embeddings_dataset, train=False)
    # TODO: flip test-set X-ray images horizontally

    embeddings_dataloader = DataLoader(embeddings_dataset, **cfg.dataloader, shuffle=False)
    module = RecognitionModule.load_from_checkpoint(
        cfg.ckpt,
        d_input=embeddings_dataset.d_input,
        results_dir=cfg.results_dir,
        test_dataset=embeddings_dataset.dataset,
        strict=False,
        **cfg.recognition_module,
    )
    trainer = pl.Trainer(**cfg.trainer)
    trainer.test(module, dataloaders=embeddings_dataloader)


@pelphix.register_experiment
def train_unet(cfg):
    from pelphix.train_transformer_unet import UNetModule

    train_dataset = PelvicWorkflowsSequences.from_configs(**cfg.sequences_train)
    counts = train_dataset.get_sequence_counts()
    val_dataset = PelvicWorkflowsSequences.from_configs(**cfg.sequences_val)

    train_dataloader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
    trainer = pl.Trainer(**cfg.trainer)

    if cfg.weights_only:
        # For resuming training after changing the scheduler or something
        module = UNetModule.load_from_checkpoint(
            cfg.ckpt, **cfg.unet_module, sequence_counts=counts
        )
        trainer.fit(module, train_dataloader, val_dataloader)
    else:
        # Normal training
        module = UNetModule(**cfg.unet_module, sequence_counts=counts)
        trainer.fit(module, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt)


@pelphix.register_experiment
def test_unet(cfg):
    from pelphix.train_transformer_unet import UNetModule

    # onedrive = OneDrive(cfg.onedrive_dir)
    # onedrive.download(cfg.liverpool.root_in_onedrive, skip=cfg.skip_download)

    if cfg.ckpt is None:
        cfg.ckpt = os.path.join(
            cfg.results_dir, "lightning_logs", "version_0", "checkpoints", "last.ckpt"
        )

    dataset = PelvicWorkflowsSequences.from_configs(**cfg.sequences_test)
    # TODO: flip test-set X-ray images horizontallyk

    dataloader = DataLoader(dataset, **cfg.dataloader, shuffle=False)
    module = UNetModule.load_from_checkpoint(
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

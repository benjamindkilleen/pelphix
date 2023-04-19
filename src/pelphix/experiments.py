#!/usr/bin/env python3
import logging
from functools import wraps
from pathlib import Path
from typing import Callable
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf

from . import utils

log = logging.getLogger(__name__)

Experiment = Callable[[DictConfig], None]
_experiment_registry = dict()


@utils.doublewrap
def register_experiment(func: Experiment, name: Optional[str] = None):
    _experiment_registry[name if isinstance(name, str) else func.__name__] = func
    return func


def get_experiment(name: str) -> Experiment:
    if name not in _experiment_registry:
        raise KeyError(f"experiment not registered: {name}")
    return _experiment_registry[name]


def run(cfg: DictConfig) -> None:
    experiment = HydraConfig.get().runtime.choices.experiment
    OmegaConf.resolve(cfg)
    log.info(f"running `{experiment}` with config:\n{OmegaConf.to_yaml(cfg.experiment)}")
    get_experiment(experiment)(cfg.experiment)

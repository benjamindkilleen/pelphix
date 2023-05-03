import shutil
from typing import List, Optional, Tuple, Type, Union, Dict, Any
from pathlib import Path
import datetime
import os
import torch

import copy
import base64
import json
from pycocotools import mask as mask_utils
from rich.progress import track
from gpustat import print_gpustat
from collections import Counter
import deepdrr
from deepdrr import geo, Volume
from deepdrr.device import SimpleDevice
from deepdrr.utils import data_utils, image_utils
from deepdrr import Projector
import re
import logging
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Queue, Process
from queue import Empty
import pyvista as pv

from perphix.data import PerphixBase

from .base import PelphixBase
from .state import Task, Activity, Acquisition, Frame, SimState, FrameState
from ..tools import Tool, Screw, get_screw, get_screw_choices
from ..shapes import Cylinder, Mesh
from ..utils import coco_utils, save_json, load_json


log = logging.getLogger(__name__)

DEGREE_SIGN = "\N{DEGREE SIGN}"
ONE_DEGREE = math.radians(1)
FIVE_DEGREES = math.radians(5)
TEN_DEGREES = math.radians(10)
FIFTEEN_DEGREES = math.radians(15)
THIRTY_DEGREES = math.radians(30)
FORTY_FIVE_DEGREES = math.radians(45)
SIXTY_DEGREES = math.radians(60)


class PelphixUniform(PelphixBase, Process):
    """Class for generating uniformly sampled datasets."""

    def __init__(
        self,
        root: Union[str, Path],
        nmdid_root: Union[str, Path],
        pelvis_annotations_dir: Union[str, Path],
        train: bool = True,
        num_val: int = 10,
        scan_name: str = "THIN_BONE_TORSO",
        image_size: tuple[int, int] = (256, 256),
        overwrite: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        job_queue: Optional[Queue] = None,
        finished_queue: Optional[Queue] = None,
        num_workers: int = 0,
        num_samples_per_case: int = 100,
    ):
        self.kwargs = locals()
        del self.kwargs["self"]
        del self.kwargs["__class__"]
        del self.kwargs["job_queue"]
        del self.kwargs["finished_queue"]

        super().__init__(
            root=root,
            nmdid_root=nmdid_root,
            pelvis_annotations_dir=pelvis_annotations_dir,
            train=train,
            num_val=num_val,
            scan_name=scan_name,
            image_size=image_size,
            overwrite=overwrite,
            cache_dir=cache_dir,
        )

        self.job_queue = job_queue
        self.finished_queue = finished_queue
        self.num_workers = num_workers
        self.num_samples_per_case = num_samples_per_case

        case_names = sorted(
            [
                case.name
                for case in self.pelvis_annotations_dir.glob("case-*/")
                if re.match(self.CASE_PATTERN, case.name) is not None
            ]
        )
        num_images = len(case_names) * self.num_samples_per_case
        if self.train:
            self.name = f"pelphix-uniform_{num_images // 1000:03d}k_train"
            self.case_names = case_names[: -self.num_val]
        else:
            self.name = f"pelphix-uniform_{num_images // 1000:03d}k_val"
            self.case_names = case_names[-self.num_val :]

        self.num_cases = len(self.case_names)
        self.num_images = self.num_cases * self.num_samples_per_case
        log.info(f"{self.name}: {self.num_cases} cases, {self.num_images} images")

        self.annotations_dir = self.root / "annotations"
        self.images_dir = self.root / self.name

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Don't sync the images directory.
        (self.images_dir / ".nosync").touch()

    # TODO: sample_case(), generate(), and run() methods. Pretty simple. Have to sample more often
    # around AP than lateral. Base it off of solid angle.

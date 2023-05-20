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


class PelphixUniform(PelphixBase):
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
        max_wires: int = 7,
        max_screws: int = 7,
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
        self.max_wires = max_wires
        self.max_screws = max_screws

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
        self.tmp_dir = self.root / f"tmp_{self.name}"

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Don't sync the images directory.
        (self.annotations_dir / ".nosync").touch()
        (self.images_dir / ".nosync").touch()
        (self.tmp_dir / ".nosync").touch()

    # TODO: sample_case(), generate(), and run() methods. Pretty simple. Have to sample more often
    # around AP than lateral. Base it off of solid angle.

    def get_tmp_annotation_path(self, case_name: str) -> Path:
        return self.tmp_dir / f"{case_name}.json"

    def get_tmp_images_dir(self, case_name: str) -> Path:
        return self.tmp_dir / case_name

    def sample_case(self, case_name: str):
        """Sample images for a given case.

        Beforehand, allocate maximum number of screws (random lengths) and wires,
        and initialize projectors for segmentations, corridors, etc.
        Initialize the C-arm as a SimpleDevice with random properties.

        For each image:
        1. Uniformly sample a number of wires in [0, max_wires].
        2. Sample a number of screws in [0, max_screws], with 0 screws much more likely than the rest.
        3. Place screws and wires uniformly in a box centered on the pelvis by sampling the tip location uniformly and the tool direction uniformly on the sphere.
        4. Sample a corridor on which to center the view, including "no-corridor" which samples the center uniformly in the pelvis-box.
        5. Sample a direction for the view, first by choosing either AP- or lateral-centered (with proportional probabilities),
            then by sampling uniformly on the solid angle about the choice.
        6. Check if we happen to be capturing a standard view. This should be done solely with the relevant corridors/keypoints and the current view, separately for each.
        7. Sample the source-to-patient distance randomly in a reasonable range.
        8. Do the projection and save the image and annotations.

        """
        annotation_path = self.get_tmp_annotation_path(case_name)
        annotation = self.get_base_annotation()
        images_dir = self.get_tmp_images_dir(case_name)

        device = self.sample_device()
        wire_catid = self.get_annotation_catid("wire")
        screw_catid = self.get_annotation_catid("screw")

        # Get the volumes
        ct, seg_volumes, seg_meshes, corridors, pelvis_landmarks = self.load_case(case_name)
        wires = deepdrr.vol.KWire.from_example()
        screw_choices = list(get_screw_choices())
        screws = []
        for screw_idx in range(self.max_screws):
            screw = screw_choices[np.random.choice(len(screw_choices))]
            screws.append(screw)

        # Probability for the number of wires and screws
        wire_probabilities = [0.5] + [0.5 / len(wires)] * len(wires)
        screw_probabilities = [0.5] + [0.5 / len(screws)] * len(screws)

        intensity_upper_bound = np.random.uniform(2, 8)
        projector = Projector(
            [ct, *wires, *screws],
            device=device,
            neglog=True,
            step=0.05,
            intensity_upper_bound=intensity_upper_bound,
            attenuate_outside_volume=True,
        )
        projector.initialize()

        # Mapping from track id to projector
        # Not really a trackid, but whatever
        seg_projectors: dict[int, Projector] = dict()

        # Add the volumes to the projector
        for seg_name, seg_volume in seg_volumes.items():
            track_id = 1000 * self.get_annotation_catid(seg_name) + 0
            seg_projectors[track_id] = Projector(seg_volume, device=device, neglog=True)
            seg_projectors[track_id].initialize()

        for wire_idx, wire in enumerate(wires):
            track_id = 1000 * wire_catid + wire_idx
            seg_projectors[track_id] = Projector(wire, device=device, neglog=True)
            seg_projectors[track_id].initialize()

        for screw_idx, screw in enumerate(screws):
            track_id = 1000 * screw_catid + screw_idx
            seg_projectors[track_id] = Projector(screw, device=device, neglog=True)
            seg_projectors[track_id].initialize()

        for i in range(self.num_samples_per_case):
            pass

        # TODO: fix cylinder projection so that the circular ends are also projected, possibly by
        # sampling edge points and taking a convex hull. can use scipy for that, but may be much
        # faster to figure out which points are actually contributing to the hull, since only those
        # one side of the edge lines are relevant. Can just add those to the point and return a
        # polygon instead of a quadrangle. Buttttttt probably doesn't really matter for this
        # application, and for wire placement, would want to grow the corridor anyway.

    def generate(self):
        """Generate the dataset."""

        tmp_annotation_paths = sorted(list(self.tmp_dir.glob("*.json")))
        cases_done = set(int(p.stem) for p in tmp_annotation_paths)

        if self.overwrite:
            if cases_done:
                log.critical(
                    f"Overwriting {len(cases_done)} existing annotations. Press CTRL+C to cancel."
                )
                input("Press ENTER to overwrite.")

            cases_done = set()
            shutil.rmtree(self.tmp_dir)
            shutil.rmtree(self.images_dir)
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
        elif len(cases_done) > 0:
            log.info(f"Skipping {len(cases_done)} existing cases.")
        else:
            log.info("No existing cases found. Starting from scratch.")

        if self.num_workers == 0:
            for case_idx, case_name in enumerate(self.case_names):
                if case_name not in cases_done:
                    self.sample_case(case_name)
                log.info(
                    f"======================= {case_idx + 1}/{len(self.case_names)} ======================="
                )

        else:
            job_queue = mp.Queue(self.num_cases)
            finished_queue = mp.Queue(self.num_cases)
            for case_idx, case_name in enumerate(self.case_names):
                if case_name not in cases_done:
                    # Add jobs to the queue
                    log.info(f"Adding case {case_name} ({case_idx}) to the queue.")
                    job_queue.put(case_name)

            # Start workers
            workers: list[PelphixUniform] = []
            for i in range(self.num_workers):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(i % torch.cuda.device_count())
                worker = PelphixUniform(
                    job_queue=job_queue, finished_queue=finished_queue, **self.kwargs
                )
                worker.start()

            while len(cases_done) < self.num_cases:
                try:
                    case_name = finished_queue.get(block=False)
                    cases_done.add(case_name)
                    log.info(
                        f"{self.name}: case {case_name} finished. ({len(cases_done)} / {self.num_procedures})."
                    )

                except Empty:
                    # No new procedures have finished
                    time.sleep(1)
                    continue
                except KeyboardInterrupt:
                    log.critical("KeyboardInterrupt. Exiting.")
                    break
                except Exception as e:
                    log.critical(e)
                    break

            # Kill all workers to finish
            for worker in workers:
                worker.terminate()

        log.info(f"Finished generating images for {self.num_cases} cases. Starting cleanup...")

        image_id = 0
        annotation_id = 0
        annotation = self.get_base_annotation()

        for case_annotation_path in track(
            list(self.tmp_dir.glob("*.json")), description="Merging annotations...."
        ):
            case_annotation = load_json(case_annotation_path)
            case_name = case_annotation_path.stem
            case_images_dir = self.tmp_dir / case_name

            if not case_images_dir.exists():
                log.warning(f"Case {case_name} does not exist. Skipping.")
                continue

            first_image_id = image_id

            for image_info in case_annotation["images"]:
                image_path = case_images_dir / str(image_info["file_name"])
                new_image_path = self.images_dir / f"{case_name}_{image_info['file_name']}"
                if image_path.exists() and not new_image_path.exists():
                    shutil.copy(image_path, new_image_path)
                elif new_image_path.exists():
                    pass
                else:
                    log.error(f"Image {image_path} does not exist. Skipping.")
                    continue

            for ann in case_annotation["annotations"]:
                ann["id"] += annotation_id
                ann["image_id"] += first_image_id + ann["image_id"]
                annotation["annotations"].append(ann)
                annotation_id += 1

        log.info("Saving annotations...")
        save_json(annotation, self.annotations_dir / f"{self.name}.json")

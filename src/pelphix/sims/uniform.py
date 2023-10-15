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

STANDARD_VIEWS = [
    Acquisition.ap,
    Acquisition.lateral,
    Acquisition.inlet,
    Acquisition.outlet,
    Acquisition.oblique_left,
    Acquisition.oblique_right,
    Acquisition.teardrop_left,
    Acquisition.teardrop_right,
]


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a vector to sum to 1."""
    x = np.array(x)
    return x / np.sum(x)


def solid_angle(theta: np.ndarray) -> np.ndarray:
    r"""Calculate the solid angle subtended by a cone with apex at the origin.

    The solid angle subtended by a cone with apex at the origin is given by

        A = 2 pi (1 - cos(theta))

    where theta is the half-angle of the cone and.

                /|\
               / | \
              /  |_/\
             /   | th\ r
            /    |    \
           /     |     \
          /      |      \
          \______|______/ A = r^2

    Args:
        theta: The half-angle of the cone in radians.

    """
    return 2 * np.pi * (1 - np.cos(theta))


class PelphixUniform(PelphixBase):
    """Class for generating uniformly sampled datasets.

    An individual dataset for each case is created, for convenience of splitting later on without
    having to load the whole JSON.

    May be joined using a Container.

    """

    def __init__(
        self,
        root: Union[str, Path],
        nmdid_root: Union[str, Path],
        pelvis_annotations_dir: Union[str, Path],
        num_val: int = 32,
        scan_name: str = "THIN_BONE_TORSO",
        image_size: tuple[int, int] = (256, 256),
        overwrite: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        bag_world: bool = False,
        num_workers: int = 0,
        num_samples_per_case: int = 100,
        max_wires: int = 7,
        max_screws: int = 7,
        center_views: dict[str, float] = {"ap": 75, "lateral": 10},
        job_queue: Optional[Queue] = None,
        finished_queue: Optional[Queue] = None,
    ):
        """Initialize a PelphixUniform object.

        Args:
            root: The root directory for generating the dataset.
            nmdid_root: The root directory for the NMDID-ARCADE dataset.
            pelvis_annotations_dir: The directory containing the pelvis annotations.
            num_val: The number of validation cases. Defaults to 32.
            scan_name: The name of the scan to use. Defaults to "THIN_BONE_TORSO".
            image_size: The size of the images to generate. Defaults to (256, 256).
            overwrite: Whether to overwrite existing files. Defaults to False.
            cache_dir: The directory to use for caching. Defaults to None.
            num_workers: The number of workers to use for generating the dataset. Defaults to 0 (main thread).
            num_samples_per_case: The number of samples to generate per case. Defaults to 100.
            max_wires: The maximum number of wires to use. Defaults to 7.
            max_screws: The maximum number of screws to use. Defaults to 7.
            center_views: The center views to use for sampling viewing angles, mapped to the
                half-angle of each cone in degrees. One of these is chosen at random for each
                sample, with probability proportional to the solid angle subtended by each sampling
                cone. Defaults to {"ap": 75, "lateral": 10}.
            job_queue: The queue to use for adding jobs, if this is a process. Don't set this manually.
            finished_queue: The queue to use for adding finished jobs, if this is a process. Don't set this manually.
        """
        self.kwargs = locals()
        del self.kwargs["self"]
        del self.kwargs["__class__"]
        del self.kwargs["job_queue"]
        del self.kwargs["finished_queue"]

        super().__init__(
            root=root,
            nmdid_root=nmdid_root,
            pelvis_annotations_dir=pelvis_annotations_dir,
            num_val=num_val,
            scan_name=scan_name,
            image_size=image_size,
            overwrite=overwrite,
            cache_dir=cache_dir,
            bag_world=bag_world,
        )

        self.job_queue = job_queue
        self.finished_queue = finished_queue
        self.num_workers = num_workers
        self.num_samples_per_case = num_samples_per_case
        self.max_wires = max_wires
        self.max_screws = max_screws
        self.center_views: dict[Acquisition, float] = dict(
            (Acquisition(acq), np.radians(theta)) for acq, theta in center_views.items()
        )

        self.case_names = sorted(
            [
                case.name
                for case in self.pelvis_annotations_dir.glob("case-*/")
                if re.match(self.CASE_PATTERN, case.name) is not None
            ]
        )
        num_images = len(self.case_names) * self.num_samples_per_case
        self.name = f"pelphix-uniform_{num_images // 1000:03d}k"
        self.num_cases = len(self.case_names)
        self.num_images = self.num_cases * self.num_samples_per_case
        log.info(f"{self.name}: {self.num_cases} cases, {self.num_images} images")

        self.data_dir = self.root / f"{self.name}"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Don't sync the images directory.
        (self.data_dir / ".nosync").touch()

    # TODO: sample_case(), generate(), and run() methods. Pretty simple. Have to sample more often
    # around AP than lateral. Base it off of solid angle.

    def get_case_annotation_path(self, case_name: str) -> Path:
        return self.data_dir / f"{case_name}.json"

    def get_case_images_dir(self, case_name: str) -> Path:
        return self.data_dir / case_name

    def sample_point(
        self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
    ) -> geo.Point3D:
        """Sample a point uniformly in the given box."""
        return geo.point(np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax]))

    def sample_case(self, case_name: str) -> bool:
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
            Can regress the angle to each standard view separateley, or do multiclass binary classification.
        7. Sample the source-to-patient distance randomly in a reasonable range.
        8. Do the projection and save the image and annotations.

        """
        annotation_path = self.get_case_annotation_path(case_name)
        annotation = self.get_base_annotation()
        images_dir = self.get_case_images_dir(case_name)

        device = self.sample_device()
        wire_catid = self.get_annotation_catid("wire")
        screw_catid = self.get_annotation_catid("screw")

        # Get the volumes
        ct, seg_volumes, seg_meshes, corridors, pelvis_landmarks = self.load_case(case_name)
        wires = [deepdrr.vol.KWire.from_example() for _ in range(self.max_wires)]
        screw_choices = list(get_screw_choices())
        screws: list[Screw] = []
        for screw_idx in range(self.max_screws):
            screw_type = screw_choices[np.random.choice(len(screw_choices))]
            screws.append(screw_type(density=0.06))

        seg_meshes_pv: dict[str, pv.PolyData] = {k: v.as_pv() for k, v in seg_meshes.items()}
        pelvis_mesh_pv: pv.PolyData = (
            seg_meshes_pv["hip_left"] + seg_meshes_pv["hip_right"] + seg_meshes_pv["sacrum"]
        )
        pelvis_bounds = pelvis_mesh_pv.bounds

        # Probability for the number of wires and screws
        wire_probabilities = np.array([0.5] + [0.5 / len(wires)] * len(wires))
        screw_probabilities = np.array([0.5] + [0.5 / len(screws)] * len(screws))
        corridor_probabilities = np.array([0.5] + [0.5 / len(corridors)] * len(corridors))
        corridor_names = ["no-corridor"] + list(corridors.keys())
        wire_corridor_probabilities = np.array([0.1] + [0.9 / len(corridors)] * len(corridors))
        wire_corridor_names = ["no-corridor"] + list(corridors.keys())
        screw_corridor_probabilities = np.array(
            [0.1, 0.2] + [0.7 / len(corridors)] * len(corridors)
        )
        screw_corridor_names = ["no-corridor", "random-wire"] + list(corridors.keys())

        # Get the views and their probabilities
        world_from_APP = self.get_APP(pelvis_keypoints=pelvis_landmarks)
        center_view_directions = dict(
            (view_name, self.get_view_direction(view_name, world_from_APP, corridors))
            for view_name in self.center_views
        )
        center_views = list(self.center_views.keys())
        center_view_probabilities = normalize(
            [solid_angle(self.center_views[view_name]) for view_name in center_views]
        )

        # Get the standard views and their directions
        standard_view_directions = dict(
            (view_name, self.get_view_direction(view_name, world_from_APP, corridors))
            for view_name in STANDARD_VIEWS
        )

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
        seg_names: dict[int, str] = dict()

        # Add the volumes to the projector
        for seg_name, seg_volume in seg_volumes.items():
            track_id = 1000 * self.get_annotation_catid(seg_name) + 0
            seg_projectors[track_id] = Projector(seg_volume, device=device, neglog=True)
            seg_projectors[track_id].initialize()
            seg_names[track_id] = seg_name

        for wire_idx, wire in enumerate(wires):
            track_id = 1000 * wire_catid + wire_idx
            seg_projectors[track_id] = Projector(wire, device=device, neglog=True)
            seg_projectors[track_id].initialize()
            seg_names[track_id] = "wire"

        for screw_idx, screw in enumerate(screws):
            track_id = 1000 * screw_catid + screw_idx
            seg_projectors[track_id] = Projector(screw, device=device, neglog=True)
            seg_projectors[track_id].initialize()
            seg_names[track_id] = "screw"

        # Sample the views
        for image_id in track(
            range(self.num_samples_per_case),
            total=self.num_samples_per_case,
            description=f"Sampling {case_name}...",
        ):
            image_path = images_dir / f"{image_id:09d}.png"

            # Sample the number of wires and screws
            num_wires = np.random.choice(len(wires) + 1, p=wire_probabilities)

            # No more screws than wires
            num_screws = max(np.random.choice(len(screws) + 1, p=screw_probabilities), num_wires)

            # Place screws and wires randomly
            wires_placed = set()
            for wire_idx in range(self.max_wires):
                wire = wires[wire_idx]
                wires_placed.add(wire_idx)
                if wire_idx >= num_wires:
                    wire.place_center(geo.point(999999, 999999, 999999))
                    continue

                # Choose a corridor for the wire
                corridor_name = np.random.choice(wire_corridor_names, p=wire_corridor_probabilities)
                if corridor_name == "no-corridor":
                    wire_tip = self.sample_point(*pelvis_bounds)
                    wire_dir = geo.random.spherical_uniform()
                    wire.align(wire_tip, wire_tip + wire_dir, progress=0)
                else:
                    corridor = corridors[corridor_name]
                    if (
                        corridor_name in [Task.ramus_left, Task.ramus_right, Task.s1, Task.s2]
                        and np.random.rand() < 0.5
                    ):
                        # Flip the corridor
                        corridor = corridor.flip()
                    wire_tip = corridor.startpoint + geo.v(
                        np.clip(np.random.normal(0, 3, size=3), -10, 10)
                    )
                    wire_dir = geo.random.spherical_uniform(
                        center=corridor.get_direction(),
                        d_phi=math.radians(15),
                    )
                    progress = np.random.uniform(-0.1, 0.99)
                    wire.align(wire_tip, wire_tip + wire_dir, progress=progress)

            for screw_idx in range(self.max_screws):
                screw = screws[screw_idx]
                if screw_idx >= num_screws:
                    screw.place_center(geo.point(999999, 999999, 999999))
                    continue

                # Choose a corridor for the screw
                corridor_name = np.random.choice(corridor_names, p=corridor_probabilities)
                if corridor_name == "no-corridor":
                    screw_tip = self.sample_point(*pelvis_bounds)
                    screw_dir = geo.random.spherical_uniform()
                    screw.align(screw_tip, screw_tip + screw_dir, progress=0)
                elif corridor_name == "random-wire" and len(wires_placed) > 0:
                    wire_idx = np.random.choice(list(wires_placed))
                    wires_placed.remove(wire_idx)
                    wire = wires[wire_idx]
                    screw_tip = wire.tip_in_world
                    screw_dir = geo.random.spherical_uniform(
                        center=wire.get_direction(),
                        d_phi=math.radians(15),
                    )
                    progress = np.random.uniform(-0.1, 0.99)
                    screw.align(screw_tip, screw_tip + screw_dir, progress=progress)
                else:
                    corridor = corridors[corridor_name]
                    if (
                        corridor_name in [Task.ramus_left, Task.ramus_right, Task.s1, Task.s2]
                        and np.random.rand() < 0.5
                    ):
                        # Flip the corridor
                        corridor = corridor.flip()
                    screw_tip = corridor.startpoint + geo.v(
                        np.clip(np.random.normal(0, 3, size=3), -10, 10)
                    )
                    screw_dir = geo.random.spherical_uniform(
                        center=corridor.get_direction(),
                        d_phi=math.radians(15),
                    )
                    screw.align(screw_tip, screw_tip + screw_dir, progress=0)

            # Sample a corridor to center the view on
            corridor_name = np.random.choice(corridor_names, p=corridor_probabilities)
            # log.info(f"({image_id}) Selected corridor: {corridor_name}")
            if corridor_name == "no-corridor":
                log.debug(f"pelvis_bounds: {pelvis_bounds}")
                view_center = self.sample_point(*pelvis_bounds)
                log.debug(f"sampled view_center from no-corridor: {view_center}")

            else:
                corridor = corridors[corridor_name]
                view_center = corridor.startpoint.lerp(
                    corridor.endpoint, np.random.uniform(0, 1)
                ) + geo.vector(
                    np.clip(np.random.normal(0, 5, size=3), -15, 15),
                )
                log.debug(f"sampled view_center from {corridor_name}: {view_center}")

            # TODO: figure out why black images

            # Sample a view direction from one of the options in center_views.
            center_view = np.random.choice(center_views, p=center_view_probabilities)
            # log.info(f"({image_id}) Selected center view: {center_view}")
            center_view_direction = center_view_directions[center_view]
            view_dir = geo.random.spherical_uniform(
                center_view_direction,
                self.center_views[center_view],
            )

            # Sample source-to-point distance and up-vector
            source_to_point_fraction = np.random.uniform(0.5, 0.8)
            up_vector = geo.random.spherical_uniform(
                ct.world_from_anatomical @ geo.v(0, 0, 1), math.radians(10)
            )

            # Set the carm view
            device.set_view(
                point=view_center,
                direction=view_dir,
                up=up_vector,
                source_to_point_fraction=source_to_point_fraction,
            )

            # Render the image and all the segmentations
            self.project_image(
                annotation,
                projector=projector,
                device=device,
                image_path=image_path,
                seg_projectors=seg_projectors,
                seg_names=seg_names,
                corridors=corridors,
                pelvis_landmarks=pelvis_landmarks,
                image_id=image_id,
                case_name=case_name,
                standard_view_directions=standard_view_directions,
                # desired_view=geo.ray(view_center, view_dir), # displays pyvista window.
            )

        # Save the annotation
        save_json(annotation_path, annotation)

        return True

    def run(self):
        """While there are procedures to be done, run them.

        Reads a procedure_idx from the queue and samples to the procedure."""

        while True:
            try:
                case_idx = self.job_queue.get(block=False)
                if case_idx is None:
                    return

                success = self.sample_case(case_idx)

                if success:
                    self.finished_queue.put(case_idx)
                else:
                    self.job_queue.put(case_idx)
            except Empty:
                return
            except KeyboardInterrupt:
                return
            except Exception as e:
                log.exception(e)
                raise e

    def generate(self):
        """Generate the dataset."""

        annotation_paths = sorted(list(self.data_dir.glob("*.json")))
        cases_done = set(p.stem for p in annotation_paths)

        if self.overwrite:
            if cases_done:
                log.critical(
                    f"Overwriting {len(cases_done)} existing annotations. Press CTRL+C to cancel."
                )
                input("Press ENTER to overwrite.")

            cases_done = set()
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
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

        # image_id = 0
        # annotation_id = 0
        # annotation = self.get_base_annotation()

        # for case_annotation_path in track(
        #     list(self.tmp_dir.glob("*.json")), description="Merging annotations...."
        # ):
        #     case_annotation = load_json(case_annotation_path)
        #     case_name = case_annotation_path.stem
        #     case_images_dir = self.tmp_dir / case_name

        #     if not case_images_dir.exists():
        #         log.warning(f"Case {case_name} does not exist. Skipping.")
        #         continue

        #     first_image_id = image_id

        #     for image_info in case_annotation["images"]:
        #         image_path = case_images_dir / str(image_info["file_name"])
        #         new_image_path = self.images_dir / f"{case_name}_{image_info['file_name']}"
        #         if image_path.exists() and not new_image_path.exists():
        #             shutil.copy(image_path, new_image_path)
        #         elif new_image_path.exists():
        #             pass
        #         else:
        #             log.error(f"Image {image_path} does not exist. Skipping.")
        #             continue

        #         image_info["file_name"] = str(new_image_path.name)
        #         image_info["first_frame_id"] = first_image_id
        #         image_info["id"] = image_id
        #         annotation["images"].append(image_info)
        #         image_id += 1

        #     for ann in case_annotation["annotations"]:
        #         ann["id"] = annotation_id
        #         ann["image_id"] = first_image_id + ann["image_id"]
        #         annotation["annotations"].append(ann)
        #         annotation_id += 1

        # log.info("Saving annotations...")
        # save_json(self.annotations_dir / f"{self.name}.json", annotation)

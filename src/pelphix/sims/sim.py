"""Class for generating sequences of activities.
"""

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
from omegaconf import OmegaConf

from perphix.data import PerphixBase

from .base import PelphixBase, Case, ImageSize
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


class PelphixSim(PelphixBase, Process):
    """Simulator for generating sequence data."""

    EXCLUDE_CASES = {"case-103485"}

    def __init__(
        self,
        root: Path,
        nmdid_root: Path,
        pelvis_annotations_dir: Path,
        job_queue: Optional[Queue] = None,
        finished_queue: Optional[Queue] = None,
        train: bool = True,
        num_val: int = 10,
        scan_name: str = "THIN_BONE_TORSO",
        image_size: tuple[int, int] = (256, 256),
        corridor_radii: dict[str, float] = 2.5,
        num_procedures: int = 10000,
        overwrite: bool = False,
        cache_dir: Optional[Path] = None,
        skill_factor: tuple[float, float] = (0.5, 1.0),
        view_skill_factor: tuple[float, float] = (0.5, 1.0),
        view_tolerance: dict[str, float] = dict(),
        random_translation_bounds: dict[str, float] = dict(),
        random_angulation_bounds: dict[str, float] = dict(),
        num_workers: int = 1,
        max_procedure_length: int = 1000,
    ):
        """Create the pelvic workflows dataset from NMDID cadavers.

        Args:
            root: Path to the root directory for this dataset. This is where the data will be generated,
                if it has not been already.
            nmdid_root: Path to the NMDID root directory.
            annotations_dir: Path to the annotations data to use.
            queue: The queue for passing procedure indices to the worker processes. This is one-way communication:
                the main process puts indices in the queue, and the worker processes get them out.
                This is only None for the main process.
            train: Whether to use/generate the training or validation data.
            num_val: Number of CT images to use for validation data.
            scan_name: Name of the scan to use.
            image_size: Size of the generated images.
            corridor_radius: Radius of the corridor in mm.
            overwrite: Whether to overwrite existing data and start dataset generation from the beginning.
            view_tolerance: A dict mapping acquisition names to angular tolerances in degrees.
            random_translation_bounds: A dict with the following structure:
                {
                    "min": {
                        "device": {
                            "ap": float,
                            "lateral": float,
                            ...
                        },
                        "wire": {
                            "ap": float,
                            "lateral": float,
                            ...
                        }
                    },
                    "max": {
                        ...
                    }
                }
                where the lowest level keys are the names of views and the values are the bounds for the random
                translation. The bounds are in mm.
            random_angulation_bounds: A dict with the same structure as random_translation_bounds, but with the
                bounds for random angulation in degrees.
            num_workers: Number of worker processes to use for generating data.
            max_procedure_length: Maximum number of steps in a procedure.

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
            train=train,
            num_val=num_val,
            scan_name=scan_name,
            image_size=image_size,
            overwrite=overwrite,
            cache_dir=cache_dir,
        )

        self.job_queue = job_queue
        self.finished_queue = finished_queue
        self.corridor_radii = corridor_radii
        self.skill_factor = tuple(skill_factor)
        self.view_skill_factor = tuple(view_skill_factor)
        self.view_tolerance = deepdrr.utils.radians(dict(view_tolerance), degrees=True)
        self.random_translation_bounds = dict(random_translation_bounds)
        self.random_angulation_bounds = deepdrr.utils.radians(
            dict(random_angulation_bounds), degrees=True
        )
        self.num_workers = num_workers
        self.max_procedure_length = max_procedure_length

        log.debug(f"pelvis_annotations_dir: {self.pelvis_annotations_dir}")

        case_names = sorted(
            [
                case.name
                for case in self.pelvis_annotations_dir.glob("case-*/")
                if re.match(self.CASE_PATTERN, case.name) is not None
                and case.name not in self.EXCLUDE_CASES
            ]
        )
        if self.train:
            self.name = f"pelphix_{num_procedures:06d}_train"
            self.num_procedures = num_procedures * (len(case_names) - num_val) // len(case_names)
            self.case_names = case_names[num_val:]
        else:
            self.name = f"pelphix_{num_procedures:06d}_val"
            self.num_procedures = num_procedures * num_val // len(case_names)
            self.case_names = case_names[:num_val]

        log.info(f"{self.name}: train={train} num_procedures={self.num_procedures}")

        self.num_cases = len(self.case_names)

        self.annotations_dir = self.root / "annotations"
        self.images_dir = self.root / self.name
        self.tmp_dir = self.root / f"tmp_{self.name}"

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Zip the images rather than upload them directly.
        (self.images_dir / ".nosync").touch()
        (self.tmp_dir / ".nosync").touch()

    def evaluate_view(self, state: SimState, device: SimpleDevice, view: geo.Ray3D) -> float:
        """Evaluate a view.

        Args:
            device: The device.
            view: The view to evaluate.

        Returns:
            True if the view is close enough to desired, False otherwise.
        """
        assert state.acquisition not in ["start", "end"]

        index_from_world = device.index_from_world
        p_index = index_from_world @ view.p
        image_center = geo.p(self.image_size[1] / 2, self.image_size[0] / 2)

        return (p_index - image_center).norm() < min(self.image_size) * 0.4 and abs(
            view.angle(device.principle_ray_in_world)
        ) < self.view_tolerance[state.acquisition]

    """Sampling functions.

    These functions should use `self` only to access the dataset parameters, and not to access
    the state of the simulation. This will enable parallelization, hopefully.
    """

    def sample_next_view(
        self,
        state: SimState,
        device: SimpleDevice,
        current_view: geo.Ray3D,
        desired_view: geo.Ray3D,
        skill_factor: float = 0.5,
    ) -> geo.Ray3D:
        """Sample a view that is closer to the desired view than the current view.

        For the translation, the desired point is sampled from a uniform distribution centered
        on the desired point with a width of `max(skill_factor * |p - p0|, min_translation_width)`.

        For the rotation, the desired view is sampled from a uniform distribution on the sphere,
        centered on the desired view, with a solid angle given by `skill_factor * v.angle(v0)`.

        Args:
            device: The device.
            current_view: The current view.
            desired_view: The desired view.
            skill_factor: The skill factor. Determines how much closer the view is likely to get.

        """

        view_name = str(state.acquisition)

        dist = (current_view.p - desired_view.p).norm()
        bound = np.clip(
            skill_factor * dist,
            self.random_translation_bounds["min"]["device"][view_name],
            self.random_translation_bounds["max"]["device"][view_name],
        )
        p = geo.random.uniform(center=desired_view.p, radius=bound)

        # First, check if the current point is in the middle of the image (roughly).
        index_from_world = device.index_from_world
        image_center = geo.p(self.image_size[1] / 2, self.image_size[0] / 2)
        if (index_from_world @ desired_view.p - image_center).norm() > min(self.image_size) * 0.4:
            # if the position is bad, only adjust that. Do the angle
            return geo.ray(p, current_view.n)

        # If the position is good, adjust the angle.
        # skill factor increases for angle (more skill needed)
        # angular_skill_factor = 1 - (1 - skill_factor) * 0.5
        log.debug(f"Acquisition counter for {view_name}: {state.acquisition_counter[view_name]}")
        if state.acquisition_counter[view_name] > 1:
            log.debug("Acquisition counter > 2, reducing angular skill factor")
            angular_skill_factor = skill_factor * 0.2
        else:
            angular_skill_factor = skill_factor
        angle = current_view.angle(desired_view)
        angle_bound = np.clip(
            angular_skill_factor * angle,
            self.random_angulation_bounds["min"]["device"][view_name],
            self.random_angulation_bounds["max"]["device"][view_name],
        )
        log.debug(
            f"Sampling next view within {bound:.2f}mm, {math.degrees(angle_bound):.1f}{DEGREE_SIGN}"
        )
        n = geo.random.spherical_uniform(center=desired_view.n, d_phi=angle_bound)
        return geo.ray(p, n)

    # Fraction of the annotatation that determines max wire insertion.
    wire_insertion_fraction = dict(
        s1_left=0.7,
        s1_right=0.7,
        s1=0.7,
        s2=0.7,
        ramus_left=0.95,
        ramus_right=0.95,
        teardrop_left=0.95,
        teardrop_right=0.95,
    )

    # Fraction of the annotaiton length that is the max for the screw.
    screw_insertion_fraction = dict(
        s1_left=0.7,
        s1_right=0.7,
        s1=0.7,
        s2=0.7,
        ramus_left=0.95,
        ramus_right=0.95,
        teardrop_left=0.65,
        teardrop_right=0.65,
    )

    # Mapping to which meshes the corridor might start on.
    corridor_mesh_names = dict(
        s1_left=["hip_left"],
        s1_right=["hip_right"],
        s1=["hip_left", "hip_right"],
        s2=["hip_left", "hip_right"],
        ramus_left=["hip_left"],
        ramus_right=["hip_right"],
        teardrop_left=["hip_left"],
        teardrop_right=["hip_right"],
    )

    def evaluate_wire_position(
        self,
        wire: Tool,
        corridor: Cylinder,
        device: SimpleDevice,
        false_positive_rate: float = 0.2,
    ) -> bool:
        """Evaluate if the wire is in the correct position.

        Args:
            wire: The wire.
            state: The state of the workflow.
            false_positive_rate: The base false positive rate. This is lerped to 0 for progress > 0.5,
                and set to 0 for wire_angle > 5 degrees.

        Returns:
            True if the wire is in the correct position, False otherwise.
        """

        if np.random.uniform() < false_positive_rate:
            return True

        principle_ray = device.principle_ray_in_world
        if (
            corridor_to_principle_ray_angle := abs(corridor.centerline.angle(principle_ray))
        ) < THIRTY_DEGREES:
            # View is down the barrel of the corridor
            log.debug("Evaluating wire down the barrel")
            angle_error = abs(wire.centerline_in_world.angle(corridor.centerline))
            false_positive_prob = false_positive_rate * (
                1 - corridor_to_principle_ray_angle / THIRTY_DEGREES
            )
            if angle_error > math.radians(3):
                return False
            elif np.random.uniform() < false_positive_prob:
                log.debug(f"False positive from: {false_positive_prob}")
                return True

            p0 = corridor.startplane().meet(wire.centerline_in_world)
            p1 = corridor.endplane().meet(wire.centerline_in_world)
            if (p0 - corridor.startpoint).norm() < corridor.radius and (
                p1 - corridor.endpoint
            ).norm() < corridor.radius:
                log.debug("Wire is in the corridor")
                return True
            else:
                log.debug("Wire is not in the corridor")
                return False

        else:
            # View is perp to wire. Check 2D projections.
            index_from_world = device.index_from_world

            # Get the projections
            startpoint_in_index = index_from_world @ corridor.startpoint
            endpoint_in_index = index_from_world @ corridor.endpoint
            tip_in_index = index_from_world @ wire.tip_in_world
            corridor_projection = corridor.project(index_from_world)
            wire_in_index = index_from_world @ wire.centerline_in_world
            corridor_in_index = index_from_world @ corridor.centerline

            # Check if the wire intersects the line segments
            if corridor_projection.start.intersects(
                wire_in_index
            ) and corridor_projection.end.intersects(wire_in_index):
                return True

            # Check the angle
            angle_error = wire_in_index.angle(corridor_in_index)
            if angle_error > math.radians(3):
                return False

            # Check how far along it is
            progress = (tip_in_index - startpoint_in_index).norm() / (
                startpoint_in_index - endpoint_in_index
            ).norm()
            progress = np.clip(progress, 0, 1)
            # Decrease the false positive rate based on progress
            false_positive_prob = np.clip(
                false_positive_rate * (1 - progress / 0.5), 0, false_positive_rate
            )

            log.debug(
                f"corr_to_principle_ray_angle: {math.degrees(corridor_to_principle_ray_angle):.2f}{DEGREE_SIGN}"
            )

            if np.random.uniform() < false_positive_prob:
                log.debug(f"False positive from: {false_positive_prob}")
                return True

            return False

    def evaluate_insertion(
        self, state: SimState, tool: Tool, corridor: Cylinder, device: SimpleDevice
    ) -> bool:
        """Evaluate if the wire is inserted, based on its projection.

        Args:
            tool: The tool.
            corridor: The corridor.
            device: The device.

        Returns:
            True if the wire is fully inserted, False otherwise.

        """
        principle_ray = device.principle_ray_in_world
        if (
            corridor_to_principle_ray_angle := abs(corridor.centerline.angle(principle_ray))
        ) < THIRTY_DEGREES:
            # View is down the barrel of the corridor. We can't tell anything.
            return False

        # index_from_world = device.index_from_world

        # # Get the projections
        # corridor_projection = corridor.project(index_from_world)
        # wire_in_index = index_from_world @ geo.segment(tool.base_in_world, tool.tip_in_world)

        # # Check if the wire intersects the line at the end of the corridor
        # if wire_in_index.intersects(corridor_projection.end.line()):
        #     return True

        index_from_world = device.index_from_world
        tool_in_index = index_from_world @ tool.centerline_in_world
        corridor_projection = corridor.project(index_from_world)

        p_index = corridor_projection.start.line().meet(tool_in_index)
        q_index = corridor_projection.end.line().meet(tool_in_index)

        if state.task.is_wire():
            tip_index = index_from_world @ tool.tip_in_world
            progress = (tip_index - p_index).dot(q_index - p_index) / (q_index - p_index).normsqr()
            return progress >= self.wire_insertion_fraction[state.task] - 0.02
        else:
            # For the screw, want to stop actually when the base is at the startpoint.
            base_index = index_from_world @ tool.base_in_world  # screw base in index
            progress = (base_index - p_index).dot(q_index - p_index) / (q_index - p_index).normsqr()
            log.debug(f"screw progress: {progress}")
            return progress >= -0.02

    def position_wire(
        self,
        wire: Tool,
        *,
        state: SimState,
        seg_meshes: dict[str, Mesh],
        corridor: Cylinder,
        device: SimpleDevice,
        skill_factor: float = 0.5,
    ) -> None:
        """Re-positiong the wire to be closer to the corridor.

        Args:
            wire: The wire tool, aligned along the current trajectory. This is modified in place.
            state: The current state.
            seg_meshes: The segmentations.
            corridors: The corridor being aligned to.
            device: The C-arm device.
            skill_factor: The skill factor of the surgeon. This is a number between 0 and 1, where 0 is
                perfect and 1 means each new position is potentially as bad as the previous.

        """
        assert state.task not in [Task.start, Task.end]

        # 1. Get the projection of the wire's trajectory onto the image.
        # 2. Get the corridor's projection onto the image, as a bounding box.
        # 3. Check if the wire's projected trajectory will exit the projected corridor through the
        #    back or the sides.
        # 4. If the back, then DONE.
        # 4. Otherwise, sample a new trajectory by reducing the tip location or the projected angle.
        #    That is, rotate the trajectory about the ray from the X-ray source to the startpoint.
        index_from_world = device.index_from_world
        principle_ray = device.principle_ray_in_world

        # For the startpoint, sample from a uniform box based on the skill factor, with the best
        # accuracy being within 0.5mm (just pivoting the wire).
        view_name = str(state.acquisition)
        corridor_name = state.task.get_wire()
        tip_distance = (wire.tip_in_world - corridor.startpoint).norm()
        tip_bound = np.clip(
            skill_factor * tip_distance,
            self.random_translation_bounds["min"]["wire"][view_name],
            self.random_translation_bounds["max"]["wire"][view_name],
        )
        free_point = corridor.startpoint + geo.v(np.random.uniform(-tip_bound, tip_bound, size=3))

        # # project the free point onto the mesh.
        startpoint = free_point
        # cdist = np.inf
        # for mesh_name in self.corridor_mesh_names[corridor_name]:
        #     mesh = seg_meshes[mesh_name]
        #     point, dist, _ = mesh.query_point(free_point)
        #     if dist < cdist:
        #         startpoint = point
        #         cdist = dist
        #         break

        if (
            corridor_to_principle_ray_angle := abs(corridor.centerline.angle(principle_ray))
        ) < THIRTY_DEGREES:
            # If the corridor is close to the principle ray, then adjust by bringing closer to
            # desired trajectory in 3D.
            angle_bound = np.clip(
                skill_factor * corridor_to_principle_ray_angle,
                self.random_angulation_bounds["min"]["wire"][view_name],
                self.random_angulation_bounds["max"]["wire"][view_name],
            )
            direction = geo.random.spherical_uniform(corridor.get_direction(), angle_bound)
        else:
            # Get the angle between the wire and the corridor in the image.
            corridor_ray_in_index = index_from_world @ corridor.center_ray
            wire_ray_in_index = index_from_world @ geo.ray(
                wire.tip_in_world, wire.tip_in_world - wire.base_in_world
            )
            wire_to_corridor_angle = corridor_ray_in_index.angle(wire_ray_in_index)
            angle_bound = np.clip(
                skill_factor * wire_to_corridor_angle,
                self.random_angulation_bounds["min"]["wire"][view_name],
                self.random_angulation_bounds["max"]["wire"][view_name],
            )
            new_wire_to_corridor_angle = np.random.uniform(-angle_bound, angle_bound)

            sign = wire_ray_in_index.n.cross(corridor_ray_in_index.n).z
            sign = sign / abs(sign)

            # log.warning(f"Not rotating by second random amount, for debugging")
            rotvec = (
                sign * wire_to_corridor_angle + new_wire_to_corridor_angle
            ) * principle_ray.hat()

            # perturb a little bit about the perpendicular direction.
            if wire.centerline_in_world.angle(principle_ray) > THIRTY_DEGREES:
                max_perp_angle = wire_to_corridor_angle / 10
                perp_rotvec = (
                    np.random.uniform(-max_perp_angle, max_perp_angle)
                    * wire.centerline_in_world.get_direction().cross(principle_ray).hat()
                )
            else:
                perp_rotvec = geo.v(0, 0, 0)

            direction = wire.tip_in_world - wire.base_in_world
            direction = direction.rotate(rotvec)

            if not np.isclose(perp_rotvec.norm(), 0):
                direction = direction.rotate(perp_rotvec)
                # wire.rotate(perp_rotvec, startpoint)

        wire.orient(startpoint, direction)

    def sample_wire_advancement(
        self,
        state: SimState,
        tool: Tool,
        corridor: Cylinder,
        device: SimpleDevice,
    ) -> float:
        """Sample the advancement of the wire.

        Args:
            tool: The tool.
            corridor: The corridor.
            device: The device.
            wire: Whether the tool is a wire.

        Returns:
            The advancement of the tool in mm.
        """

        assert state.task.is_wire(), f"Cannot sample wire advancement for {state.task}"

        index_from_world = device.index_from_world
        tool_in_index = index_from_world @ tool.centerline_in_world
        corridor_projection = corridor.project(index_from_world)
        p_index = corridor_projection.start.line().meet(tool_in_index)
        q_index = corridor_projection.end.line().meet(tool_in_index)
        tip_index = index_from_world @ tool.tip_in_world

        progress = (tip_index - p_index).dot(q_index - p_index) / (q_index - p_index).normsqr()
        if progress >= self.wire_insertion_fraction[state.task]:
            # Wire is fully inserted; continue.
            return 0

        log.debug(f"Wire progress: {progress:.2f}")

        # For wires, sample less. But be at least 10% of the way through. Otherwise not really insertion.
        if progress < 0.05:
            new_progress = np.random.uniform(0.05, 0.15)
        else:
            new_progress = np.clip(
                np.random.uniform(progress + 0.1, progress + 0.3),
                0.05,
                self.wire_insertion_fraction[state.task],
            )
        advancement = (new_progress - progress) * corridor.length()
        log.debug(f"Wire advancement: {advancement:.2f}")

        return advancement

    def sample_screw_advancement(
        self,
        state: SimState,
        screw: Tool,
        corridor: Cylinder,
    ) -> float:
        """Sample screw advancement.

        Args:
            state: The state of the simulation.
            screw: The screw.
            corridor: The corridor.

        Returns:
            The advancement of the screw in mm.
        """
        startpoint = corridor.startplane().meet(screw.centerline_in_world)
        trajectory = screw.tip_in_world - screw.base_in_world

        # Advancement to bring tip of the screw to the startpoint.
        tip_advancement = (startpoint - screw.tip_in_world).dot(trajectory) / trajectory.norm()
        log.debug(f"tip_advancement: {tip_advancement}")
        if tip_advancement > 1:
            # Haven't taken the shot with the screw up against the bone yet,
            # so just advance the tip to the bone.
            return tip_advancement

        # Advancement to bring base of the screw to the startpoint.
        advancement = (startpoint - screw.base_in_world).dot(trajectory) / trajectory.norm()
        advancement = np.random.uniform(5, advancement)
        log.debug(f"advancement: {advancement}")
        return advancement

    def get_procedure_name(self, procedure_idx: int) -> str:
        """Get the name of the procedure."""
        return f"{procedure_idx:09d}"

    def get_tmp_annotation_path(self, procedure_idx: int) -> Path:
        """Get the path to the procedure annotation file."""

        # Write the procedure-local annotation (with no awareness of global frame ids)
        annotation_path = self.tmp_dir / f"{self.get_procedure_name(procedure_idx)}.json"
        return annotation_path

    def get_tmp_images_dir(self, procedure_idx: int) -> Path:
        images_dir = self.tmp_dir / f"{self.get_procedure_name(procedure_idx)}"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(exist_ok=True, parents=True)
        return images_dir

    def sample_procedure(self, procedure_idx: int) -> bool:
        """Sample a procedure and write the resulting images/annotations/sequences.

        Writes everything to self.tmp_dir / f"{procedure_idx:09d}", for parallelization. Then at the
        end (or by a separate process), all the images and annotations are moved to the right spot
        and updated with their global ids.

        Args:
            first_frame_id: The id of the first frame of the procedure sequence.

        Returns:
            True if the procedure was successfully sampled, False otherwise.

        The annotation saved by this function is local to the procedure. It has the following
        structure and gets saved to f"{procedure_idx:09d}.json"}:

        {
            "info": {
                "description": "The description of the dataset.",
                "url": "The URL of the dataset.",
                "version": "The version of the dataset.",
                "year": "The year of the dataset.",
                "contributor": "The contributor of the dataset.",
                "date_created": "The date the dataset was created.",
            },
            "licenses": [
                {
                    "url": "The URL of the license.",
                    "id": "The ID of the license.",
                    "name": "The name of the license.",
                },
            ],
            "images": [
                {
                    "license": "The ID of the license.",
                    "file_name": "The name of the image file, which can be found in the tmp/{procedure_idx} dir",
                    "height": "The height of the image.",
                    "width": "The width of the image.",
                    "date_captured": "The date the image was captured.",
                    "id": "The ID of the image IN THE PROCEDURE. This gets changed to a global ID when the procedure is added to the final dataset.",
                    "frame_id": "The ID of the frame within the sequence. (This is not changed when finalized.)",
                    "seq_length": "The length of the sequence. (This is not known in the procedures, so it is changed when the procedure is finished (but before integrating into global dataset).)",
                    "first_frame_id": "The ID of the first image in the sequence IN THE PROCEDURE, which is always 0 in the single-procedure datasets. This gets changed to the global id of the first frame in the sequence when the procedure is added to the final dataset.",
                    "case_name": "The name (case-XXXXXX) of the case.",
                },
                ...
            ],
            "annotations": [
                {
                    "segmentation": {
                        "size": [height, width],
                        "counts": "The run-length encoding of the segmentation.",
                    }
                    "area": "The area of the object.",
                    "iscrowd": "Whether the object is a crowd.",
                    "image_id": "The ID of the image IN THE PROCEDURE. This gets changed to a global ID when the procedure is added to the final dataset.",
                    "bbox": "The bounding box of the object.",
                    "category_id": "The ID of the category.",
                    "id": "The ID of the annotation.",
                    "track_id": "A unique ID for this instance of the object, constant over the procedure.",
                },
                ...
            ],
            "sequences": [
                {
                    "id": "The ID of the sequence.",
                    "seq_category_id": "The ID of the sequence category.",
                    "seq_length": "The length of the sequence.",
                    "first_frame_id": "The ID of the first image in the sequence IN THE PROCEDURE, which is always 0 in the single-procedure datasets. This gets changed to the global id of the first frame in the sequence when the procedure is added to the final dataset.",
                },
                ...
            ],
            "categories": [
                {
                    "supercategory": "The supercategory of the category.",
                    "id": "The ID of the category.",
                    "name": "The name of the category.",
                },
                ...
            ],
            "seq_categories": [
                {
                    "supercategory": "The supercategory of the sequence category.",
                    "id": "The ID of the category.",
                    "name": "The name of the category.",
                }

        }

        """

        # TODO: if multiprocessing, lock this.
        case_name = self.case_names[procedure_idx % len(self.case_names)]

        log.info(f"Sampling procedure {procedure_idx} for case {case_name}")

        annotation_path = self.get_tmp_annotation_path(procedure_idx)
        annotation = self.get_base_annotation()
        images_dir = self.get_tmp_images_dir(procedure_idx)
        # Sample the patient and all related objects.
        ct, seg_volumes, seg_meshes, corridors, pelvis_landmarks = self.load_case(case_name)
        world_from_APP = self.get_APP(pelvis_keypoints=pelvis_landmarks)

        # log.warning(f"TODO: remove this hard-coded false positive rate.")
        false_positive_rate = np.random.uniform(0.05, 0.15)

        # sample the skill factor
        # smaller skill factor is more skilled
        skill_factor = np.random.uniform(*self.skill_factor)
        view_skill_factor = np.random.uniform(*self.view_skill_factor)
        # log.info(f"Sampling skill factor: {skill_factor}")
        # log.warning(f"TODO: remove this hard-coded skill factor.")
        # skill_factor = 0.1
        # view_skill_factor = 0.8

        # Sample the device parameters randomly.
        device = self.sample_device()
        current_view = geo.ray(
            geo.random.uniform(pelvis_landmarks["r_gsn"].lerp(pelvis_landmarks["l_gsn"], 0.5), 100),
            geo.random.spherical_uniform(
                self.get_view_direction("ap", world_from_APP, corridors), FIVE_DEGREES
            ),
        )
        device.set_view(
            *current_view,
            ct.world_from_anatomical @ geo.v(0, 0, 1),
            source_to_point_fraction=np.random.uniform(0.65, 0.75),
        )

        wire_catid = self.get_annotation_catid("wire")
        screw_catid = self.get_annotation_catid("screw")

        # Sample the procedure.
        # log.warning(f"TODO: remove this hard-coded procedure length.")
        max_corridors = len(corridors)
        # max_corridors = np.random.randint(min(len(corridors), 4), len(corridors) + 1)
        log.info(f"Sampling {max_corridors} corridors.")

        # Wires and screws accessed by the corresponding corridor name.
        wires: dict[str, Tool] = dict()
        screws: dict[str, Screw] = dict()
        wire_track_ids: dict[str, int] = dict()
        screw_track_ids: dict[str, int] = dict()

        # Mapping to whether the wire has been moved into view at start for that corridor.
        wire_placed: dict[str, bool] = dict()
        screw_placed: dict[str, bool] = dict()

        # initialize the above.
        for i, name in enumerate(corridors):
            corr = corridors[name]
            wire = deepdrr.vol.KWire.from_example()
            insertable_length = corr.length() * self.screw_insertion_fraction[name]
            screw_choices = list(
                get_screw_choices(insertable_length) - get_screw_choices(insertable_length - 20)
            )
            screw_type = screw_choices[np.random.choice(len(screw_choices))]
            screw = screw_type(density=0.06)
            wire.place_center([99999, 99999, 99999])
            screw.place_center([99999, 99999, 99999])
            wire_track_id = 1000 * wire_catid + i
            screw_track_id = 1000 * screw_catid + i
            wires[name] = wire
            screws[name] = screw
            wire_track_ids[name] = wire_track_id
            screw_track_ids[name] = screw_track_id
            wire_placed[name] = False
            screw_placed[name] = False

        intensity_upper_bound = np.random.uniform(2, 8)
        projector = Projector(
            [ct, *wires.values(), *screws.values()],
            device=device,
            neglog=True,
            step=0.05,
            intensity_upper_bound=intensity_upper_bound,
            attenuate_outside_volume=True,
        )

        # Initialize the main projector.
        projector.initialize()

        # Mapping from track_id to projector.
        seg_projectors: dict[int, Projector] = dict()
        seg_names: dict[int, str] = dict()

        # Add the anatomy segmentations by track_id, and initialize.
        for seg_name, v in seg_volumes.items():
            track_id = 1000 * self.get_annotation_catid(seg_name) + 0
            seg_projector = Projector(v, device=device, neglog=True)
            seg_projector.initialize()
            seg_projectors[track_id] = seg_projector
            seg_names[track_id] = seg_name

        # Add the wires by track_id, but don't initialize until needed.
        for name in corridors:
            track_id = wire_track_ids[name]
            wire = wires[name]
            seg_projectors[track_id] = Projector(wire, device=device, neglog=True)
            seg_names[track_id] = "wire"

        # Add the screws by track_id, but don't initialize until needed.
        for name in corridors:
            track_id = screw_track_ids[name]
            screw = screws[name]
            seg_projectors[track_id] = Projector(screw, device=device, neglog=True)
            seg_names[track_id] = "screw"

        # Initialize projectors
        log.info(f"Initalized {len(seg_projectors)} projectors.")

        state = SimState(max_corridors=max_corridors)

        # Mapping from the supercategory to the length of the sequence in that category.
        # So contains counts for the "task", "activity", "acquisition", and "frame" categories.
        sequence_counts = Counter()
        running_frame_state: Optional[FrameState] = None

        frame_id = 0
        while not (state := state.next()).is_finished() and frame_id < self.max_procedure_length:
            log.info(f"{state}")

            if state.task == Task.start:
                # Start of the procedure
                continue

            # A task has been started. Get the wire and screw tasks for the current corridor
            # (regardless of if task is for screw/wire).
            corridor_name = state.task.get_wire()
            corridor = corridors.get(corridor_name)
            wire = wires[corridor_name]
            screw = screws[corridor_name]

            if not seg_projectors[wire_track_ids[corridor_name]].initialized:
                # Initialize the wire and screw projectors.
                seg_projectors[wire_track_ids[corridor_name]].initialize()
            if not seg_projectors[screw_track_ids[corridor_name]].initialized:
                seg_projectors[screw_track_ids[corridor_name]].initialize()

            if state.activity == Activity.start:
                # Nothing to do here but continue.
                continue
            elif (
                state.activity == Activity.insert_wire
                and state.get_previous_activity() == Activity.position_wire
                and wire_placed[corridor_name]
            ):
                # This is the first insert_wire image, so the wire hasn't been advanced yet.
                # Advance it.
                advancement = self.sample_wire_advancement(state, wire, corridor, device)
                log.debug(f"Advancing wire by {advancement} (previous activity was positioning)")
                wire.advance(advancement)

            # Pre-image assessments, basically of the previous image.
            if state.acquisition == Acquisition.start:
                # TODO: I think here just evaluate whether the wire/screw looks good. This is just evaluating the previous image, since we are continuing directly after

                # So if the wire does not look good, and we're in insertion, it will go back to
                # positioning, even though it's about to start a new acquisition, so it will be
                # about to go to fluoro-hunting, potentially. Then in the state machine, when
                # sampling an activity... We only want to change to positioning when the wire is
                # retracted. TODO: figure out if this is a big deal, or if we can correct for it. So
                # the issue is if the activity is not evaluated here to be good, then it will cause
                # the state machine to go back to positioning, even though it's about to start a new
                # acquisition. BUT if it's in insertion, and it goes to fluoro-hunting, then it will
                # go back to positioning as soon as a bad wire is detected. One solution is that
                # wires always look good if we're fluoro-hunting.

                # Actual solution: fluoro hunting is an activity. When you go into fluoro-hunting,
                # it necessessitates a new acquisition, so that makes sense. This will change a lot
                # of the structure though, since it will change the sequences in the dataset. But
                # for the paper, it's probably a minor enough detail that no one will care. Worth
                # doing? Would avoid a lot of headache down the road. Don't change the dataset
                # structure, just change how sequences are recorded so the period during
                # fluoro-hunting continues whatever activity was happening before.
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                if state.activity == Activity.insert_wire and wire_placed[corridor_name]:
                    state.wire_looks_inserted = self.evaluate_insertion(
                        state, wire, corridor, device
                    )
                elif state.activity == Activity.insert_screw and screw_placed[corridor_name]:
                    state.screw_looks_inserted = self.evaluate_insertion(
                        state,
                        screw,
                        corridor,
                        device,
                    )
                continue

            # Need to sample the desired views for the current acquisitions deterministically.
            desired_view = geo.ray(
                corridor.midpoint,
                self.get_view_direction(state.acquisition, world_from_APP, corridors),
            )

            # log.debug(
            #     f"desired_view in anatomical: {ct.anatomical_from_world @ desired_view.p}, {ct.anatomical_from_world @ desired_view.n}"
            # )
            direction_in_anatomical = ct.anatomical_from_world @ desired_view.n
            if state.acquisition != Acquisition.lateral and direction_in_anatomical.y < 0:
                log.debug(f"desired_view is in the wrong direction. Flipping it.")
                desired_view = geo.ray(desired_view.p, -desired_view.n)

            # TODO: figure out here where we need to see if the view is good and if we need to
            # sample a new view. If the view is good, then we can just use the current view and go
            # straight to assessment from start, by setting view_looks_good = True and continuing.
            # So if we've gotten here, we have a new acquisition in which the object does not look good.
            # If this is frame.start,
            if state.frame == Frame.start:
                # Evaluate the view and whether it will be an assessment or not.
                state.view_looks_good = self.evaluate_view(state, device, desired_view)
                continue
            elif state.frame == Frame.fluoro_hunting:
                # The view didn't look good, so sample a new one, and determine whether it looks good.
                current_view = self.sample_next_view(
                    state,
                    device,
                    current_view,
                    desired_view,
                    skill_factor=view_skill_factor,
                )
                device.set_view(
                    *current_view,
                    up=ct.world_from_anatomical @ geo.random.spherical_uniform(d_phi=FIVE_DEGREES),
                    source_to_point_fraction=np.random.uniform(0.6, 0.75),
                )
                state.view_looks_good = self.evaluate_view(state, device, desired_view)
                if state.view_looks_good:
                    # The view looks good, so we go to assessment to actually take this image.
                    continue
            elif state.frame == Frame.assessment:
                log.debug(
                    f"Assessment with {state}, {wire_placed[corridor_name]}, state.previous={state.previous}"
                )

                if (
                    state.task.is_wire()
                    and not wire_placed[corridor_name]
                    and state.get_previous_frame() == Frame.assessment
                ):
                    # Start of a wire task, after a view has been achieved, but a wire has not yet been placed. Sample the initial wire position.
                    # TODO: only do this if the view looks good. Sampling the initial wire position shouldn't be done here.
                    log.debug(f"Sampling initial wire position for {corridor_name}")
                    wire.orient(
                        geo.random.normal(
                            corridor.startpoint
                            - corridor.get_direction().hat() * np.random.uniform(0, 0.01),
                            scale=4,
                            radius=10,
                        ),
                        geo.random.spherical_uniform(
                            corridor.get_direction(), d_phi=FIFTEEN_DEGREES
                        ),
                    )
                    wire_placed[corridor_name] = True
                elif state.task.is_screw() and not screw_placed[corridor_name]:
                    wire_looks_good = self.evaluate_wire_position(
                        wire, corridor, device, false_positive_rate=false_positive_rate
                    )
                    if not wire_looks_good:
                        # We were going to do screw insertion, but turns out the wire doesn't look good.
                        # So need to go back to wire positioning and continue.
                        # This is a wire reset.
                        state.fix_wire = True
                        wire_placed[corridor_name] = False
                        continue
                    # Start of a screw insertion.
                    # Goes over the corresponding wire.
                    screw.orient(
                        wire.centerline_in_world.meet(corridor.startplane()),
                        wire.tip_in_world - wire.base_in_world,
                        distance=0,
                    )
                    screw_placed[corridor_name] = True

            image_path = images_dir / f"{frame_id:09d}_{'-'.join(state.values())}.png"

            ##################### Taking the image #####################
            t0 = time.time()
            self.project_image(
                annotation,
                projector=projector,
                device=device,
                image_path=image_path,
                seg_projectors=seg_projectors,
                seg_names=seg_names,
                corridors=corridors,
                pelvis_landmarks=pelvis_landmarks,
                image_id=frame_id,
                case_name=case_name,
            )
            t1 = time.time()

            # Update the sequence counts and, if one of them has changed, update the sequences.
            # Mapping from the supercategory to the length of the sequence in that category.
            # So contains counts for the "task", "activity", "acquisition", and "frame" categories.
            frame_state = state.framestate()
            for supercategory in ["task", "activity", "acquisition", "frame"]:
                if running_frame_state is None:
                    pass
                elif getattr(frame_state, supercategory) != getattr(
                    running_frame_state, supercategory
                ):
                    # this supercategory in the running state is done. Save out the sequence and
                    # reset the counter
                    seq_name = getattr(running_frame_state, supercategory)
                    if isinstance(seq_name, Task):
                        seq_name = seq_name.get_wire()
                    seq = {
                        "id": len(annotation["sequences"]),
                        "seq_length": sequence_counts[supercategory],
                        "first_frame_id": frame_id - sequence_counts[supercategory],
                        "seq_category_id": self.get_sequence_catid(supercategory, seq_name),
                    }
                    annotation["sequences"].append(seq)
                    sequence_counts[supercategory] = 0

                # Increment the counter for this (or the new) supercategory.
                sequence_counts[supercategory] += 1

            running_frame_state = frame_state

            log.info(
                f"=============================== Acquired image {frame_id:03d} in {t1 - t0:.02f}s. ==============================="
            )
            frame_id += 1

            if state.frame == Frame.fluoro_hunting:
                # If we're in fluoro hunting, we need to sample a new view.
                # TODO: potentially check the wire here.
                continue

            assert state.frame == Frame.assessment, f"unexpected frame: {state.frame}"

            if state.activity == Activity.position_wire:
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                if state.wire_looks_good:
                    # Set need_new_view? Nahhhh, should be fine just by setting end probability very low.
                    pass
                else:
                    # If the wire doesn't look good, we need to reposition it.
                    # Position the wire and evaluate the position to determine if it looks good for the next state.
                    self.position_wire(
                        wire=wire,
                        state=state,
                        seg_meshes=seg_meshes,
                        corridor=corridor,
                        device=device,
                        skill_factor=skill_factor,
                    )

            elif state.activity == Activity.insert_wire:
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                if (
                    state.wire_looks_good
                    and abs(corridor.centerline.angle(device.principle_ray)) > THIRTY_DEGREES
                    and wire_placed[corridor_name]
                ):
                    # If the current view can actually assess the insertion of the wire
                    # Insert the wire and determine if it looks good for the next state.
                    state.wire_looks_inserted = self.evaluate_insertion(
                        state, wire, corridor, device
                    )
                    log.debug(f"Wire looks inserted: {state.wire_looks_inserted}")
                    advancement = self.sample_wire_advancement(state, wire, corridor, device)
                    wire.advance(advancement)
                    if (
                        np.random.rand() < 0.5
                        and (wire.tip_in_world - corridor.startpoint).norm() < 50
                    ):
                        # 50% chance of switching views when the wire is close to the startpoint.
                        state.need_new_view = True

                elif state.wire_looks_good and wire_placed[corridor_name]:
                    # The wire looks good, but we need to sample a new acquisition with a different view.
                    state.need_new_view = True
                    log.info("Need new view.")
                    continue

            elif state.activity == Activity.insert_screw:
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                state.wire_looks_inserted = self.evaluate_insertion(state, wire, corridor, device)
                state.screw_looks_inserted = self.evaluate_insertion(state, screw, corridor, device)
                if (
                    state.wire_looks_good
                    and abs(wire.centerline_in_world.angle(device.principle_ray)) > THIRTY_DEGREES
                    and not state.screw_looks_inserted
                    and screw_placed[corridor_name]
                ):
                    # Screw not yet inserted, and we can tell from this angle, so advance it.
                    advancement = self.sample_screw_advancement(state, screw, corridor)
                    screw.advance(advancement)
                elif state.wire_looks_good:
                    # The wire looks good, but we need to sample a new acquisition with a different view to do the insertion.
                    state.need_new_view = True
                    log.info("Need new view.")
                    continue
                else:
                    # The wire doesn't look good. Unknown what to do here. This is reached if the wire is in a bad position,
                    # but we have already moved onto screw insertion. The next state should go to wire positioning.
                    wire_placed[corridor_name] = False
                    screw_placed[corridor_name] = False
                    screw.place_center([99999, 99999, 99999])
                    # Makes it so the next state is wire positioning for the current task.
                    state.fix_wire = True
            else:
                raise ValueError(f"Unknown activity: {state.activity}")

        # Free the projectors
        projector.free()
        for seg_projector in seg_projectors.values():
            seg_projector.free()

        if running_frame_state is None:
            log.info("No frames were acquired.")
            return False

        # At the end, do some minor clean-up of the annotation.
        # First, finish off the last sequences.
        for supercategory in ["task", "activity", "acquisition", "frame"]:
            seq_name = getattr(running_frame_state, supercategory)
            if isinstance(seq_name, Task):
                seq_name = seq_name.get_wire()
            seq = {
                "id": len(annotation["sequences"]),
                "seq_length": sequence_counts[supercategory],
                "first_frame_id": frame_id - sequence_counts[supercategory],
                "seq_category_id": self.get_sequence_catid(supercategory, seq_name),
            }
            annotation["sequences"].append(seq)
            log.info(f"Saved sequence for {seq_name}:\n{seq}.")

        # Then, update the seq_length for all the images to the procedure length.
        procedure_length = len(annotation["images"])
        for i in range(procedure_length):
            annotation["images"][i]["seq_length"] = procedure_length

        save_json(annotation_path, annotation)
        return True

    def run(self):
        """While there are procedures to be done, run them.

        Reads a procedure_idx from the queue and samples to the procedure."""

        while True:
            try:
                procedure_idx = self.job_queue.get(block=False)
                if procedure_idx is None:
                    return

                success = self.sample_procedure(procedure_idx)

                if success:
                    self.finished_queue.put(procedure_idx)
                else:
                    self.job_queue.put(procedure_idx)
            except Empty:
                return
            except KeyboardInterrupt:
                return
            except Exception as e:
                log.exception(e)
                raise e

    def generate(self):
        """Generate the dataset by sampling procedures.

        This should only be run from the main process. Spawns the workers and specifies GPUs for
        them. Meanwhile, transfers images from completed procedures from the tmp directories to the
        final structure and modifies the annotation json accordingly. At the VERY end, creates a
        SINGLE json file from all the tmp annotations.

        Definitions:
            - Procedure: a full sequence video.
            - Sequence: any sequence of frames. May refer to the full video or a sub-sequence for a
              given phase.


        The dataset is written in an extended version of the COCO dataset format, with sequences.

        Since COCO writes annotations to a single JSON file, we have to make sure that dataset
        generation can be interrupted. To do this, we write the annotations to temporary files, and
        then move them to the final location when the dataset is fully generated.

        Specifically, at the end of every procedure sequence, we write root/tmp/XXXXXXXXX.json,
        which contains the images, annotations, and sequences data for the frames in that procedure.

        The "sequences" key is a custom extension to the COCO format. It contains a list of
        sequences, each with the first_frame_id, seq_length, and seq_type. The seq_type is one of

        Most segmentations will be "maps" encoded using COCO run-length encoding, with
        isCrowd=False. The pelvis segmentation can have associated keypoints, which are the
        landmarks (not implemented).

        """

        # Look for existing annotations in self.annotations_tmp_dir
        tmp_annotation_paths = sorted(list(self.tmp_dir.glob("*.json")))
        procedures_done = set(int(p.stem) for p in tmp_annotation_paths)

        if self.overwrite:
            if procedures_done:
                log.critical(
                    f"Found {len(procedures_done)} existing procedures. Press CTRL+C to abort!"
                )
                input("Press ENTER to overwrite...")

            procedures_done = set()
            shutil.rmtree(self.tmp_dir)
            shutil.rmtree(self.images_dir)
            self.tmp_dir.mkdir()
            self.images_dir.mkdir()
        elif len(procedures_done) > 0:
            log.info(f"Found {len(procedures_done)} existing procedures.")
            # for tmp_annotation_path in track(
            #     tmp_annotation_paths, description="Fixing annotations"
            # ):
            #     log.info(f"Fixing {tmp_annotation_path}")
            #     try:
            #         annotation = load_json(tmp_annotation_path)
            #     except json.JSONDecodeError:
            #         log.error(f"Failed to load {tmp_annotation_path}")
            #         continue

            #     # save_json(tmp_annotation_path.with_suffix(".bak"), annotation)
            #     fixed_annotation = self.fix_annotation(annotation)
            #     save_json(tmp_annotation_path, fixed_annotation)

        else:
            log.info("No existing procedures found. Starting from scratch.")

        if self.num_workers == 0:
            for procedure_idx in range(self.num_procedures):
                if procedure_idx not in procedures_done:
                    self.sample_procedure(procedure_idx)
                log.info(f"Finished procedure {procedure_idx} / {self.num_procedures}")
        else:
            job_queue = mp.Queue(self.num_procedures)
            finished_queue = mp.Queue(self.num_procedures)
            for procedure_idx in range(self.num_procedures):
                if procedure_idx not in procedures_done:
                    # Add jobs to the queue
                    log.info(f"Adding procedure {procedure_idx} to the queue.")
                    job_queue.put(procedure_idx)

            # Start workers
            workers: list[PelphixSim] = []
            for i in range(self.num_workers):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(i % torch.cuda.device_count())
                worker = PelphixSim(
                    job_queue=job_queue, finished_queue=finished_queue, **self.kwargs
                )
                worker.start()

            while len(procedures_done) < self.num_procedures:
                try:
                    procedure_idx = finished_queue.get(block=False)
                    procedures_done.add(procedure_idx)
                    log.info(
                        f"{self.name}: procedure {procedure_idx} finished. ({len(procedures_done)} / {self.num_procedures})."
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

            # Wait for all workers to finish
            for worker in workers:
                worker.terminate()

        log.info(f"Finished generating {self.num_procedures} procedures. Starting cleanup...")
        image_id = 0
        annotation_id = 0
        sequence_id = 0
        annotation = self.get_base_annotation()

        for procedure_annotation_path in track(
            list(self.tmp_dir.glob("*.json")), description="Merging annotations..."
        ):
            procedure_annotation = load_json(procedure_annotation_path)
            procedure_name = procedure_annotation_path.stem
            procedure_images_dir = self.tmp_dir / procedure_annotation_path.stem
            if not procedure_images_dir.exists():
                log.warning(
                    f"Images for procedure {procedure_annotation_path} not found. Skipping."
                )
                continue
            first_frame_id = image_id

            for image in procedure_annotation["images"]:
                image_path = procedure_images_dir / str(image["file_name"])
                new_image_path = self.images_dir / f"{procedure_name}_{image['file_name']}"
                if image_path.exists() and not new_image_path.exists():
                    shutil.copy(image_path, new_image_path)
                elif new_image_path.exists():
                    pass
                else:
                    log.error(f"Image {image_path} not found, new or old. Skipping")
                    continue

                image["file_name"] = new_image_path.name
                image["first_frame_id"] = first_frame_id
                image["id"] = image_id
                annotation["images"].append(image)
                image_id += 1

            for ann in procedure_annotation["annotations"]:
                ann["id"] = annotation_id
                ann["image_id"] = first_frame_id + ann["image_id"]
                annotation["annotations"].append(ann)
                annotation_id += 1

            for seq in procedure_annotation["sequences"]:
                seq["id"] = sequence_id
                seq["first_frame_id"] = first_frame_id + seq["first_frame_id"]
                annotation["sequences"].append(seq)
                sequence_id += 1

        log.info("Saving annotations...")
        save_json(self.annotation_path, annotation, indent=None, sort_keys=False)
        save_json(
            self.instance_annotation_path,
            self.remove_keypoints(annotation),
            indent=None,
            sort_keys=False,
        )
        save_json(
            self.keypoints_annotation_path,
            self.pelvis_only(annotation),
            indent=None,
            sort_keys=False,
        )

        # Zip the images
        log.info("Zipping images (may take a while)...")
        if not (self.root / f"{self.images_dir.stem}.zip").exists():
            shutil.make_archive(self.images_dir.stem, "zip", self.root, self.images_dir)

    @property
    def annotation_path(self) -> Path:
        """Get the path to the annotation file."""
        return self.annotations_dir / f"{self.name}.json"

    @property
    def instance_annotation_path(self) -> Path:
        """Get the path to the annotation file."""
        return self.annotations_dir / f"{self.name}_instances.json"

    @property
    def keypoints_annotation_path(self) -> Path:
        """Get the path to the annotation file."""
        return self.annotations_dir / f"{self.name}_keypoints.json"

    def get_annotation(self) -> dict[str, Any]:
        """Get the annotation dictionary."""
        return load_json(self.annotation_path)

    @classmethod
    def _merge_annotations(self, *annotations: dict[str, Any]) -> dict[str, Any]:
        """Merge the RLE annotations together.

        Resulting annotation will not have an ID, category id, or image_id. Also, keypoints will be
        concatenated together in the order they are passed in.

        Args:
            annotations: The annotations to merge.

        Returns:
            The merged annotation.

        """
        seg = mask_utils.merge([a["segmentation"] for a in annotations])
        seg["counts"] = seg["counts"].decode("utf-8")
        bbox = mask_utils.toBbox(seg).tolist()
        area = int(mask_utils.area(seg))
        if "keypoints" in annotations[0]:
            keypoints = np.concatenate([a["keypoints"] for a in annotations]).tolist()
        return {
            "segmentation": seg,
            "bbox": bbox,
            "area": area,
            "keypoints": keypoints,
            "num_keypoints": len(keypoints) // 3,
            "iscrowd": 0,
        }

    # TODO: add a function to convert annotation files, adding individual segmentation masks/boxes
    # for the pelvis keypoints. This will allow us to segment them with DICE scores. Be sure to only
    # take the intersection with the relevant pelvis mask.
    @classmethod
    def fix_annotation(cls, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Fix the annotation to take the hip_left, hip_right keypoints and transfer to the pelvis."""
        annotation = copy.deepcopy(annotation)
        annotation["categories"] = cls.categories

        # First, get the mapping from image_id to a dictionary of annotations by annotation name?
        image_id_to_annotations: dict[int, dict[str, list[dict]]] = {}
        image_id = None
        for anno in annotation["annotations"]:
            name = cls.get_annotation_name(anno["category_id"])
            if "keypoints" in anno and "num_keypoints" not in anno:
                anno["num_keypoints"] = len(anno["keypoints"]) // 3

            if name == "hip_left":
                image_id = anno["image_id"]

            if "image_id" not in anno and image_id is not None:
                log.debug(
                    f"Annotation for {name} does not have an image_id. Borrowing from most recent hip_left."
                )
                anno["image_id"] = image_id
            elif image_id is None:
                raise ValueError("First annotation does not have an image_id")

            if anno["image_id"] not in image_id_to_annotations:
                image_id_to_annotations[anno["image_id"]] = {}

            if name not in image_id_to_annotations[anno["image_id"]]:
                image_id_to_annotations[anno["image_id"]][name] = []

            image_id_to_annotations[anno["image_id"]][name].append(anno)

        log.info(f"Found {len(image_id_to_annotations)} images with annotations")
        # Now, go through and combine hips when available to make the pelvis
        for image_id in image_id_to_annotations:
            if "pelvis" in image_id_to_annotations[image_id]:
                continue

            if not (
                "hip_left" in image_id_to_annotations[image_id]
                and "hip_right" in image_id_to_annotations[image_id]
                and len(image_id_to_annotations[image_id]["hip_left"]) == 1
                and len(image_id_to_annotations[image_id]["hip_right"]) == 1
            ):
                continue

            # If the pelvis is not in the image, then we need to create it
            # First, get the hip_left and hip_right annotations
            hip_left = image_id_to_annotations[image_id]["hip_left"][0]
            hip_right = image_id_to_annotations[image_id]["hip_right"][0]
            pelvis = cls._merge_annotations(hip_right, hip_left)
            pelvis["category_id"] = cls.get_annotation_catid("pelvis")
            pelvis["image_id"] = image_id
            pelvis["track_id"] = 1000 * cls.get_annotation_catid("pelvis")
            image_id_to_annotations[image_id]["pelvis"] = [pelvis]

        # log.info(f"Re-indexing annotations")
        # Re-index all the annotations.
        new_annotations = []
        for image_id in image_id_to_annotations:
            for name in image_id_to_annotations[image_id]:
                for anno in image_id_to_annotations[image_id][name]:
                    anno["id"] = len(new_annotations)
                    new_annotations.append(anno)

        annotation["annotations"] = new_annotations
        return annotation

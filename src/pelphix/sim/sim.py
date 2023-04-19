"""Class for generating data for the NMDID workflows dataset.

Data is generated in an extended COCO format, with the following fields:

The first num_val CT images are used for generating validation data, and the remaining CT images
are used for generating training data.

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

from perphix.data import PerphixBase

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
DEGREE_SIGN = "\N{DEGREE SIGN}"


# CT, CT seg volumes, corridors, keypoints
Case = tuple[
    Volume, dict[str, Volume], dict[str, Mesh], dict[str, Cylinder], dict[str, geo.Point3D]
]
ImageSize = Union[int, Tuple[int, int]]


class PelphixSim(PerphixBase, Process):
    CASE_PATTERN = r"case-\d\d\d\d\d\d"

    # For wire adjustments
    MAX_ANGLE_BOUND = math.radians(30)
    MIN_ANGLE_BOUND = math.radians(3)
    MAX_TIP_BOUND = 10
    MIN_TIP_BOUND = 5

    # For view adjustments
    MAX_VIEW_ANGLE_BOUND = math.radians(45)
    MIN_VIEW_ANGLE_BOUND = math.radians(1)
    MAX_VIEW_POSITION_BOUND = 100
    MIN_VIEW_POSITION_BOUND = 5

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
        view_tolerance: float = dict[str, float],
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
            num_workers: Number of worker processes to use for generating data.
            max_procedure_length: Maximum number of steps in a procedure.

        """
        super().__init__()
        self.kwargs = locals()
        del self.kwargs["self"]
        del self.kwargs["__class__"]
        del self.kwargs["job_queue"]
        del self.kwargs["finished_queue"]

        self.job_queue = job_queue
        self.finished_queue = finished_queue
        self.root = Path(root).expanduser()
        self.nmdid_root = Path(nmdid_root).expanduser()
        self.pelvis_annotations_dir = Path(pelvis_annotations_dir).expanduser()
        self.num_val = num_val
        self.scan_name = scan_name
        self.train = train
        self.image_size = tuple(image_size)
        self.corridor_radii = corridor_radii
        self.overwrite = overwrite
        self.view_tolerance = deepdrr.utils.radians(dict(view_tolerance), degrees=True)
        self.num_workers = num_workers
        self.max_procedure_length = max_procedure_length

        if cache_dir is None:
            self.cache_dir = self.root / "cache"
        else:
            self.cache_dir = Path(cache_dir).expanduser()

        if not self.root.exists():
            self.root.mkdir(parents=True)

        case_names = sorted(
            [
                case.name
                for case in self.pelvis_annotations_dir.glob("case-*/")
                if re.match(self.CASE_PATTERN, case.name) is not None
            ]
        )
        if train:
            self.name = f"pelvic_workflows_{num_procedures:06d}_train"
            self.num_procedures = num_procedures * (len(case_names) - num_val) // len(case_names)
            self.case_names = case_names[num_val:]
        else:
            self.name = f"pelvic_workflows_{num_procedures:06d}_val"
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

    def load_case(self, case_name: str) -> Case:
        """loads the case

        Args:
            case_name (str): The case name, like "case-XXXXXX".

        Returns:
            Volume: The CT volume.
            dict[str, Volume]: The patient segmentation volumes.
            dict[str, Mesh]: The patient segmentation meshes.
            dict[str, Cylinder]: The corridors, in world coordinates.
            dict[str, geo.Point3D]: The pelvis keypoints in world coordinates.

        """
        ct_dir = self.nmdid_root / "nifti" / self.scan_name / case_name
        ct_paths = list(ct_dir.glob("*.nii.gz"))
        if not ct_paths:
            raise ValueError(f"Could not find CT for case {case_name}")

        cache_dir = self.cache_dir / "cropped_volumes" / case_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        ct_cache_dir = cache_dir / "ct"
        seg_cache_dirs = dict(
            hip_left=cache_dir / "hip_left",
            hip_right=cache_dir / "hip_right",
            femur_left=cache_dir / "femur_left",
            femur_right=cache_dir / "femur_right",
            sacrum=cache_dir / "sacrum",
            vertebrae_L5=cache_dir / "vertebrae_L5",
        )
        if ct_cache_dir.exists() and all(
            seg_cache_dir.exists() for seg_cache_dir in seg_cache_dirs.values()
        ):
            # Load the CT.
            ct = Volume.load(ct_cache_dir)
            seg_volumes = dict(
                hip_left=Volume.load(seg_cache_dirs["hip_left"], segmentation=True),
                hip_right=Volume.load(seg_cache_dirs["hip_right"], segmentation=True),
                femur_left=Volume.load(seg_cache_dirs["femur_left"], segmentation=True),
                femur_right=Volume.load(seg_cache_dirs["femur_right"], segmentation=True),
                sacrum=Volume.load(seg_cache_dirs["sacrum"], segmentation=True),
                vertebrae_L5=Volume.load(seg_cache_dirs["vertebrae_L5"], segmentation=True),
            )
        else:
            # Load the CT.
            ct_path = ct_paths[0]
            ct = Volume.from_nifti(ct_path)
            ct.orient_patient(head_first=True, supine=True)

            seg_dir = self.nmdid_root / "TotalSegmentator" / self.scan_name / case_name
            kwargs = dict(segmentation=True, world_from_anatomical=ct.world_from_anatomical)
            seg_volumes = dict(
                hip_left=Volume.from_nifti(seg_dir / "hip_left.nii.gz", **kwargs),
                hip_right=Volume.from_nifti(seg_dir / "hip_right.nii.gz", **kwargs),
                femur_left=Volume.from_nifti(seg_dir / "femur_left.nii.gz", **kwargs),
                femur_right=Volume.from_nifti(seg_dir / "femur_right.nii.gz", **kwargs),
                sacrum=Volume.from_nifti(seg_dir / "sacrum.nii.gz", **kwargs),
                vertebrae_L5=Volume.from_nifti(seg_dir / f"vertebrae_L5.nii.gz", **kwargs),
            )
            # Get the crop sizes for the CT and segmentations.
            bbox = np.zeros((3, 2))
            bbox[:, 0] = np.array(ct.shape)
            exclude = []
            for i, name in enumerate(track(seg_volumes, description="Cropping segmentations...")):
                seg_bbox = seg_volumes[name].get_bbox_IJK()
                if seg_bbox is None:
                    exclude.append(name)
                    continue
                if "femur" not in name:
                    bbox[:, 0] = np.minimum(bbox[:, 0], seg_bbox[:, 0])
                    bbox[:, 1] = np.maximum(bbox[:, 1], seg_bbox[:, 1])

                # Crop the segmentation.
                seg_volumes[name] = seg_volumes[name].crop(seg_bbox)

            # Exclude the segmentations that are empty.
            for name in exclude:
                del seg_volumes[name]

            # Crop the CT.
            bbox[:, 0] = np.maximum(bbox[:, 0] - np.array([100, 100, 200]), 0)
            bbox[:, 1] = np.minimum(bbox[:, 1] + np.array([100, 100, 200]), ct.shape)
            ct = ct.crop(bbox)

            # # Save the cropped volumes.
            # TODO: currently, the density is saved, not the houndsfield units. Could convert back, but mapping isn't really linear
            ct.save(ct_cache_dir)
            for name in track(seg_volumes, description="Caching segmentations..."):
                seg_volumes[name].save(seg_cache_dirs[name], segmentation=True)

        mesh_dir = self.nmdid_root / "TotalSegmentator_mesh" / self.scan_name / case_name
        seg_meshes: dict[str, Mesh] = {}
        for name in track(seg_volumes, description="Loading segmentations..."):
            seg_meshes[name] = Mesh.from_file(mesh_dir / f"{name}.stl")
            seg_meshes[name].transform(ct.world_from_anatomical @ geo.RAS_from_LPS)
            # seg_meshes[name].transform(ct.world_from_anatomical)

        corridors: dict[str, Cylinder] = dict()
        pelvis_keypoints: dict[str, geo.Point3D] = dict()
        case_annotations_dir = self.pelvis_annotations_dir / case_name
        flip_ramus = np.random.uniform() < 0.5
        flip_sacrum = np.random.uniform() < 0.5
        for path in case_annotations_dir.glob("*.fcsv"):
            if path.stem == "landmarks":
                pelvis_keypoints_LPS, landmark_names = data_utils.load_fcsv(path)
                pelvis_keypoints_world_arr = (
                    ct.world_from_anatomical @ geo.RAS_from_LPS
                ).transform_points(pelvis_keypoints_LPS)
                pelvis_keypoints_world = [geo.point(p) for p in pelvis_keypoints_world_arr]
                _pelvis_keypoints = dict(zip(landmark_names, pelvis_keypoints_world))
                pelvis_keypoints.update(_pelvis_keypoints)
            else:
                name = path.stem
                cylinder = Cylinder.from_fcsv(path, radius=self.corridor_radii[name])
                # corridors[name] = cylinder.transform(ct.world_from_anatomical)
                corridors[name] = cylinder.transform(ct.world_from_anatomical @ geo.RAS_from_LPS)

                # Flip the ramus and sacrum randomly.
                if (name in ["ramus_left", "ramus_right"] and flip_ramus) or (
                    name == "s2" and flip_sacrum
                ):
                    corridors[name] = corridors[name].flip()

                corridors[name] = corridors[name].shorten(self.corridor_fraction[name])

        return ct, seg_volumes, seg_meshes, corridors, pelvis_keypoints

    def get_APP(self, pelvis_keypoints: dict[str, geo.Point3D]) -> geo.FrameTransform:
        """Get the anterior pelvic plane coordinate (APP) frame.

        See https://www.nature.com/articles/s41598-019-49573-4 for details.

        Args:
            pelvis_keypoints: The pelvis keypoints in anatomical coordinates.
        """

        r_sps = pelvis_keypoints["r_sps"]
        l_sps = pelvis_keypoints["l_sps"]
        r_asis = pelvis_keypoints["r_asis"]
        l_asis = pelvis_keypoints["l_asis"]

        sps_midpoint = l_sps.lerp(r_sps, 0.5)
        asis_midpoint = l_asis.lerp(r_asis, 0.5)

        z_axis = (asis_midpoint - sps_midpoint).hat()
        x_axis_approx = (r_asis - l_asis).hat()

        y_axis = z_axis.cross(x_axis_approx).hat()
        x_axis = y_axis.cross(z_axis).hat()

        rot = np.stack([x_axis, y_axis, z_axis], axis=0)
        return geo.F.from_rt(rot.T, sps_midpoint)

    inlet_angle = math.radians(45)  # about X
    outlet_angle = math.radians(-45)  # about X
    oblique_left_angle = math.radians(45)  # about Z
    oblique_right_angle = math.radians(-45)  # about Z

    canonical_views_in_APP = {
        Acquisition.ap: geo.v(0, 1, 0),
        Acquisition.lateral: geo.v(-1, 0, 0),
        Acquisition.inlet: geo.v(0, math.cos(inlet_angle), math.sin(inlet_angle)),
        Acquisition.outlet: geo.v(0, math.cos(outlet_angle), math.sin(outlet_angle)),
        Acquisition.oblique_left: geo.v(
            math.sin(oblique_left_angle), math.cos(oblique_left_angle), 0
        ),
        Acquisition.oblique_right: geo.v(
            math.sin(oblique_right_angle), math.cos(oblique_right_angle), 0
        ),
    }

    def get_view_direction(
        self, view: Acquisition, world_from_APP: geo.FrameTransform, corridors: dict[str, Cylinder]
    ) -> geo.Vector3D:
        """Get a viewing direction in world coordinates.

        Args:
            view: The view to get.
            APP: The APP frame.
        """
        if view == Acquisition.ap:
            return world_from_APP @ geo.v(0, 1, 0)
        elif view == Acquisition.lateral:
            return world_from_APP @ geo.v(-1, 0, 0)
        elif view == Acquisition.inlet:
            return world_from_APP @ geo.v(0, math.sin(self.inlet_angle), math.cos(self.inlet_angle))
        elif view == Acquisition.outlet:
            return world_from_APP @ geo.v(
                0, math.sin(self.outlet_angle), math.cos(self.outlet_angle)
            )
        elif view == Acquisition.oblique_left:
            return world_from_APP @ geo.v(
                math.cos(self.oblique_left_angle),
                math.sin(self.oblique_left_angle),
                0,
            )
        elif view == Acquisition.oblique_right:
            return world_from_APP @ geo.v(
                math.cos(self.oblique_right_angle), math.sin(self.oblique_right_angle), 0
            )
        elif view in corridors:
            return corridors[view].get_direction()
        else:
            raise ValueError(f"Unknown view: '{view}'")

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

        dist = (current_view.p - desired_view.p).norm()
        bound = np.clip(
            skill_factor * dist, self.MIN_VIEW_POSITION_BOUND, self.MAX_VIEW_POSITION_BOUND
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
        angular_skill_factor = 1 - (1 - skill_factor) * 0.5
        angle = current_view.angle(desired_view)
        angle_bound = np.clip(
            angular_skill_factor * angle, self.MIN_VIEW_ANGLE_BOUND, self.MAX_VIEW_ANGLE_BOUND
        )
        n = geo.random.spherical_uniform(center=desired_view.n, d_phi=angle_bound)
        return geo.ray(p, n)

    def sample_device(self) -> SimpleDevice:
        """Sample a device with random (but reasonable) parameters.

        Returns:
            A device.

        """
        sensor_height = self.image_size[1]
        sensor_width = self.image_size[0]
        detector_width = np.random.uniform(300, 400)
        pixel_size = detector_width / sensor_width
        source_to_detector_distance = np.random.uniform(900, 1200)
        return SimpleDevice(
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            pixel_size=pixel_size,
            source_to_detector_distance=source_to_detector_distance,
        )

    # Fraction of the annotatation that is used for the corridor.
    corridor_fraction = dict(
        s1_left=0.7,
        s1_right=0.7,
        s1=0.7,
        s2=0.7,
        ramus_left=0.98,
        ramus_right=0.98,
        teardrop_left=0.98,
        teardrop_right=0.98,
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

    corridor_radii = dict(
        s1_left=5,
        s1_right=5,
        s1=5,
        s2=5,
        ramus_left=5,
        ramus_right=5,
        teardrop_left=5,
        teardrop_right=5,
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
                return True
            else:
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
        self, tool: Tool, corridor: Cylinder, device: SimpleDevice, is_wire: bool = True
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

        if is_wire:
            tip_index = index_from_world @ tool.tip_in_world
            progress = (tip_index - p_index).dot(q_index - p_index) / (q_index - p_index).normsqr()
        else:
            # For the screw, want to stop actually when the base is at the startpoint.
            base_index = index_from_world @ tool.base_in_world
            progress = (base_index - p_index).dot(q_index - p_index) / (
                q_index - p_index
            ).normsqr() + 1

        log.debug(f"Apparent progress: {progress:.3f}")
        if progress >= 0.95:
            # Wire/screw is fully inserted; continue.
            return True

        return False

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
        corridor_name = state.task
        tip_distance = (wire.tip_in_world - corridor.startpoint).norm()
        tip_bound = np.clip(skill_factor * tip_distance, self.MIN_TIP_BOUND, self.MAX_TIP_BOUND)
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
                self.MIN_ANGLE_BOUND,
                self.MAX_ANGLE_BOUND,
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
                skill_factor * wire_to_corridor_angle, self.MIN_ANGLE_BOUND, self.MAX_ANGLE_BOUND
            )
            new_wire_to_corridor_angle = np.random.uniform(-angle_bound, angle_bound)

            sign = wire_ray_in_index.n.cross(corridor_ray_in_index.n).z
            sign = sign / abs(sign)

            # log.warning(f"Not rotating by second random amount, for debugging")
            # rotvec = sign * wire_to_corridor_angle * principle_ray.hat()
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

        index_from_world = device.index_from_world
        tool_in_index = index_from_world @ tool.centerline_in_world
        corridor_projection = corridor.project(index_from_world)
        p_index = corridor_projection.start.line().meet(tool_in_index)
        q_index = corridor_projection.end.line().meet(tool_in_index)
        tip_index = index_from_world @ tool.tip_in_world

        progress = (tip_index - p_index).dot(q_index - p_index) / (q_index - p_index).normsqr()
        if progress >= 0.95:
            # Wire is fully inserted; continue.
            return 0

        # For wires, sample less.
        new_progress = np.clip(np.random.uniform(progress + 0.1, progress + 0.3), 0, 1)
        advancement = (new_progress - progress) * corridor.length()

        return advancement

    def sample_screw_advancement(
        self,
        screw: Tool,
        corridor: Cylinder,
    ) -> float:
        """Sample screw advancement."""
        startpoint = corridor.startplane().meet(screw.centerline_in_world)
        trajectory = screw.tip_in_world - screw.base_in_world
        advancement = -(screw.base_in_world - startpoint).dot(trajectory) / trajectory.norm()
        if np.random.uniform() < 0.5:
            advancement = np.random.uniform(0, advancement)
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

        # sample the skill factor
        # log.warning(f"TODO: remove this hard-coded false positive rate.")
        skill_factor = np.random.uniform(0.6, 0.8)
        false_positive_rate = np.random.uniform(0.05, 0.2)

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

        max_corridors = np.random.randint(len(corridors) + 1)

        # Wires and screws accessed by the corresponding corridor name.
        wires: dict[str, Tool] = dict()
        screws: dict[str, Screw] = dict()
        wire_track_ids: dict[str, int] = dict()
        screw_track_ids: dict[str, int] = dict()
        log.info(f"Sampling {max_corridors} corridors.")
        for i, name in enumerate(corridors):
            corr = corridors[name]
            wire = deepdrr.vol.KWire.from_example()
            if np.random.uniform() < 0.8:
                screw_type = get_screw(corr.length())
            else:
                screw_choices = list(
                    get_screw_choices(corr.length()) - get_screw_choices(corr.length() - 40)
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

        intensity_upper_bound = np.random.uniform(2, 10)
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
                # Start of a new task.
                if state.task.is_wire():
                    # Start of a wire task. Sample a wire position.
                    wire.orient(
                        geo.random.normal(
                            corridor.startpoint
                            - corridor.get_direction().hat() * np.random.uniform(0, 0.01),
                            scale=2,
                            radius=5,
                        ),
                        geo.random.spherical_uniform(
                            corridor.get_direction(), d_phi=math.radians(15)
                        ),
                    )
                elif state.task.is_screw():
                    # Start of a screw insertion.
                    # Goes over the corresponding wire.
                    screw.orient(
                        wire.centerline_in_world.meet(corridor.startplane()),
                        wire.tip_in_world - wire.base_in_world,
                        distance=0,
                    )
                continue

            if state.acquisition == Acquisition.start:
                # Start of the activity.
                # TODO: I think here just evaluate whether the wire/screw looks good.

                # Start of a new activity. Reposition the wire.
                # NOTE: later, we'll have to label this with the view annotation (e.g. "AP", "PA", etc.)
                # from the previous acquisition.
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                if state.activity == Activity.position_wire:
                    pass
                elif state.activity == Activity.insert_wire:
                    state.wire_looks_inserted = self.evaluate_insertion(
                        wire, corridor, device, is_wire=True
                    )
                elif state.activity == Activity.insert_screw:
                    state.screw_looks_inserted = self.evaluate_insertion(
                        screw, corridor, device, is_wire=False
                    )
                else:
                    raise ValueError(f"Unknown activity {state.activity}")
                continue

            # log.debug(
            # f'TODO: get desired viewpoint properly (only if the viewpoint has changed, and verify these "ideal" views with greg.'
            # )
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

            # Taking an acquisition. Need to get the desired view for the current acquisition.
            image = projector()
            index_from_world = device.index_from_world

            # Save the image.
            image_path = images_dir / f"{frame_id:09d}_{'-'.join(state.values())}.png"
            image_utils.save(image_path, image)

            # # Plot the wire and screw.
            # plotter = pv.Plotter(off_screen=True, window_size=(1536, 2048))
            # plotter.set_background("white")
            # plotter.add_mesh(device.get_mesh_in_world(), color="black", opacity=1)
            # for volume in seg_volumes.values():
            #     plotter.add_mesh(volume.get_mesh_in_world(full=False), color="black", opacity=1)
            # x_axis = ct.anatomical_from_world @ geo.v(1, 0, 0)
            # y_axis = ct.anatomical_from_world @ geo.v(0, 1, 0)
            # z_axis = ct.anatomical_from_world @ geo.v(0, 0, 1)
            # plotter.add_mesh(pv.Line(desired_view.p, desired_view.p + 100 * x_axis), color="red")
            # plotter.add_mesh(pv.Line(desired_view.p, desired_view.p + 100 * y_axis), color="green")
            # plotter.add_mesh(pv.Line(desired_view.p, desired_view.p + 100 * z_axis), color="blue")
            # plotter.add_mesh(
            #     pv.Line(desired_view.p, desired_view.p + 100 * desired_view.n), color="yellow"
            # )
            # plotter.set_position(desired_view.p + geo.v(0, 200, -300))
            # plotter.set_focus(desired_view.p)
            # plotter.set_viewup(list(ct.world_from_anatomical @ geo.v(0, 1, 0)))
            # # screenshot = plotter.show(screenshot=True)
            # screenshot = plotter.screenshot(return_img=True)
            # if screenshot is not None:
            #     image_utils.save(
            #         images_dir / f"{image_path.stem}_screenshot.png",
            #         screenshot,
            #     )
            # else:
            #     log.debug("No screenshot")
            # plotter.close()

            annotation["images"].append(
                {
                    "license": 0,
                    "file_name": image_path.name,
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "date_captured": datetime.datetime.now().isoformat(),
                    "id": frame_id,  # to be changed
                    "frame_id": frame_id,
                    "seq_length": -1,  # to be changed
                    "first_frame_id": 0,  # to be changed
                    "case_name": case_name,
                }
            )

            hip_left_ann: Optional[dict[str, Any]] = None
            hip_right_ann: Optional[dict[str, Any]] = None
            for track_id, seg_projector in seg_projectors.items():
                if not seg_projector.initialized:
                    continue

                seg_image = seg_projector()
                binary_mask = np.array(seg_image > 0, dtype=np.uint8)

                rle_mask = mask_utils.encode(np.asfortranarray(binary_mask))
                rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
                area = mask_utils.area(rle_mask)
                if area == 0:
                    continue
                name = seg_names[track_id]
                cat_id = self.get_annotation_catid(name)
                bbox = mask_utils.toBbox(rle_mask).tolist()
                ann = {
                    "segmentation": rle_mask,
                    "area": int(area),
                    "iscrowd": 0,
                    "image_id": frame_id,  # to be changed
                    "bbox": bbox,
                    "category_id": cat_id,
                    "id": len(annotation["annotations"]),
                    "track_id": track_id,
                }

                # Add the keypoints, if on hip_left, hip_right.
                if seg_names[track_id] in {"hip_left", "hip_right"}:
                    # Add the relevant keypoints
                    keypoints = []
                    keypoint_names = self.get_annotation_category(seg_names[track_id])["keypoints"]
                    for i, name in enumerate(keypoint_names):
                        landmark = pelvis_landmarks[name]
                        land = index_from_world @ landmark
                        if (
                            land.x >= 0
                            and land.x < image.shape[1]
                            and land.y >= 0
                            and land.y < image.shape[0]
                        ):
                            visibility = 2
                        else:
                            visibility = 1
                        keypoints.extend([int(land.x), int(land.y), visibility])
                    ann["keypoints"] = keypoints
                    ann["num_keypoints"] = len(keypoint_names)

                if seg_names[track_id] == "hip_left":
                    hip_left_ann = ann
                elif seg_names[track_id] == "hip_right":
                    hip_right_ann = ann

                annotation["annotations"].append(ann)

            # Add the pelvis landmarks/keypoints
            if hip_left_ann is not None and hip_right_ann is not None:
                pelvis_ann = self._merge_annotations(hip_right_ann, hip_left_ann)
                pelvis_ann["category_id"] = self.get_annotation_catid("pelvis")
                pelvis_ann["id"] = len(annotation["annotations"])
                pelvis_ann["track_id"] = 1000 * pelvis_ann["category_id"]
                pelvis_ann["image_id"] = frame_id
                annotation["annotations"].append(pelvis_ann)

            for corr_name, corr in corridors.items():
                corr_projection = corr.project(index_from_world)
                segmentation, area = corr_projection.get_segmentation(self.image_size)
                if not segmentation or area == 0:
                    continue
                cat_id = self.get_annotation_catid(corr_name)
                ann = {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": frame_id,  # to be changed
                    "bbox": coco_utils.segmentation_to_bbox(segmentation),
                    "category_id": cat_id,
                    "id": len(annotation["annotations"]),
                    "track_id": 1000 * cat_id,
                }
                annotation["annotations"].append(ann)

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

            frame_id += 1
            log.info(f"Acquired image {frame_id}.")

            # Set things up for the next transition.
            # Evaluations "after" the image acquisition, to determine if the view looks good and, if so, if the object looks good.
            if state.frame == Frame.fluoro_hunting:
                # The view didn't look good, so sample a new one, and determine whether it looks good.
                current_view = self.sample_next_view(
                    device,
                    current_view,
                    desired_view,
                    skill_factor=skill_factor,
                )
                device.set_view(
                    *current_view,
                    up=ct.world_from_anatomical @ geo.random.spherical_uniform(d_phi=FIVE_DEGREES),
                    source_to_point_fraction=np.random.uniform(0.65, 0.75),
                )
                state.view_looks_good = self.evaluate_view(state, device, desired_view)
                continue

            assert state.frame == Frame.assessment, f"unexpected frame: {state.frame}"

            # Assess the wire position.
            if state.activity == Activity.position_wire:
                # Position the wire and evaluate the position to determine if it looks good for the next state.
                log.debug("Positioning wire.")
                self.position_wire(
                    wire=wire,
                    state=state,
                    seg_meshes=seg_meshes,
                    corridor=corridor,
                    device=device,
                    skill_factor=skill_factor,
                )
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )

            elif state.activity == Activity.insert_wire:
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                if (
                    state.wire_looks_good
                    and abs(corridor.centerline.angle(device.principle_ray)) > THIRTY_DEGREES
                ):
                    # If the current view can actually assess the insertion of the wire
                    # Insert the wire and determine if it looks good for the next state.
                    advancement = self.sample_wire_advancement(wire, corridor, device)
                    wire.advance(advancement)
                    state.wire_looks_inserted = self.evaluate_insertion(
                        wire, corridor, device, is_wire=True
                    )
                    log.debug(f"Wire looks inserted: {state.wire_looks_inserted}")
                elif state.wire_looks_good:
                    # The wire looks good, but we need to sample a new acquisition with a different view.
                    state.need_new_view = True
                    log.info("Need new view.")
                    continue
            elif state.activity == Activity.insert_screw:
                state.wire_looks_good = self.evaluate_wire_position(
                    wire, corridor, device, false_positive_rate=false_positive_rate
                )
                if (
                    state.wire_looks_good
                    and abs(wire.centerline_in_world.angle(device.principle_ray)) > THIRTY_DEGREES
                ):
                    advancement = self.sample_screw_advancement(screw, corridor)
                    screw.advance(advancement)
                    state.screw_looks_inserted = self.evaluate_insertion(
                        screw, corridor, device, is_wire=False
                    )
                elif state.wire_looks_good:
                    # The wire looks good, but we need to sample a new acquisition with a different view to do the insertion.
                    state.need_new_view = True
                    log.info("Need new view.")
                    continue
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

        At the end of generation, these are joined into a single dataset json with the structure:
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
                    "file_name": "The name of the image file.",
                    "coco_url": "The URL of the image.",
                    "height": "The height of the image.",
                    "width": "The width of the image.",
                    "date_captured": "The date the image was captured.",
                    "flickr_url": "The URL of the image.",
                    "id": "The ID of the image.",
                    "frame_id": "The ID of the frame within the sequence.",
                    "seq_length": "The length of the sequence.",
                    "first_frame_id": "The ID of the first image in the sequence.",
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
                    "image_id": "The ID of the image.",
                    "bbox": "The bounding box of the object.",
                    "category_id": "The ID of the category.",
                    "id": "The ID of the annotation.",
                    "track_id": "A unique ID for this instance of the object.",
                },
                ...
            ],
            "sequences": [
                {
                    "first_frame_id": "The ID of the first frame in the sequence.",
                    "seq_length": "The length of the sequence.",
                    "seq_category_id": "The id of the sequence category.",
                }
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
                    "id": "The ID of the sequence type.",
                    "name": "The name of the sequence type.",
                }
        }

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
            # For debugging
            for procedure_idx in range(self.num_procedures):
                if procedure_idx not in procedures_done:
                    pass
                    # self.sample_procedure(procedure_idx)
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

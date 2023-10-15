from typing import List, Optional, Tuple, Type, Union, Dict, Any
from pathlib import Path
import datetime

from pycocotools import mask as mask_utils
from rich.progress import track
from gpustat import print_gpustat
from collections import Counter
import deepdrr
from deepdrr import geo, Volume
from deepdrr.device import SimpleDevice
from deepdrr.utils import data_utils, image_utils
from deepdrr import Projector
import numpy as np
import math
import time
from multiprocessing import Process
import logging
import pyvista as pv

from perphix.data import PerphixBase
from perphix.utils import vis_utils

from .state import Task, Activity, Acquisition, Frame, SimState, FrameState
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


class PelphixBase(PerphixBase, Process):
    """Base class for generating Pelphix simulated datasets."""

    CASE_PATTERN = r"case-\d\d\d\d\d\d"

    def __init__(
        self,
        root: Union[str, Path],
        nmdid_root: Union[str, Path],
        pelvis_annotations_dir: Union[str, Path],
        num_val: int = 10,
        scan_name: str = "THIN_BONE_TORSO",
        image_size: tuple[int, int] = (256, 256),
        overwrite: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        bag_world: bool = False,
    ):
        """Initialize the PelphixBase class."""
        super().__init__()
        self.root = Path(root).expanduser()
        self.nmdid_root = Path(nmdid_root).expanduser()
        self.pelvis_annotations_dir = Path(pelvis_annotations_dir).expanduser()
        self.num_val = num_val
        self.scan_name = scan_name
        self.image_size = tuple(image_size)
        self.overwrite = overwrite
        self.bag_world = bag_world

        if cache_dir is None:
            self.cache_dir = self.root / "cache"
        else:
            self.cache_dir = Path(cache_dir).expanduser()

        if not self.root.exists():
            self.root.mkdir(parents=True)

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

    def load_case(self, case_name: str) -> Case:
        """loads the case

        Args:
            case_name (str): The case name, like "case-XXXXXX".

        Returns:
            Volume: The CT volume.
            dict[str, Volume]: The patient segmentation volumes.
            dict[str, Mesh]: The patient segmentation meshes, transformed to world coordinates.
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
            mesh = Mesh.from_file(mesh_dir / f"{name}.stl")
            seg_meshes[name] = mesh.transform(ct.world_from_anatomical @ geo.RAS_from_LPS)

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

        return ct, seg_volumes, seg_meshes, corridors, pelvis_keypoints

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
    outlet_angle = math.radians(-40)  # about X
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

    def project_image(
        self,
        annotation: dict[str, Any],
        *,
        projector: Projector,
        device: SimpleDevice,
        image_path: Path,
        seg_projectors: dict[int, Projector],
        seg_names: dict[int, str],
        corridors: dict[str, Cylinder],
        pelvis_landmarks: dict[str, geo.Point3D],
        image_id: int,
        case_name: str,
        standard_view_directions: dict[Acquisition, geo.Vector3D] = {},
        desired_view: Optional[geo.Ray3D] = None,
    ):
        """Simulate a single image and all its spatial annotations.

        Args:
            annotation: The COCO-style annotation dict to add to.
            projector: The projector for the main image.
            device: The C-arm device.
            image_path: The path to save the image to.
            seg_projectors: Mapping from track_id to the projector for that segmentation.
            seg_names: Mapping from track_id to the segmentation name.
            corridors: Mapping from corridor name to the Cylinder object for each corridor.
            pelvis_landmarks: Mapping from landmark name to the 3D landmark in anatomical coordinates.
            image_id: The image id.
            case_name: The case name for the CT.
            standard_view_directions: Mapping from view to the standard view direction in world coordinates.
                For each standard view provided, angle to that view from current view is included in the dataset.
        """
        image = projector()
        index_from_world = device.index_from_world

        if desired_view is not None:
            # Plot the wire and screw.
            ct = projector.volumes[0]
            # plotter = pv.Plotter(off_screen=True, window_size=(1536, 2048))
            plotter = pv.Plotter()
            plotter.set_background("white")
            plotter.add_mesh(device.get_mesh_in_world(), color="black", opacity=1)
            for seg_projector in seg_projectors.values():
                volume = seg_projector.volumes[0]
                plotter.add_mesh(volume.get_mesh_in_world(full=False), color="black", opacity=1)
            x_axis = ct.anatomical_from_world @ geo.v(1, 0, 0)
            y_axis = ct.anatomical_from_world @ geo.v(0, 1, 0)
            z_axis = ct.anatomical_from_world @ geo.v(0, 0, 1)
            plotter.add_mesh(pv.Line(desired_view.p, desired_view.p + 100 * x_axis), color="red")
            plotter.add_mesh(pv.Line(desired_view.p, desired_view.p + 100 * y_axis), color="green")
            plotter.add_mesh(pv.Line(desired_view.p, desired_view.p + 100 * z_axis), color="blue")
            plotter.add_mesh(
                pv.Line(desired_view.p, desired_view.p + 100 * desired_view.n), color="yellow"
            )
            plotter.set_position(desired_view.p + geo.v(0, 1000, -500))
            plotter.set_focus(list(desired_view.p))
            plotter.set_focus(desired_view.p)
            plotter.set_viewup(list(ct.world_from_anatomical @ geo.v(0, 1, 0)))
            screenshot = plotter.show(screenshot=True)

            # screenshot = plotter.screenshot(return_img=True)
            if screenshot is not None:
                image_utils.save(
                    image_path.with_suffix(".screenshot.png"),
                    screenshot,
                )
            else:
                log.debug("No screenshot")
            plotter.close()

        # Get the angles to the standard views.
        view_dir = device.principle_ray_in_world
        standard_view_angles: dict[str, float] = dict()
        for standard_view, std_view_dir in standard_view_directions.items():
            standard_view_angles[str(standard_view)] = math.degrees(
                min(view_dir.angle(std_view_dir), view_dir.angle(-std_view_dir))
            )

        annotation["images"].append(
            {
                "license": 0,
                "file_name": image_path.name,
                "height": image.shape[0],
                "width": image.shape[1],
                "date_captured": datetime.datetime.now().isoformat(),
                "id": image_id,  # to be changed
                "frame_id": image_id,
                "seq_length": -1,  # to be changed
                "first_frame_id": 0,  # to be changed
                "case_name": case_name,
                "standard_view_angles": standard_view_angles,
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
                "image_id": image_id,  # to be changed
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
            pelvis_ann["image_id"] = image_id
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
                "image_id": image_id,  # to be changed
                "bbox": coco_utils.segmentation_to_bbox(segmentation),
                "category_id": cat_id,
                "id": len(annotation["annotations"]),
                "track_id": 1000 * cat_id,
            }
            annotation["annotations"].append(ann)

        # For debugging
        # corr = corridors["ramus_left"]
        # startpoint = index_from_world @ corr.startpoint
        # endpoint = index_from_world @ corr.endpoint
        # image = vis_utils.draw_keypoints(image, np.array([startpoint, endpoint]), ["start", "end"])

        # Save the image
        image_utils.save(image_path, image)

        # TODO: save everything else: poses for all the volumes, etc.
        if self.bag_world:
            bag = dict(
                image_id=image_id,
                volumes=[v.get_config() for v in projector.volumes],
                index_from_world=device.index_from_world.get_config(),
                source_to_detector_distance=device.source_to_detector_distance,
                pixel_size=device.pixel_size,
            )
            bag_path = image_path.with_suffix(".json")
            data_utils.save_json(bag_path, bag)

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

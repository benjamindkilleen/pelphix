# !/usr/bin/env python3
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path

from typing import Optional, List, Any, Tuple
from hydra.utils import get_original_cwd
import numpy as np
from skimage.transform import resize
import pyvista as pv
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from rich.progress import track

# Nosort
from .utils.onedrive_utils import OneDrive
from .shapes.ssm import StatisticalShapeModel
from .shapes.mesh import Mesh
from deepdrr.utils import data_utils

log = logging.getLogger(__name__)


def ssm_build(cfg: DictConfig):
    onedrive = OneDrive(**cfg.onedrive)

    nmdid_dir = Path(cfg.nmdid_dir)
    n_points = cfg.n_points
    n_components = cfg.n_components

    # TODO: this is not working for some reason.
    mesh_dir = onedrive.download(nmdid_dir / "TotalSegmentator_mesh", skip=True)

    case_dirs = sorted(list(mesh_dir.glob("THIN_BONE_TORSO/case-*")))
    hip_cache_dir = Path(get_original_cwd()) / "cache" / f"hips_{int(n_points/1000):03d}k"
    images_dir = Path(get_original_cwd()) / "images"
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    data_dir = Path(get_original_cwd()) / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    # First, get the paths
    hip_paths: list[Path] = []
    case_names: list[str] = []
    cache_names: list[str] = []
    for i, case_dir in enumerate(track(case_dirs, description="Getting paths")):
        for side in ["left", "right"]:
            hip_path = case_dir / f"hip_{side}.stl"
            if not hip_path.exists():
                continue
            hip_paths.append(hip_path)
            cache_names.append(f"{case_dir.name}_{side}")
            case_names.append(case_dir.name)

    hip_ssm_name = f"hip_ssm_{2 * len(case_dirs):04d}-meshes_{int(n_points/1000):03d}k-pts_{n_components:02d}-comps"
    hip_ssm_path = data_dir / f"{hip_ssm_name}.npz"
    if hip_ssm_path.exists():
        hip_ssm = StatisticalShapeModel.from_npz(hip_ssm_path)
    else:
        hip_meshes = []
        for i, hip_path in enumerate(track(hip_paths, description="Loading meshes")):
            right_side = "right" in hip_path.name
            pv_mesh = pv.read(str(hip_path))
            pv_mesh.flip_normals()
            if not pv_mesh.is_all_triangles():
                pv_mesh.triangulate(inplace=True)

            if not right_side:
                pv_mesh.points[:, 0] *= -1

            hip_meshes.append(pv_mesh)

        hip_ssm = StatisticalShapeModel.from_meshes(
            hip_meshes,
            n_points=n_points,
            n_components=n_components,
            cache_dir=hip_cache_dir,
            cache_names=cache_names,
        )
        hip_ssm.to_npz(hip_ssm_path)

    hip_ssm.sample().save(hip_ssm_path.with_suffix(".stl"))
    # plot_components(ssm, images_dir / "ssm_components")

    # Load ssm annotations, if they exist
    ssm_annotation_dir = data_dir / f"annotations_{hip_ssm_name}"
    if not ssm_annotation_dir.exists():
        log.info(f"No annotations found at {ssm_annotation_dir}")
        exit()

    hip_ssm.read_annotations(ssm_annotation_dir)
    if not "R_landmarks" in hip_ssm.annotations:
        log.debug(f"Annotations: {hip_ssm.annotations.keys()}")
        log.info("No landmarks found in annotations")
        exit()

    annotations_dir = data_dir / "annotations"
    mesh: Optional[Mesh] = None
    deformed_model: Mesh
    if not annotations_dir.exists():
        annotations_dir.mkdir(parents=True)
    for i, hip_path in enumerate(track(hip_paths, description="Propagating landmarks")):
        log.info(f"Propagating landmarks for {hip_path} ({i+1}/{len(hip_paths)})")
        right_side = "right" in hip_path.name
        side = "right" if right_side else "left"
        case_annotations_dir = annotations_dir / case_names[i]
        case_annotations_dir.mkdir(parents=True, exist_ok=True)
        landmarks_path = case_annotations_dir / "{}_landmarks.fcsv".format(
            "R" if right_side else "L"
        )
        weights_path = case_annotations_dir / f"weights_{side}.npy"

        if landmarks_path.exists():
            log.info(f"Landmarks already exist at {landmarks_path}")
            continue

        pv_mesh = pv.read(str(hip_path))
        pv_mesh.decimate(1 - n_points / pv_mesh.n_points, inplace=True)
        mesh = Mesh.from_pv(pv_mesh)
        if not right_side:
            # Flip the left hip to be on the right
            mesh = mesh.fliplr()

        # Fit the points.
        points_target = np.array(mesh.vertices)
        # If weights already exist, load them
        if weights_path.exists():
            weights = np.load(weights_path)
            weights, deformed_model, _ = hip_ssm.fit(
                points_target, max_iterations=20, weights=weights, fix_weights=True
            )
        else:
            weights, deformed_model, _ = hip_ssm.fit(points_target, max_iterations=20)
        distances = deformed_model.project_annotations(mesh)
        if not right_side:
            mesh = mesh.fliplr()

        mesh.save_annotations(case_annotations_dir)
        np.save(weights_path, weights)
        mesh = None

    log.info("----------------- Building pelvis SSM -----------------")

    # Load the pelvis meshes
    pelvis_n_points = cfg.pelvis_n_points
    pelvis_ssm_name = f"pelvis_ssm_{len(case_dirs):04d}-meshes_{int(pelvis_n_points/1000):03d}k-pts_{n_components:02d}-comps"
    pelvis_ssm_path = data_dir / f"{pelvis_ssm_name}.npz"
    pelvis_chache_dir = (
        Path(get_original_cwd()) / "cache" / f"pelvis_{int(pelvis_n_points/1000):03d}k"
    )
    pelvis_case_dirs = sorted(list(mesh_dir.glob("THIN_BONE_TORSO/case-*")))

    # [n, m, 3] array of landmarks for n meshes, m landmarks. Load order from left to right.
    landmarks = []
    pelvis_case_names = []
    pelvis_mesh_paths = []
    for i, case_dir in enumerate(track(pelvis_case_dirs, description="Loading pelvis meshes")):
        hip_left_path = case_dir / "hip_left.stl"
        hip_right_path = case_dir / "hip_right.stl"
        sacrum_path = case_dir / "sacrum.stl"
        if not hip_left_path.exists() or not hip_right_path.exists() or not sacrum_path.exists():
            log.error(f"No pelvis mesh found at {case_dir} (skipping)")
            continue

        # Get the annotations
        annotation_dir = annotations_dir / case_names[i]
        left_landmarks_path = annotation_dir / "L_landmarks.fcsv"
        right_landmarks_path = annotation_dir / "R_landmarks.fcsv"
        if not left_landmarks_path.exists() or not right_landmarks_path.exists():
            log.info(f"No landmarks found at {annotation_dir} (skipping)")
            continue

        left_landmarks, _ = data_utils.load_fcsv(left_landmarks_path)
        right_landmarks, _ = data_utils.load_fcsv(right_landmarks_path)
        landmarks.append(np.concatenate((left_landmarks, right_landmarks), axis=0))
        pelvis_mesh_paths.append([hip_left_path, hip_right_path, sacrum_path])
        pelvis_case_names.append(case_dir.name)

    landmarks = np.array(landmarks)
    if pelvis_ssm_path.exists():
        pelvis_ssm = StatisticalShapeModel.from_npz(pelvis_ssm_path)
    else:
        pelvis_ssm = StatisticalShapeModel.from_meshes(
            pelvis_mesh_paths,
            n_points=pelvis_n_points,
            n_components=n_components,
            cache_dir=pelvis_chache_dir,
            cache_names=pelvis_case_names,
            landmarks=landmarks,
            bootstrap_iters=5,
        )
        pelvis_ssm.to_npz(pelvis_ssm_path)

    pelvis_ssm.mean_mesh.save(data_dir / f"{pelvis_ssm_name}_mean.stl")

    log.info("----------------- Propagating annotations -----------------")
    pelvis_ssm_annotation_dir = data_dir / f"annotations_{pelvis_ssm_name}"
    if not pelvis_ssm_annotation_dir.exists():
        log.info(f"No annotations found at {pelvis_ssm_annotation_dir}")
        exit()

    pelvis_ssm.read_annotations(pelvis_ssm_annotation_dir)
    log.debug(f"Loaded {len(pelvis_ssm.annotations)} annotations from {pelvis_ssm_annotation_dir}")

    # Propagate the annotations
    pelvis_annotations_dir: Path = data_dir / "pelvis_annotations"
    mesh: Optional[Mesh] = None
    deformed_model: Mesh
    if not pelvis_annotations_dir.exists():
        pelvis_annotations_dir.mkdir(parents=True)
    for i, pelvis_paths in enumerate(track(pelvis_mesh_paths, description="Propagating landmarks")):
        log.info(f"Propagating landmarks for {pelvis_paths} ({i+1}/{len(pelvis_mesh_paths)})")
        case_annotations_dir = pelvis_annotations_dir / pelvis_case_names[i]
        case_annotations_dir.mkdir(parents=True, exist_ok=True)

        # Check if all the annotations are already present. If so, continue.
        case_annotations = set(p.stem for p in case_annotations_dir.glob("*.fcsv"))
        if all(annotation_name in case_annotations for annotation_name in pelvis_ssm.annotations):
            log.debug(f"Annotations: {case_annotations}")
            continue

        pv_mesh = pv.PolyData()
        for p in pelvis_paths:
            pv_mesh += pv.read(str(p))
        # Not necessary to decimate, using already decimated meshes
        pv_mesh.decimate(1 - pelvis_n_points / pv_mesh.n_points, inplace=True)
        mesh = Mesh.from_pv(pv_mesh)
        points_target = np.array(mesh.vertices)

        # If weights already exist, load them
        weights_path = case_annotations_dir / f"weights.npy"
        if weights_path.exists():
            log.info(f"Loading weights from {weights_path}")
            weights = np.load(weights_path)
            freeze_weights = True
        else:
            weights = None
            freeze_weights = False

        try:
            weights, deformed_model, _ = pelvis_ssm.fit(
                points_target, max_iterations=20, weights=weights, freeze_weights=freeze_weights
            )
        except RuntimeError:
            log.error(f"Failed to fit {pelvis_paths}")
            continue

        distances = deformed_model.project_annotations(mesh)
        log.info(f"Mean/max distance: {np.mean(distances):.2f}/{np.max(distances):.2f} mm")
        mesh.save_annotations(case_annotations_dir)
        np.save(weights_path, weights)
        np.save(case_annotations_dir / f"distances.npy", distances)
        mesh = None

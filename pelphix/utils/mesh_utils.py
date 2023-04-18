import logging
import os
import shutil
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from deepdrr import geo
from deepdrr.utils import listify
from rich.progress import Progress
from rich.progress import track

log = logging.getLogger(__name__)


def voxelize(
    surface: pv.PolyData,
    density: float = 0.2,
    bounds: Optional[List[float]] = None,
) -> Tuple[np.ndarray, geo.FrameTransform]:
    """Voxelize the surface mesh with the given density.

    Args:
        surface (pv.PolyData): The surface.
        density (Union[float, Tuple[float, float, float]]): Either a single float or a
            list of floats giving the size of a voxel in x, y, z.
            (This is really a spacing, but it's misnamed in pyvista.)

    Returns:
        Tuple[np.ndarray, geo.FrameTransform]: The voxelized segmentation of the surface as np.uint8 and the associated world_from_ijk transform.
    """
    density = listify(density, 3)
    voxels = pv.voxelize(surface, density=density, check_surface=False)

    spacing = np.array(density)
    if bounds is None:
        bounds = surface.bounds

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    size = np.array([(x_max - x_min), (y_max - y_min), (z_max - z_min)])
    if np.any(size) < 0:
        raise ValueError(f"invalid bounds: {bounds}")
    x, y, z = np.ceil(size / spacing).astype(int) + 1
    origin = np.array([x_min, y_min, z_min])
    world_from_ijk = geo.FrameTransform.from_rt(np.diag(spacing), origin)
    ijk_from_world = world_from_ijk.inv

    data = np.zeros((x, y, z), dtype=np.uint8)
    for p in track(voxels.points, "Rasterizing..."):
        p = geo.point(p)
        ijk = ijk_from_world @ p
        i, j, k = np.array(ijk).astype(int)
        data[i, j, k] = 1

    return data, world_from_ijk


def voxelize_file(path: str, output_path: str, **kwargs):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface = pv.read(path)
    try:
        data, world_from_ijk = voxelize(surface, **kwargs)
    except ValueError:
        log.warning(f"skipped {path} due to size error")
        return

    img = nib.Nifti1Image(data, geo.get_data(geo.RAS_from_LPS @ world_from_ijk))
    nib.save(img, output_path)


def voxelize_dir(input_dir: str, output_dir: str, use_cached: bool = True, **kwargs):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    input_len = len(input_dir.parts)
    paths: List[Path] = list(input_dir.glob("*/*.stl"))
    output_path: Path
    with Progress() as progress:
        surfaces_voxelized = progress.add_task("Voxelizing surfaces", total=len(paths))
        for path in paths:
            log.info(f"voxelizing {path}")
            output_path = output_dir / os.path.join(*path.parts[input_len:])
            output_path = output_path.with_suffix(".nii.gz")
            if output_path.exists() and use_cached:
                progress.advance(surfaces_voxelized)
                continue

            voxelize_file(path, output_path, **kwargs)
            progress.advance(surfaces_voxelized)


def to_trimesh(mesh: Union[pv.PolyData, trimesh.Trimesh]) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    elif isinstance(mesh, pv.PolyData):
        return trimesh.Trimesh(mesh.points, mesh.faces.reshape(-1, 4)[:, :3])
    else:
        raise ValueError(f"invalid mesh type: {type(mesh)}")

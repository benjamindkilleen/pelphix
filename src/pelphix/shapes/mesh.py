from __future__ import annotations
from matplotlib import pyplot as plt
from typing import Type

import pyvista as pv
from typing import Optional, Tuple
import logging
import numpy as np
from deepdrr import geo
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from rich.progress import track
import time
from pathlib import Path
from hydra.utils import get_original_cwd
from deepdrr.utils import data_utils
from ..utils import geo_utils

log = logging.getLogger(__name__)


def fliplr_str(s: str) -> str:
    if "left" in s:
        s = s.replace("left", "right")
    elif "right" in s:
        s = s.replace("right", "left")
    elif "Left" in s:
        s = s.replace("Left", "Right")
    elif "Right" in s:
        s = s.replace("Right", "Left")
    elif "L_" in s:
        s = s.replace("L_", "R_")
    elif "R_" in s:
        s = s.replace("R_", "L_")
    elif "r_" in s:
        s = s.replace("r_", "l_")
    elif "l_" in s:
        s = s.replace("l_", "r_")
    else:
        log.warning(f"Unknown side: {s}")
    return s


class Mesh:
    """A mesh for facilitating fast closest point on the surface."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        annotations: dict[str, np.ndarray] = {},
        descriptions: dict[str, np.ndarray] = {},
    ) -> None:
        """Make a Mesh.

        Args:
            vertices (np.ndarray): [N, 3] vertices of the triangles
            faces (np.ndarray): [M, 3] indices of the vertices that make up the triangles
            annotations (dict[str, np.ndarray], optional): Any annotations on the mesh, mapping to [m,3] array.
                Defaults to {}.
            descriptions (dict[str, list[str]], optional): Descriptive labels on the annotations. Each entry
                should be an [m] array of strings. Defaults to {}.
        """
        self.vertices = vertices
        self.faces = faces

        # [M, 3, 3] vertices of the triangles, in the same order as faces
        self.triangle_vertices = np.array([self.vertices[[u, v, w]] for u, v, w in self.faces])

        # [M, 3] the centroid of each triangle
        self.centroids = np.mean(self.triangle_vertices, axis=1)

        # [M] the radius of the bounding sphere for each triangle
        self.bounding_spheres: np.ndarray = np.linalg.norm(
            self.triangle_vertices - self.centroids[:, None, :], axis=2
        ).max(axis=1)

        # Spatial data structure for finding closest triangle to a point.
        self.centroids_tree = cKDTree(self.centroids)
        # self.covtree = CovNode(self)

        # Any annotations on the mesh.
        self.annotations = annotations
        self.descriptions = descriptions

    def __str__(self):
        return f"Mesh with {len(self.vertices)} vertices, {len(self.faces)} faces, annotations: {list(self.annotations.keys())}."

    @classmethod
    def from_pv(cls: Type[Mesh], mesh: pv.PolyData):
        """Make a CovMesh from a pyvista PolyData object."""
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        vertices = np.array(mesh.points)
        faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]
        return cls(vertices, faces, annotations={}, descriptions={})

    def as_pv(self) -> pv.PolyData:
        """Convert the mesh to a pyvista PolyData object."""
        faces = np.concatenate([np.full((len(self.faces), 1), 3), self.faces], axis=1).reshape(-1)
        return pv.PolyData(self.vertices, faces)

    @classmethod
    def from_file(cls, filename: str):
        """Make a CovMesh from a file."""
        mesh = pv.read(filename)
        return Mesh.from_pv(mesh)

    def save(self, path: str):
        """Save a mesh to a file (not including annotations)."""
        mesh = self.as_pv()
        mesh.save(str(path))

    def to_npz(self, path: str):
        """Save the mesh to a NPZ file."""
        np.savez_compressed(
            path,
            vertices=self.vertices,
            faces=self.faces,
            annotations=self.annotations,
            descriptions=self.descriptions,
        )

    @classmethod
    def from_npz(cls, path: str):
        """Load a mesh from a NPZ file."""
        data = dict(np.load(path, allow_pickle=True))

        return Mesh(
            data["vertices"],
            data["faces"],
            annotations=data.get("annotations", {}),
            descriptions=data.get("descriptions", {}),
        )

    def save_annotations(self, annotation_dir: str):
        """Save the annotations as fcsv files to the path."""
        annotation_dir = Path(annotation_dir)
        if not annotation_dir.exists():
            annotation_dir.mkdir(parents=True)
        for name, data in self.annotations.items():
            data_utils.save_fcsv(annotation_dir / f"{name}.fcsv", data, self.descriptions[name])

    def transform(self, f: geo.FrameTransform) -> Mesh:
        """Transform the mesh by a FrameTransform.

        Args:
            f (FrameTransform): The transform to apply to the mesh.

        Returns:
            Mesh: The transformed mesh.
        """
        f = geo.frame_transform(f)
        vertices = f.transform_points(self.vertices)
        annotations = {k: f.transform_points(v) for k, v in self.annotations.items()}
        return Mesh(vertices, self.faces, annotations, self.descriptions)

    def fliplr(self) -> Mesh:
        """Flip the mesh left to right.

        Returns:
            Mesh: The flipped mesh.

        """
        f = geo.frame_transform(np.diag([-1, 1, 1, 1]))
        vertices = f.transform_points(self.vertices.copy())
        faces = np.copy(self.faces)
        annotations = {}
        descriptions = {}
        for k, v in self.annotations.items():
            name = fliplr_str(k)
            annotations[name] = f.transform_points(v)
            descriptions[name] = np.array([fliplr_str(d) for d in self.descriptions[k]])

        return Mesh(vertices, faces, annotations, descriptions)

    def project_annotations(
        self,
        mesh: Mesh,
    ) -> np.ndarray:
        """Project the annotations from one mesh to another.

        Assumes the two meshes are aligned.

        Args:
            mesh (Mesh): The mesh to project the annotations to.

        Returns:
            np.ndarray: The distances between the projected points and the original points.
                This can be used to determine if the projection was successful.

        """
        ds = []
        for name, points in self.annotations.items():
            projected_points, distances, _ = mesh.query(points)
            mesh.annotations[name] = projected_points
            mesh.descriptions[name] = self.descriptions[name]

            ds.append(distances)
        ds = np.concatenate(ds)

        return ds

    def query(self, ss: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the closest points to the points ss on the mesh.

        Args:
            ss (np.ndarray): [N, 3] points to query
            k (int, optional): Number of nearest neighbors to check. Defaults to 5.

        Returns:
            cs (np.ndarray): [N, 3] closest points on the mesh
            cdists (np.ndarray): [N] distances to the closest points
            cidxs (np.ndarray): [N] indices of the closest triangles

        """
        _, cidxs_candidates = self.centroids_tree.query(ss, k=k)
        cs = np.zeros_like(ss)
        cidxs = np.zeros(len(ss), dtype=int)
        cdists = np.zeros(len(ss), dtype=float)
        for i, tidxs in enumerate(cidxs_candidates):
            # cidx: [k] indices of the closest centroids
            cdists[i] = np.inf
            for j, tidx in enumerate(tidxs):
                if (
                    j > 0
                    and np.linalg.norm(self.centroids[tidx] - cs[i]) - self.bounding_spheres[tidx]
                    > cdists[i]
                ):
                    continue
                    # Check if we are in the bounding sphere of the triangle.
                # Check each of the k closest centroids to see if we can find a closer point.
                p, q, r = self.triangle_vertices[tidx]
                new_c = geo_utils.project_on_triangle_np(p, q, r, ss[i])
                dist = np.linalg.norm(new_c - ss[i])
                if dist < cdists[i]:
                    cdists[i] = dist
                    cs[i] = new_c
                    cidxs[i] = tidx

        return cs, cdists, cidxs

    def query_point(self, p: geo.Point3D, k: int = 5) -> Tuple[geo.Point3D, float, int]:
        """Get the closest point to p on the mesh.

        Args:
            p (geo.Point3D): The point to query.
            k (int, optional): Number of triangles to check. Defaults to 5.

        Returns:
            Tuple[geo.Point3D, float, int]: The point, distance, and triangle index.
        """
        cs, cdists, cidxs = self.query(np.array(p).reshape(1, 3), k=k)
        return geo.point(cs[0]), cdists[0], cidxs[0]

    def _from_(self, *args, **kwargs) -> geo.FrameTransform:
        return self.register(*args, **kwargs)

    def register(
        self,
        points: np.ndarray,
        initial: Optional[geo.FrameTransform] = None,
        max_iterations: int = 100,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> geo.FrameTransform:
        """Register points to the surface of the mesh.

        Goal is to find a transform `F` that minimizes
        || F @ p - (mesh.query(F @ p))||^2
        over p in points.

        Args:
            points (np.ndarray): [N, 3] points to register

        Returns:
            mesh_from_points (geo.FrameTransform): [4, 4] transform that registers points to the mesh.

        """
        if initial is None:
            # mesh_from_points = geo.F.from_rt(translation=np.mean(points, axis=0))
            mesh_from_points = geo.F.identity()
        else:
            mesh_from_points = geo.frame_transform(initial)

        prev_cost = np.inf
        best_cost = np.inf
        best_mesh_from_points = mesh_from_points
        for i in range(max_iterations):
            cost = 0
            ss = mesh_from_points.transform_points(points)

            # Find the closest centroids to the points. Then check each one to get closest point.
            cs, _, _ = self.query(ss, k=5)

            cost = np.linalg.norm(ss - cs, axis=1).mean()
            log.debug(f"ICP Cost: {cost:.3f}")
            mesh_from_points = geo.F.from_point_correspondence(cs, points)

            if cost < best_cost:
                best_cost = cost
                best_mesh_from_points = mesh_from_points

            if np.abs(cost - prev_cost) < atol + rtol * prev_cost or cost > prev_cost:
                log.debug(f"ICP converged after {i} iterations.")
                break

            prev_cost = cost

        return best_mesh_from_points

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
        if not mesh.is_all_triangles():
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


class CovNode(object):
    """Represents a node of the covariance tree, with reference back to global arrays.

    Not used, because it's too slow (Python).

    """

    def __init__(
        self,
        mesh: Mesh,
        which: Optional[np.ndarray] = None,
        level: int = 0,
        min_points_per_node: int = 10,
    ):
        """Define a covariance tree.

        Args:
            points_in_root (np.ndarray): (n, 3) All the points associated with the data structure (not just in this box.)
            which (np.ndarray, optional): (n,) Boolean array indicating membership in this box. If None, then this is assumed to be the root node. Defaults to None.
            level (int, optional): How deep in the tree is this node? Defaults to 0.

        """

        self.level = level
        self.mesh = mesh
        self.which = which if which is not None else np.ones(self.mesh.centroids.shape[0], bool)
        self.n = self.which.sum()
        centroids = self.mesh.centroids
        bounding_spheres = self.mesh.bounding_spheres

        if self.n <= min_points_per_node:
            self.left_node = None
            self.right_node = None
            self.node_from_root = geo.F.identity(3)
            lower_bounds_in_node = centroids[self.which] - bounding_spheres[self.which, None]
            upper_bounds_in_node = centroids[self.which] + bounding_spheres[self.which, None]
            self.upper_bound_in_node = np.max(lower_bounds_in_node, axis=0)
            self.lower_bound_in_node = np.min(upper_bounds_in_node, axis=0)
        else:
            self.node_from_root = CovNode._get_node_from_root(centroids[self.which])

            points_in_node = self.node_from_root.transform_points(centroids)
            lower_bounds_in_node = points_in_node[self.which] - bounding_spheres[self.which, None]
            upper_bounds_in_node = points_in_node[self.which] + bounding_spheres[self.which, None]
            self.lower_bound_in_node = np.min(lower_bounds_in_node, axis=0)
            self.upper_bound_in_node = np.max(upper_bounds_in_node, axis=0)
            midpoint = np.median(points_in_node[self.which, 0])
            which_left = np.logical_and(points_in_node[:, 0] <= midpoint, self.which)
            which_right = np.logical_and(points_in_node[:, 0] > midpoint, self.which)

            self.left_node = CovNode(
                self.mesh,
                which=which_left,
                level=level + 1,
            )
            self.right_node = CovNode(
                self.mesh,
                which=which_right,
                level=level + 1,
            )

    @property
    def has_children(self) -> bool:
        # log.debug(f"{self.left_node is None} == {self.right_node is None}")
        # assert (
        #     self.left_node is None == self.right_node is None
        # ), f"Both or neither child should be None, got {self.left_node} and {self.right_node}"
        return self.left_node is not None and self.right_node is not None

    def __str__(self):
        return f"Node(n={self.n}, level={self.level})"

    @staticmethod
    def _get_node_from_root(points_in_root: np.ndarray) -> geo.FrameTransform:
        """Computes the transformation to this node's coordinate system from the root points.

        Based on aligning the x axis with the principal component of variation among the points.

        Args:
            points_in_root: [N, 3] points in root coordinates

        """

        p = points_in_root.mean(axis=0)
        u, sig, vt = np.linalg.svd(points_in_root - p)
        new_x, _, _ = vt  # rows are new basis vectors
        x = np.array([1, 0, 0], np.float64)  # x axis
        theta = np.arccos((x @ new_x) / (np.linalg.norm(x) * np.linalg.norm(new_x)))
        rotation_axis = np.cross(x, new_x)
        r = rotation_axis / np.linalg.norm(rotation_axis) * theta
        R = Rotation.from_rotvec(r).as_matrix()
        root_from_node = geo.FrameTransform.from_rt(R, p)
        return root_from_node.inverse()

    def contains(self, x_root: np.ndarray, slack: float = 0):
        """Check whether point is in the node, expanded by `slack`.

        :param point: the point, in root.
        :param slack: border to add around this bounding node.
        :returns:
        :rtype:
        """
        x_node = self.node_from_root.transform_points(x_root)

        return np.all(x_node >= self.lower_bound_in_node - slack) and np.all(
            x_node <= self.upper_bound_in_node + slack
        )

    def __contains__(self, point):
        return self.contains(point)

    def _brute_query(
        self, x_root: np.ndarray, cdist: Optional[float]
    ) -> Tuple[np.ndarray, float, int]:
        """Linear search for closest point .

        Args:
            x_root: [3] point in root coordinates.

        Returns:
            c: [3] closest point in root coordinates.
            cdist: distance to closest point.
            cidx: index of triangle containing closest point.

        :param x: [3] point
        :return: [3] closest point on mesh
        """
        if cdist is None:
            cdist = np.inf
        cidx = None
        c = None
        for tidx in np.where(self.which)[0]:
            centroid = self.mesh.centroids[tidx]
            bounding_sphere = self.mesh.bounding_spheres[tidx]
            if np.square(x_root - centroid).sum() > (bounding_sphere + cdist) ** 2:
                continue

            p, q, r = self.mesh.triangle_vertices[tidx]
            c_ = np.array(geo_utils.project_on_triangle_np(p, q, r, x_root))
            dist = np.linalg.norm(c_ - x_root)
            if dist < cdist:
                cdist = dist
                cidx = tidx
                c = c_

        return c, cdist, cidx

    def _query(
        self, x: np.ndarray, c: Optional[np.ndarray], cdist: Optional[float], cidx: int
    ) -> Tuple[float, int]:
        """Recursive query function.

        Takes the index of the current closest point to `x`, c.
        - If the distance to the borders of the node is greater than the distance to c, then
            no point in the node could be closer than c, so return c.
        - If there are children, then search each one and return the closest of the two points
          returned by each child.
        - Else, exhaustively search the triangles in this box for the one with the closest point
          one, and return it.

        Args:
            x: [3] point in root coordinates.
            cdist: distance to current closest point `c`.
            cidx: index of current closest point `c`.
        """
        if c is None or cdist is None:
            p, q, r = self.mesh.triangle_vertices[cidx]
            c = geo_utils.project_on_triangle_np(p, q, r, x)
            cdist = np.linalg.norm(x - c)

        check_left = False
        check_right = False

        if not self.contains(x, slack=cdist):
            # closest point couldn't be in box; slack expands borders by cdist.
            return c, cdist, cidx
        elif self.has_children:
            # the point is in the box, and the box has children, so search them.

            # TODO: only check one of the nodes, the one closer to the child
            if self.left_node.contains(x, slack=cdist):
                left_c, left_cdist, left_cidx = self.left_node._query(x, c, cdist, cidx)
            else:
                left_c, left_cdist, left_cidx = c, cdist, cidx

            if self.right_node.contains(x, slack=cdist):
                right_c, right_cdist, right_cidx = self.right_node._query(x, c, cdist, cidx)
            else:
                right_c, right_cdist, right_cidx = c, cdist, cidx

            if left_cdist < right_cdist:
                new_c, new_cdist, new_cidx = left_c, left_cdist, left_cidx
            else:
                new_c, new_cdist, new_cidx = right_c, right_cdist, right_cidx
        else:
            # exhaustively search the points in this box
            new_c, new_cdist, new_cidx = self._brute_query(x, cdist)

        if new_cdist < cdist:
            return new_c, new_cdist, new_cidx
        else:
            return c, cdist, cidx

    def query(self, x: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Get the distance and index of the point closest to `x`.

        Args:
            x: [3] point in root coordinates.

        Returns:
            np.ndarray: [3] closest point on mesh
            float: distance to closest point
            int: index of triangle containing closest point

        """
        return self._query(x, None, None, 0)

from __future__ import annotations

from typing import Tuple, List, Optional, Union, Type
import logging
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
from rich.progress import track
from procrustes.generalized import generalized as generalized_procrustes
from pycpd import DeformableRegistration, RigidRegistration
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import trimesh
from trimesh.registration import icp
from deepdrr import geo
from pathlib import Path
import subprocess as sp
from hydra.utils import get_original_cwd
import time
from deepdrr.utils import data_utils
from ..utils import mesh_utils
from ..utils import geo_utils
from .mesh import Mesh

log = logging.getLogger(__name__)


class StatisticalShapeModel(object):
    """A statistical shape model.

    This consists of a [n, 3] array of points (the mean shape), a [k, n, 3] array of components (the
    modes, or principal components of variation), and an optional array of faces, in the pyvista
    format.

    The principle function of the statistical shape model is to yield a pyvista surface model given
    a set of eigenvalues, or weights, for each component. This is done with the `get_mesh` method.

    Pyvista `faces` are a 1D array where each entry is a variable length list of vertex indices. The
    first element of each list is the number of vertices in the face. For example, a single triangle
    would be represented as [3, i, j, k].

    Attributes:
        points (np.ndarray): [n, 3] array of points defining the mean shape.
        components (np.ndarray): [k, n, 3] array of components.
        faces (np.ndarray): [f] array of faces on the points, pyvista format.

    """

    def __init__(
        self,
        points: np.ndarray,
        components: np.ndarray,
        faces: np.ndarray,
        annotations: dict[str, np.ndarray] = {},
        descriptions: dict[str, np.ndarray] = {},
    ):
        """Initialize a StatisticalShapeModel.

        Args:
            points (np.ndarray): [n, 3] array of points defining the mean shape.
            components (np.ndarray): [k, n, 3] array of components.
            faces (np.ndarray): [f, 3] array of faces on the points.
            annotations: mapping from annotation names to [m, 3] arrays containing points on the mean mesh.
                (Will be projected onto it.) Any meshes sampled from the model will have these
                annotations as well, deformed.
            descriptions: mapping from label names to [m] arrays containing labels for each point on the mean mesh.
        """

        if faces.ndim == 2:
            log.debug("faces is 2D, assuming it is a list of faces.")
            faces = np.hstack([np.ones((faces.shape[0], 1), dtype=int) * 3, faces])

        mesh = pv.PolyData(points, faces)
        if not mesh.is_all_triangles():
            mesh = mesh.triangulate()

        # Gets rid of pad points
        points = np.array(mesh.points)
        faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]
        indices = np.isin(np.arange(points.shape[0]), faces, assume_unique=True, invert=False)

        self.points = points[indices]
        self.components = components[:, indices]
        self.faces = faces

        self.mean_mesh = self.sample()

        # mapping from annotation name to [m, 3] array of points, or descriptions
        self.annotations = {}
        self.descriptions = {}
        # mapping from annotation name to [m, 3] array of face indices
        self.annotation_faces = {}
        # mapping from annotation name to [m, 3, 3] array of triangle vertices
        self.annotation_face_vertices = {}
        # Mapping from annotation name to [m, 3] array of barycentric coordinates
        self.annotation_barycentric = {}

        for name, points in annotations.items():
            self.add_annotation(name, points, descriptions.get(name, None))

    def add_annotation(self, name: str, points: np.ndarray, description: np.ndarray[str] = None):
        """Add an annotation to the model.

        Args:
            name (str): The name of the annotation.
            points (np.ndarray): [m, 3] The points to add. If a 1D array, will be reshaped to [m, 3].
            description (np.ndarray): [m] array of labels for each point. If none, indices will be used.

        """
        if name in self.annotations:
            log.warning(f"Overwriting annotation {name}.")

        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        self.annotations[name], _, cidxs = self.mean_mesh.query(points)

        if description is None:
            description = np.array([str(i) for i in range(len(points))])
        self.descriptions[name] = np.array(description)

        self.annotation_faces[name] = self.faces[cidxs]
        self.annotation_face_vertices[name] = self.mean_mesh.triangle_vertices[cidxs]
        ps, qs, rs = (
            self.annotation_face_vertices[name][:, 0],
            self.annotation_face_vertices[name][:, 1],
            self.annotation_face_vertices[name][:, 2],
        )
        self.annotation_barycentric[name] = geo_utils.barycentric(ps, qs, rs, points)

    def sample_annotation(self, name: str, weights: np.ndarray) -> np.ndarray:
        """Sample an annotation from the model.

        Args:
            name (str): The name of the annotation.
            weights (np.ndarray): The weights for each component.

        Returns:
            np.ndarray: [m, 3] array of points.

        """
        if name not in self.annotations:
            raise ValueError(f"Unknown annotation {name}.")

        # [n_components, m, 3, 3] * [n_components, 1, 1, 1] -> [n_components, m, 3, 3] -> [m, 3, 3]
        sampled_components = self.components[:, self.annotation_faces[name]]
        deformations = sampled_components * weights[:, None, None, None]
        deformation = deformations.sum(0)

        # [m, 3, 3] + [m, 3]
        verts = self.annotation_face_vertices[name] + deformation
        ps, qs, rs = verts[:, 0], verts[:, 1], verts[:, 2]
        points = geo_utils.from_barycentric(ps, qs, rs, self.annotation_barycentric[name])
        return points

    def read_annotation(self, path: Path):
        """Load an annotation from an fcsv or mrk.json file."""
        path = Path(path)
        name = path.stem.split(".")[0]
        if path.suffixes == [".mrk", ".json"]:
            data = data_utils.load_json(path)
            points = np.array([entry["position"] for entry in data["markups"][0]["controlPoints"]])
            description = np.array(
                [entry["label"] for entry in data["markups"][0]["controlPoints"]]
            )

        elif path.suffix == ".fcsv":
            data, description = data_utils.load_fcsv(path)

        else:
            raise ValueError(f"Unknown file type {path.suffix}")

        self.add_annotation(name, points, description)

    def read_annotations(self, annotation_dir: Path):
        """Read all annotations in fcsv or mrk.json format from a directory."""
        for path in Path(annotation_dir).glob("*.mrk.json"):
            self.read_annotation(path)
        for path in Path(annotation_dir).glob("*.fcsv"):
            self.read_annotation(path)

    @property
    def n_points(self) -> int:
        """The number of points in the model."""
        return self.points.shape[0]

    @property
    def n_components(self) -> int:
        """The number of components in the model."""
        return self.components.shape[0]

    def sample_pv(self, *weights: Union[float, int, np.ndarray]) -> pv.PolyData:
        """Get a pyvista mesh from the shape model.

        Args:
            weights (np.ndarray): The weights for each component.

        Returns:
            pv.PolyData: The mesh.

        """
        if len(weights) == 1 and isinstance(weights[0], np.ndarray):
            ws = weights[0]
        else:
            ws = np.zeros(self.n_components)
            ws[: len(weights)] = weights

        points = self.points + np.sum(self.components * ws[:, None, None], axis=0)

        pv_faces = np.hstack(np.full((self.faces.shape[0], 1), 3), self.faces).reshape(-1)
        out = pv.PolyData(points, pv_faces)
        return out

    def sample(
        self, *weights: Union[float, int, np.ndarray], with_annotations: bool = False
    ) -> Mesh:
        """Get a covariance mesh from the shape model.

        Args:
            weights (np.ndarray): The weights for each component.
            with_annotations (bool): Whether to include annotations (slight slowdown)

        Returns:
            Mesh: The mesh.

        """
        if len(weights) == 1 and isinstance(weights[0], np.ndarray):
            ws = weights[0]
        else:
            ws = np.zeros(self.n_components)
            ws[: len(weights)] = weights

        points = self.points + np.sum(self.components * ws[:, None, None], axis=0)
        faces = self.faces

        if not with_annotations:
            return Mesh(points, faces)

        annotations = {}
        for name in self.annotations:
            annotations[name] = self.sample_annotation(name, ws)

        # TODO: include annotations
        return Mesh(points, faces, annotations=annotations, descriptions=self.descriptions)

    def sample_trimesh(self, *weights: Union[float, int, np.ndarray]) -> trimesh.Trimesh:
        """Get a trimesh from the shape model.

        Args:
            weights (np.ndarray): The weights for each component.

        Returns:
            trimesh.Trimesh: The mesh.

        """
        return mesh_utils.to_trimesh(self.sample_pv(*weights))

    def to_npz(self, path: Path):
        """Save the model to an NPZ file.

        Args:
            path (Path): The path to the file.

        """
        np.savez_compressed(
            path,
            points=self.points,
            components=self.components,
            faces=self.faces,
            annotations=self.annotations,  # requires pickling
            descriptions=self.descriptions,
        )

    @classmethod
    def from_npz(cls: Type[StatisticalShapeModel], path: Path) -> StatisticalShapeModel:
        """Load a StatisticalShapeModel from an NPZ file.

        Args:
            path (Path): The path to the file.

        Returns:
            StatisticalShapeModel: The loaded model.

        """
        data = dict(np.load(path, allow_pickle=True))
        annotations = data.get("annotations", {})
        descriptions = data.get("descriptions", {})
        if not isinstance(annotations, dict):
            annotations = {}
        if not isinstance(descriptions, dict):
            descriptions = {}

        # log.debug(f"Loaded {path} with keys {list(data.keys())}")
        # log.debug(f"annotations: {data.get('annotations', dict())[0]}")
        return cls(
            data["points"],
            data["components"],
            data["faces"],
            annotations=annotations,
            descriptions=descriptions,
        )

    def fit(
        self,
        points_target: np.ndarray,
        max_iterations: int = 100,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        weights: Optional[np.ndarray] = None,
        freeze_weights: bool = False,
    ) -> Tuple[np.ndarray[np.float64], Mesh, geo.FrameTransform]:
        """Fit the model to a mesh.

        Args:
            target_points (np.ndarray): The points to fit to.
            weights (np.ndarray): The initial weights for each component. If None, use zeros.
            freeze_weights (bool): Whether to freeze the weights of the model. If True, the weights
                will not be updated during the fitting process from those provided. This is
                equivalent to ICP on the sampled shape.

        Returns:
            np.ndarray: The weights for each component.
            Mesh: The fitted mesh, in the same coordinate frame as the model.
            geo.FrameTransform: The model_from_target transform.

        """

        if weights is None:
            weights = np.zeros(self.n_components)
        elif weights.shape != (self.n_components,):
            log.warning(
                f"Expected weights to be of shape ({self.n_components},), got {weights.shape}. Continuing with zeros."
            )
            weights = np.zeros(self.n_components)
            fix_weights = False
        else:
            weights = weights.copy()

        model_mesh = self.sample(*weights, with_annotations=False)
        model_from_target = geo.F.from_rt(translation=-np.mean(points_target, axis=0))
        # TODO: need to put the deformed model back in the target_points frame.
        kwargs = dict(s=0.1, alpha=0.6)

        best_cost = np.inf
        prev_cost = np.inf
        best_weights = weights
        done = False
        for i in range(max_iterations):
            if done:
                break
            # TODO: rebuilding the coveriance mesh is slow, so we should edit the points inside and
            # only rebuild when the overlap is high.
            log.info(f"Iteration {i} of {max_iterations}.")

            t0 = time.time()
            model_from_target = model_mesh.register(
                points_target,
                initial=model_from_target,
                atol=atol,
                rtol=rtol,
                max_iterations=max_iterations,
            )
            log.debug(f"ICP took {time.time() - t0:.2f} seconds.")

            if freeze_weights:
                # Weights are frozen, so we are finished.
                break

            # [n, 3]
            ss = model_from_target.transform_points(points_target)

            prev_inner_cost = np.inf
            for j in range(max_iterations):

                # cs: [n, 3] corresponding to ss, on the mesh
                cs, distances, cidxs = model_mesh.query(ss, k=10)

                inner_cost = distances.mean()
                log.info(f"Inner iteration {j} of {max_iterations}. Cost: {inner_cost:.4f}.")
                if (
                    np.abs(inner_cost - prev_inner_cost) < atol + rtol * prev_inner_cost
                    or inner_cost > prev_inner_cost
                ):
                    break
                prev_inner_cost = inner_cost

                # [n] indices of the triangle vertices
                us = model_mesh.faces[cidxs, 0]
                vs = model_mesh.faces[cidxs, 1]
                ws = model_mesh.faces[cidxs, 2]

                # [n]
                ps = model_mesh.vertices[us]
                qs = model_mesh.vertices[vs]
                rs = model_mesh.vertices[ws]

                # [n, 3]
                # TODO: possible divide by zero here
                try:
                    cs_bary = geo_utils.barycentric(ps, qs, rs, cs)
                except ZeroDivisionError:
                    done = True
                    break

                if np.any(np.isnan(cs_bary)):
                    log.warning("NaN in barycentric coordinates. Breaking.")
                    done = True
                    break

                # Get the means (control points) for each of the triangle vertices, [n, 3]
                u_means = self.points[us]
                v_means = self.points[vs]
                w_means = self.points[ws]

                # Get the components for each of the triangle vertices, [n_components, n, 3]
                u_components = self.components[:, us]
                v_components = self.components[:, vs]
                w_components = self.components[:, ws]

                # [n, 1] * [n, 3] = [n, 3]
                qs_mean = (
                    cs_bary[:, 0, None] * u_means
                    + cs_bary[:, 1, None] * v_means
                    + cs_bary[:, 2, None] * w_means
                )

                # [1, f, 1] * [n_components, f, 3] = [n_components, f, 3]
                qs_component = (
                    cs_bary[None, :, 0, None] * u_components
                    + cs_bary[None, :, 1, None] * v_components
                    + cs_bary[None, :, 2, None] * w_components
                )

                # So we want ss = qs_mean + qs_component @ weights
                a = qs_component.reshape(self.n_components, -1).T
                b = (ss - qs_mean).reshape(-1)
                try:
                    ws, _, _, _ = np.linalg.lstsq(a, b)
                except np.linalg.LinAlgError:
                    done = True
                    break
                weights = np.array(ws, dtype=np.float64)

                model_mesh = self.sample(weights, with_annotations=False)

            cost = inner_cost
            if cost < best_cost:
                best_cost = cost
                best_weights = weights

            if np.abs(cost - prev_cost) < atol + rtol * prev_cost or cost > prev_cost:
                break

            prev_cost = cost

            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # ax.plot_trisurf(
            #     *model_mesh.vertices.T, triangles=model_mesh.faces, color="b", alpha=0.3
            # )
            # ax.scatter(*ss.T, c="r", **kwargs)
            # ax.axis("off")
            # ax.view_init(elev=-20, azim=-100, roll=0)
            # plt.tight_layout()
            # fig.savefig(images_dir / f"{i:03d}_step.png", dpi=300)
            # plt.close(fig)

        best_model_mesh = self.sample(best_weights, with_annotations=True).transform(
            model_from_target.inverse()
        )
        return best_weights, best_model_mesh, model_from_target

    @classmethod
    def from_meshes(
        cls,
        meshes: list[Union[pv.PolyData, str, list[str]]],
        n_points: int = 5000,
        n_components: int = 10,
        cache_dir: Optional[Path] = None,
        cache_names: Optional[list[str]] = None,
        landmarks: Optional[np.ndarray] = None,
        bootstrap: Optional[StatisticalShapeModel] = None,
        bootstrap_iters: int = 0,
        _bootstrap_iter: int = 0,
    ) -> StatisticalShapeModel:
        """Compute a statistical shape model from a set of shapes.

        Args:
            meshes (list[pv.PolyData]): The `n` shapes, or paths to them. If a list of paths is provided,
                resulting meshes will be combined.
            n_points (int): The approximate number of points to use in the model. The actual number
                may be slightly different.
            n_components (int): The number of components to use in the model.
            cache_dir: The directory to cache individual steps along the way. If None, no caching is done.
            cache_names: The unique names corresponding to each mesh. If cache_dir is not None, this must
                be provided to ensure that the cache is preserved across runs.
            landmarks: [n, m, 3] array of landmark annotations. If provided, these are used to align
                the shapes before computing the model.
            bootstrap: A model to bootstrap from. If provided, landmarks are only used to initialize
                the alignment, and the point correspondences are provided by fitting the bootstrap model.
            bootstrap_iters: The number of bootstrap iterations to perform. If 0, no bootstrap is performed.
            _bootstrap_iter: The current bootstrap iteration. This is used internally to recursively
                call this method. Do not provide this argument.

        TODO: Add support for bootstrapping, where the resulting ssm is used to initialize the next
        one. Can be done in a recursive call at the end, providing the current ssm and the bootstrap
        iter minus 1.

        Returns:
            StatisticalShapeModel: The computed model.
        """
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        if cache_names is None:
            raise ValueError("cache_names must be provided if cache_dir is provided")
        if len(cache_names) != len(meshes):
            raise ValueError("cache_names must be the same length as meshes")

        if cache_dir is None:
            raise ValueError("cache_dir must be provided")
        if cache_names is None:
            cache_names = [f"{i:06d}" for i in range(len(meshes))]

        # Make the various cache directories.
        decimated_dir = cache_dir / "decimated"
        registered_dir = cache_dir / f"registered_{_bootstrap_iter:02d}"
        deformed_dir = cache_dir / f"deformed_{_bootstrap_iter:02d}"
        deformed_stl_dir = cache_dir / f"deformed_stl_{_bootstrap_iter:02d}"
        weights_dir = cache_dir / f"weights_{_bootstrap_iter:02d}"
        prev_weights_dir = cache_dir / f"weights_{_bootstrap_iter - 1:02d}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        decimated_dir.mkdir(parents=True, exist_ok=True)
        registered_dir.mkdir(parents=True, exist_ok=True)
        deformed_dir.mkdir(parents=True, exist_ok=True)
        deformed_stl_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

        if bootstrap_iters > 0:
            log.info(
                f"--------------- Bootstrap Iteration {_bootstrap_iter + 1} / {bootstrap_iters} ---------------"
            )

        bootstrap_path = cache_dir / f"bootstrap_{_bootstrap_iter:02d}.npz"
        if bootstrap_iters > 0:
            if _bootstrap_iter > 0 and bootstrap is None:
                raise ValueError("bootstrap must be provided if bootstrap_iter > 0")

            if bootstrap_path.exists():
                log.info("Loading previous bootstrap from %s", bootstrap_path)
                ssm = cls.from_npz(bootstrap_path)
                return cls.from_meshes(
                    meshes,
                    n_points=n_points,
                    n_components=n_components,
                    cache_dir=cache_dir,
                    cache_names=cache_names,
                    landmarks=landmarks,
                    bootstrap=ssm,
                    bootstrap_iters=bootstrap_iters,
                    _bootstrap_iter=_bootstrap_iter + 1,
                )
            # if landmarks is None:
            # raise ValueError("landmarks must be provided if bootstrap_iter > 0")

        images_dir = Path(get_original_cwd()) / "images"
        if not images_dir.exists():
            images_dir.mkdir(parents=True)
        gif_dir = images_dir / f"pelvis_ssm_bootstrap_{_bootstrap_iter:02d}"

        if bootstrap is not None and not gif_dir.exists():
            plot_components(bootstrap, gif_dir, n=25, n_components=5)

        # Resample the meshes to a common number of points.
        log.info("Resampling meshes to %d points", n_points)

        decimated_meshes: list[pv.PolyData] = []
        ns: list[int] = []
        mesh_or_path_or_paths: Union[pv.PolyData, str, list[str]]
        mesh: pv.PolyData
        for i, mesh_or_path_or_paths in enumerate(track(meshes, "Decimating...")):

            if (
                decimated_dir is not None
                and (decimated_path := (decimated_dir / f"{cache_names[i]}_decimated.stl")).exists()
            ):
                decimated_mesh = pv.read(str(decimated_path))
            else:
                if isinstance(mesh_or_path_or_paths, pv.PolyData):
                    mesh = mesh_or_path_or_paths
                elif isinstance(mesh_or_path_or_paths, str):
                    mesh = pv.read(mesh_or_path_or_paths)
                elif isinstance(mesh_or_path_or_paths, list):
                    mesh = pv.PolyData()
                    for path in mesh_or_path_or_paths:
                        mesh += pv.read(path)
                mesh.decimate(
                    1 - n_points / mesh.n_points,
                    volume_preservation=True,
                    inplace=True,
                )
                if not mesh.is_all_triangles():
                    mesh.triangulate(inplace=True)

                decimated_mesh = mesh
                decimated_mesh.save(str(decimated_path))

            decimated_meshes.append(decimated_mesh)
            ns.append(decimated_mesh.n_points)

        n_points = max(ns)
        for i, mesh in enumerate(track(decimated_meshes, "Centering and padding...")):
            center = np.mean(mesh.points, axis=0)
            mesh.points -= center
            if landmarks is not None:
                landmarks[i] -= center

            # Add dummy points to make all meshes the same size.
            if mesh.n_points < n_points:
                mesh.points = np.pad(
                    mesh.points,
                    ((0, n_points - mesh.n_points), (0, 0)),
                    mode="edge",
                )

            decimated_meshes[i] = mesh

        assert all(mesh.n_points == n_points for mesh in decimated_meshes)

        reference_mesh_pv = decimated_meshes[0]
        reference_mesh = Mesh.from_pv(reference_mesh_pv)
        reference_points = np.array(reference_mesh_pv.points)
        shapes_points = [mesh.points for mesh in decimated_meshes]
        if landmarks is None:
            log.info("Aligning with Procrustes...")
            # Align the points in the meshes
            aligned_shapes, _ = generalized_procrustes(
                shapes_points, ref=reference_points, n_iter=1000
            )
        else:
            log.info("Aligning landmarks with point correspondence...")
            reference_landmarks = landmarks[0]
            aligned_landmarks = np.zeros_like(landmarks)
            aligned_landmarks[0] = reference_landmarks
            aligned_shapes = []
            aligned_shapes.append(reference_points)
            for i in range(1, len(landmarks)):
                aligned_from_instance = geo.F.from_point_correspondence(
                    reference_landmarks, landmarks[i]
                )
                aligned_landmarks[i] = aligned_from_instance.transform_points(landmarks[i])
                aligned_shapes.append(aligned_from_instance.transform_points(shapes_points[i]))
            aligned_shapes = np.array(aligned_shapes)

        # Register each shape to the reference.
        # If landmarks are present, use the landmarks to do the deformable registration,
        # then apply that transform on the whole shape.
        registered_meshes: list[pv.PolyData] = []
        deformed_meshes: list[Optional[Mesh]] = []
        for i, shape_points in enumerate(track(aligned_shapes, "Registering...")):
            log.info(f"Registering ({i + 1}/{len(aligned_shapes)})")
            registered_path = registered_dir / f"{cache_names[i]}_registered.stl"
            deformed_mesh_path = deformed_dir / f"{cache_names[i]}.npz"
            deformed_mesh_stl_path = deformed_stl_dir / f"{cache_names[i]}.stl"

            if registered_path.exists() and (bootstrap is None or deformed_mesh_path.exists()):
                log.info(f"Loading from cache {registered_path.name}, {deformed_mesh_path.name}...")
                registered_mesh = pv.read(str(registered_path))
                deformed_mesh = Mesh.from_npz(deformed_mesh_path) if bootstrap is not None else None

            elif bootstrap is None:
                # First iteration of a bootstrap, run deformable registration.
                reg = DeformableRegistration(X=reference_points, Y=np.array(shape_points))
                reg.register()
                registered_mesh = pv.PolyData(reg.TY, decimated_meshes[i].faces)
                deformed_mesh = reference_mesh
                # deformed_mesh = None

            else:
                # Bootstrap has been provided, and iterations are > 0
                # Also, the deformed mesh has not been saved yet.
                # Remember registered mesh and deformed mesh need to align with the reference.
                weights = None
                if (prev_weights_dir / f"{cache_names[i]}_weights.npy").exists():
                    # Initialize the weights to the previous iteration, assuming shape components are similar.
                    weights = np.load(prev_weights_dir / f"{cache_names[i]}_weights.npy")

                weights_path = weights_dir / f"{cache_names[i]}_weights.npy"
                if weights_path.exists():
                    weights = np.load(weights_path)
                    log.debug(f"Using existing weights for {cache_names[i]}")
                    weights, deformed_mesh, bootstrap_from_shape = bootstrap.fit(
                        shape_points,
                        weights=weights,
                        max_iterations=20,
                        atol=0.05,
                        freeze_weights=True,
                    )
                else:
                    try:
                        weights, deformed_mesh, bootstrap_from_shape = bootstrap.fit(
                            shape_points, weights=weights, max_iterations=20, atol=0.05
                        )
                    except RuntimeError:
                        log.warning(f"Bootstrap failed for {cache_names[i]}")
                        continue

                deformed_mesh = deformed_mesh.transform(bootstrap_from_shape)
                shape_points_in_reference = bootstrap_from_shape.transform_points(shape_points)
                np.save(str(weights_path), weights)
                deformed_mesh.to_npz(deformed_mesh_path)
                deformed_mesh.save(str(deformed_mesh_stl_path))
                registered_mesh = pv.PolyData(shape_points_in_reference, decimated_meshes[i].faces)

            registered_mesh.save(str(registered_path))
            registered_meshes.append(registered_mesh)
            deformed_meshes.append(deformed_mesh)

        # Use the registered meshes to order the indices of each mesh according to the closest point
        # on the reference. This means some points may be repeated and others dropped, but should be
        # fine overall.
        shapes: list[np.ndarray] = []
        for i, pv_mesh in enumerate(track(registered_meshes, "Ordering points...")):
            mesh = Mesh.from_pv(pv_mesh)
            if deformed_meshes[i] is not None:
                # Use the deformed mesh as the reference to order the points.
                shape_points, _, _ = mesh.query(deformed_meshes[i].vertices)
            else:
                shape_points, _, _ = mesh.query(reference_points)

            shapes.append(shape_points)

        shapes = np.array(shapes)

        # Run PCA on the shapes.
        # TODO: should everything be re-centered here?
        log.info("Running PCA...")
        n_points = shapes.shape[1]
        shapes = shapes.reshape(len(shapes), -1)
        # shapes = np.array(shapes).reshape(len(shapes), -1)
        pca = PCA(n_components=n_components)
        pca.fit(shapes)

        points = pca.mean_.reshape(n_points, 3)
        components = pca.components_.reshape(n_components, n_points, 3)

        ssm = cls(
            points=points,
            components=components,
            faces=reference_mesh_pv.faces,
        )
        ssm.to_npz(bootstrap_path)

        if _bootstrap_iter < bootstrap_iters:
            # If this is a bootstrap iteration, return
            return cls.from_meshes(
                meshes,
                n_points=n_points,
                n_components=n_components,
                cache_dir=cache_dir,
                cache_names=cache_names,
                landmarks=landmarks,
                bootstrap=ssm,
                bootstrap_iters=bootstrap_iters,
                _bootstrap_iter=_bootstrap_iter + 1,
            )
        else:
            return ssm


def plot_components(
    ssm: StatisticalShapeModel,
    output_dir: Path,
    n: int = 5,
    w: float = 1000,
    n_components: Optional[int] = None,
):
    """Plot variations of the SSM.

    Args:
        ssm: The SSM to plot.
        output_dir: The directory to save the plots to.
        n: The number of points to plot along each axis.
        w: The width of the coefficients to sample from.
        n_components: The number of components to plot.

    """
    # Make plots of the principle components.
    if n_components is None:
        n_components = ssm.n_components

    for i in range(n_components):
        component_dir = output_dir / f"lambda_{i}"
        if component_dir.exists():
            rmtree(component_dir)
        component_dir.mkdir(parents=True)
        for j, lam in enumerate(track(np.linspace(-w, w, n), description=f"Plotting lambda_{i}")):
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ws = np.zeros(ssm.n_components)
            ws[i] = lam
            mesh = ssm.sample(ws)
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                triangles=mesh.faces,
                Z=mesh.vertices[:, 2],
                color=[1, 1, 1, 0.5],
            )
            ax.view_init(elev=-20, azim=-100, roll=0)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(component_dir / f"lambda_{i:02d}_{j:02d}.png", dpi=400)
            plt.close()

        sp.Popen(
            [
                "/usr/bin/ffmpeg",
                "-framerate",
                "10",
                "-i",
                f"lambda_{i:02d}_%02d.png",
                f"../lambda_{i:02d}.gif",
            ],
            cwd=component_dir,
        )

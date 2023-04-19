from typing import Union, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import trimesh


def plot_trimesh(*meshes: trimesh.Trimesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for mesh in meshes:
        ax.plot_trisurf(
            mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2]
        )
    plt.show()


def plot_pyvista(*meshes: pv.PolyData, path: Optional[str] = None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for mesh in meshes:
        if not mesh.is_all_triangles():
            mesh.triangulate(inplace=True)
        ax.plot_trisurf(
            mesh.points[:, 0],
            mesh.points[:, 1],
            triangles=mesh.faces.reshape(-1, 4)[:, 1:],
            Z=mesh.points[:, 2],
        )

    if path is not None:
        plt.savefig(path, dpi=300)
    else:
        plt.show()

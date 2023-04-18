from typing import Optional
from typing import Tuple
from typing import Union, List

import logging
import numpy as np
from deepdrr import geo
from deepdrr.utils import listify, mappable

log = logging.getLogger(__name__)


def project_on_segment_np(c: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Project the point `c` onto the line segment `pq`."""
    # Get the vector from p to q.
    v = q - p

    # Get the vector from p to c.
    u = p - c

    # Project u onto v.
    t = -v.dot(u) / v.dot(v)

    # Get the projected point.
    if t <= 0:
        return p
    elif t >= 1:
        return q
    else:
        return v * t + p


def project_on_segment(c: geo.Point, p: geo.Point, q: geo.Point) -> geo.Point:
    """Project the point `c` onto the line segment `pq`.

    Args:
        c (geo.Point): Point to project.
        p (geo.Point): One endpoint of the line segment.
        q (geo.Point): Other endpoint of the line segment.

    Returns:
        geo.Point: Projected point.

    """
    # Ensure types.
    c = geo.point(c)
    p = geo.point(p)
    q = geo.point(q)

    # Get the vector from p to q.
    v = q - p

    # Get the vector from p to c.
    u = p - c

    # Project u onto v.
    t = -v.dot(u) / v.dot(v)

    # Get the projected point.
    if t <= 0:
        return p
    elif t >= 1:
        return q
    else:
        return v * t + p


def project_on_triangle(
    p: geo.Point3D, q: geo.Point3D, r: geo.Point3D, x: geo.Point3D
) -> geo.Point3D:
    """Get the closest point on the triangle `pqr` to `x`."""

    p = np.array(p)
    q = np.array(q)
    r = np.array(r)
    x = np.array(x)

    pq = q - p
    pr = r - p
    sol, _, _, _ = np.linalg.lstsq(np.array([pq, pr]).T, x - p, rcond=None)
    lam, mu = sol
    c = p + lam * pq + mu * pr

    if lam >= 0 and mu >= 0 and lam + mu <= 1:
        return geo.p(c)
    elif lam < 0:
        return project_on_segment(c, r, p)
    elif mu < 0:
        return project_on_segment(c, p, q)
    elif lam + mu > 1:
        return project_on_segment(c, q, r)
    else:
        raise RuntimeError("unhandled edge case")


def project_on_triangle_np(
    p: np.ndarray, q: np.ndarray, r: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Get the closest point on the triangle `pqr` to `x`."""
    pq = q - p
    pr = r - p
    sol, _, _, _ = np.linalg.lstsq(np.array([pq, pr]).T, x - p, rcond=None)
    lam, mu = sol
    c = p + lam * pq + mu * pr

    if lam >= 0 and mu >= 0 and lam + mu <= 1:
        return c
    elif lam < 0:
        return project_on_segment_np(c, r, p)
    elif mu < 0:
        return project_on_segment_np(c, p, q)
    elif lam + mu > 1:
        return project_on_segment_np(c, q, r)
    else:
        raise RuntimeError("unhandled edge case")


def triangle_area(p, q, r):
    a = p - q
    b = r - q
    num = a @ b
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if np.isclose(num, den):
        return 0.0
    sin_theta = np.sqrt(den**2 - num**2) / den
    out = den * sin_theta / 2
    return out


@mappable(ndim=1, every=True)
def barycentric(p: np.ndarray, q: np.ndarray, r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the barycentric coordinates of `x` in triangle `pqr`

    Finds the scalars u, v, w such that u + v + w == 1 and u*p + v*q + w*r =
    x. If 0 <= u, v, w <= 1 does not hold, then x is outside the triangle.

    Args:
        p (np.ndarray): Vertex of the triangle.
        q (np.ndarray): Vertex of the triangle.
        r (np.ndarray): Vertex of the triangle.
        x (np.ndarray): Point.

    Returns:
        np.ndarray: [u, v, w] barycentric coordinates of `x` in triangle `pqr`.

    """
    pqr = triangle_area(p, q, r)
    u = triangle_area(p, q, x) / pqr
    v = triangle_area(p, r, x) / pqr
    w = 1 - u - v
    return np.array([u, v, w], np.float64)


@mappable(ndim=1, every=True)
def from_barycentric(p: np.ndarray, q: np.ndarray, r: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the point `x` from barycentric coordinates `b`.

    Args:
        p (np.ndarray): Vertex of the triangle.
        q (np.ndarray): Vertex of the triangle.
        r (np.ndarray): Vertex of the triangle.
        b (np.ndarray): Barycentric coordinate.

    Returns:
        np.ndarray: Point `x` in triangle `pqr` corresponding to barycentric coordinates
            `b = [u, v, w]`
    """
    return b[0] * p + b[1] * q + b[2] * r


def fit_line(points: np.ndarray) -> Tuple[geo.Point, geo.Vector]:
    """Fit a line to points in 3D

    Args:
        points (np.ndarray): [N, 3] array of points.

    Returns:
        geo.Point3D, geo.Vector3D: Point, vector parameterization of the best fit line.
    """
    points = np.array(points)
    c = points.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(points - c[np.newaxis, :])

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    direction = geo.vector(vv[0]).hat()
    return geo.point(c), direction


def camera_point_from_index(
    pred_startpoint_in_index: geo.Point2D,
    d: float,
    camera3d_from_world: geo.FrameTransform,
    index_from_world: geo.Transform,
) -> geo.Point3D:
    # Get the vector (in camera space), going from the source to the landmark pixel,
    a_ray_hat = (camera3d_from_world @ index_from_world.inv @ pred_startpoint_in_index).hat()

    a_z = d
    a_y = a_ray_hat[1] * d / a_ray_hat[2]
    num = a_z * a_z + a_y * a_y
    den = a_ray_hat[2] * a_ray_hat[2] + a_ray_hat[1] * a_ray_hat[1]
    a_x = a_ray_hat[0] * np.sqrt(num / den)
    pred_startpoint_in_camera3d = geo.point(a_x, a_y, a_z)
    return pred_startpoint_in_camera3d


def distance_to_line(points: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Get the distance to the line for a bunch of points.

    Args:
        points (np.ndarray): The points as an [N, D] array.
        c (np.ndarray): A point on the line.
        v (np.ndarray): A (unit?) vector pointing in the direction of the line.

    Returns:
        np.ndarray: The distance to the line for each point.
    """
    v = v / np.linalg.norm(v)
    diff = points - c
    return np.linalg.norm(diff - (diff @ v[:, np.newaxis]) * v[np.newaxis, :], axis=1)


def points_on_line(points: np.ndarray, c: np.ndarray, v: np.ndarray):
    v = v / np.linalg.norm(v)
    diff = points - c
    return c + (diff @ v[:, np.newaxis]) * v[np.newaxis, :]


def us_on_line(points: np.ndarray, c: np.ndarray, v: np.ndarray):
    v = v / np.linalg.norm(v)
    diff = points - c
    return (diff @ v[:, np.newaxis])[:, 0]


def geo_points_on_line(point: Union[geo.Point, List[geo.Point]], c: geo.Point, v: geo.Vector):
    """Deepdrr geo version, more flexible."""
    points = listify(point)
    ps = points_on_line(np.array(points), np.array(c), np.array(v))
    if isinstance(point, (list, tuple)):
        return [geo.point(p) for p in ps]
    elif isinstance(point, geo.Point):
        return geo.point(ps[0])
    else:
        raise TypeError(f"unexpected type: {type(point)}")


def geo_distance_to_line(point: Union[geo.Point, List[geo.Point]], c: geo.Point, v: geo.Vector):
    """Deepdrr geo version, more flexible."""
    points = listify(point)
    ps = distance_to_line(np.array(points), np.array(c), np.array(v))
    if isinstance(point, (list, tuple)):
        return [geo.point(p) for p in ps]
    elif isinstance(point, geo.Point):
        return ps[0]
    else:
        raise TypeError(f"unexpected type: {type(point)}")


def radius_of_circumscribing_cylinder(p: geo.Point3D, q: geo.Point3D, l: geo.Line3D) -> float:
    """Get the radius of a cylinder centered on (p -> q) that circumscribes the line.

    This is the radius of the cylinder with one end at p, the other at q, and large enough that the
    line enters one end and exits the other.

    Args:
        p (geo.Point3D): One end of the line segment.
        q (geo.Point3D): Other end of the line segment.
        l (geo.Line3D): The line.

    Returns:
        float: Radius of the circumscribing cylinder.
    """

    # Normal of ends.
    n = q - p

    pl_p = geo.plane(p, n)
    pl_q = geo.plane(q, n)

    # Points on each cylinder end through which the line passes.
    lp = pl_p.meet(l)
    lq = pl_q.meet(l)

    # Radius of the cylinder is the max of the two distances.
    return max((lp - p).norm(), (lq - q).norm())

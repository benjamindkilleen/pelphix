from __future__ import annotations
from typing import Optional, Tuple, List, NamedTuple
import math
import numpy as np
from deepdrr import geo
from deepdrr.utils import data_utils
from pathlib import Path
from shapely.geometry import Polygon
import logging


log = logging.getLogger(__name__)


# The four lines that define the edges of the cylinder projection.
# left/right sides are from the perspective of the start, looking at end
class CylinderProjectionEdges(NamedTuple):
    start: geo.Segment2D
    end: geo.Segment2D
    side_left: geo.Segment2D
    side_right: geo.Segment2D

    def get_points(self) -> List[geo.Point2D]:
        return [
            self.start.q,
            self.side_left.q,
            self.end.q,
            self.side_right.q,
        ]

    def get_segmentation(self, image_size: tuple[int, int]) -> Tuple[List[List[int]], float]:
        """Constrain the projection to the image.

        Args:
            image_size (tuple[int, int]): The size of the image (height, width).

        Returns:
            CylinderProjectionEdges: The constrained projection. If the projection is not visible in the image,
                returns None.
        """
        height, width = image_size
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
        image_rect = Polygon(corners)
        corr_proj = Polygon([tuple(p) for p in self.get_points()])

        intersection = image_rect.intersection(corr_proj)
        if intersection.area > 0:
            poly = np.array(intersection.exterior.coords).reshape(-1).tolist()
            return [poly], intersection.area
        else:
            return [], 0


class Cylinder:
    def __init__(self, startpoint: geo.Point3D, endpoint: geo.Point3D, radius: float):
        """Create a Cylinder.

        Args:
            startpoint (geo.Point3D): The startpoint of the cylinder (world coordinates).
            endpoint (geo.Point3D): The endpoint of the cylinder (world coordinates).
            radius (float): The radius of the cylinder.
        """
        self.startpoint = geo.point(startpoint)
        self.endpoint = geo.point(endpoint)
        self.radius = radius

    @classmethod
    def from_fcsv(cls, path: Path, radius: float = 1) -> Cylinder:
        """Create a Cylinder from an FCSV file.

        If the FCSV file contains more than one point, the first two points are used
        and a warning is printed.

        Args:
            path: The path to the FCSV file.
            radius (float): The radius of the cylinder.

        Returns:
            Cylinder: The cylinder.
        """
        points, _ = data_utils.load_fcsv(path)
        if len(points) > 2:
            log.warning(f"Found more than two points in {path}. Using the first two points.")
        return cls(points[0], points[1], radius)

    def transform(self, transform: geo.Transform) -> Cylinder:
        """Transform the cylinder.

        Args:
            transform (geo.Transform): The transformation to apply. Should be rigid (not checked).

        Returns:
            Cylinder: The transformed cylinder.
        """
        return Cylinder(transform @ self.startpoint, transform @ self.endpoint, self.radius)

    def _project_edges(
        self, index_from_world: geo.CameraProjection
    ) -> Tuple[geo.Line2D, geo.Line2D]:
        """Project the outer edges of the cylinder to the image index space.

        Args:
            index_from_world (geo.CameraProjection): The camera projection matrix.

        Returns:
            Tuple[geo.Line2D, geo.Line2D]: The projected edges of the cylinder (in no particular order).
        """
        centerline = geo.line(self.startpoint, self.endpoint)
        v = self.endpoint - self.startpoint
        s = index_from_world.center_in_world  # camera source
        c = centerline.project(s)  # center of cylinder closest to source
        d = c - s  # vector from source to center of cylinder
        theta = math.asin(self.radius / d.norm())  # angle between d and cylinder edge
        p1 = c + d.rotate(v, theta)
        p2 = c + d.rotate(v, -theta)
        g1 = geo.line(p1, p1 + v)
        g2 = geo.line(p2, p2 + v)
        return index_from_world @ g1, index_from_world @ g2

    def project(self, index_from_world: geo.CameraProjection) -> CylinderProjectionEdges:
        """Project the cylinder to the image index space.

        Args:
            index_from_world (geo.CameraProjection): The camera projection matrix.


        ```

                   0                side_left                  1
                    +-----------------------------------------+
                    |                                         |
            start   |         ------------------>             | end
                    |                                         |
                    +-----------------------------------------+
                   3                side_right                 2

        ```


        Returns:
            CylinderProjectionEdges: The projected edges of the cylinder, as `(start, end, side_left, side_right)`
        """
        startpoint_index = index_from_world @ self.startpoint
        endpoint_index = index_from_world @ self.endpoint
        direction_index = endpoint_index - startpoint_index
        radius_seg = geo.segment(
            self.midpoint, self.radius * index_from_world.principle_ray_in_world.perpendicular()
        )
        radius_seg_index = index_from_world @ radius_seg

        if direction_index.norm() < (radius_index := radius_seg_index.length()):
            # The cylinder is mostly being seen from the side, so just make a rectangle
            # around its projected midpoint
            c = radius_seg_index.midpoint()
            p0 = c + geo.v(-radius_index, -radius_index)
            p1 = c + geo.v(radius_index, -radius_index)
            p2 = c + geo.v(radius_index, radius_index)
            p3 = c + geo.v(-radius_index, radius_index)

        else:
            centerline = geo.line(self.startpoint, self.endpoint)
            v = self.endpoint - self.startpoint
            s = index_from_world.center_in_world  # camera source
            c = centerline.project(s)  # center of cylinder closest to source
            d = c - s  # vector from source to center of cylinder
            d_mag = d.norm()

            if d_mag < self.radius:
                # In this case, just project the center segment with a few pixels of padding
                seg = index_from_world @ geo.segment(self.startpoint, self.endpoint)
                direction_index = seg.get_direction()
                left_perpendicular = geo.vector(-direction_index.y, direction_index.x)
                p0 = seg.p + left_perpendicular.hat()
                p1 = seg.q + left_perpendicular.hat()
                p2 = seg.q - left_perpendicular.hat()
                p3 = seg.p - left_perpendicular.hat()
                return CylinderProjectionEdges(
                    start=geo.segment(p3, p0),
                    side_left=geo.segment(p0, p1),
                    end=geo.segment(p2, p3),
                    side_right=geo.segment(p1, p2),
                )

            try:
                theta = math.asin(self.radius / d.norm())  # angle between d and cylinder edge
            except ValueError as e:
                log.debug(
                    f"v: {v}, s: {s}, c: {c}, d: {d}, |d|: {d.norm()}, radius: {self.radius}, radius / |d|: {self.radius / d.norm()}"
                )
                raise e

            p1 = c + d.rotate(v, theta)
            p2 = c + d.rotate(v, -theta)
            g0 = geo.line(p1, p1 + v)
            g1 = geo.line(p2, p2 + v)

            # Get the lines in index space
            g0_index = index_from_world @ g0
            g1_index = index_from_world @ g1
            left_perpendicular = geo.vector(-direction_index.y, direction_index.x)
            startline = geo.line(startpoint_index, left_perpendicular)
            endline = geo.line(endpoint_index, left_perpendicular)

            # Order the sides by their projection onto left_perpendicular
            p0 = startline.meet(g0_index)
            if (p0 - startpoint_index).dot(left_perpendicular) > 0:
                side_left = g0_index
                side_right = g1_index
            else:
                side_left = g1_index
                side_right = g0_index

            p0 = startline.meet(side_left)
            p1 = side_left.meet(endline)
            p2 = endline.meet(side_right)
            p3 = side_right.meet(startline)

        return CylinderProjectionEdges(
            start=geo.segment(p3, p0),
            side_left=geo.segment(p0, p1),
            end=geo.segment(p1, p2),
            side_right=geo.segment(p2, p3),
        )

    def flip(self) -> Cylinder:
        """Flip the cylinder.

        Returns:
            Cylinder: The flipped cylinder.
        """
        return Cylinder(self.endpoint, self.startpoint, self.radius)

    @property
    def centerline(self) -> geo.Line3D:
        """The centerline of the cylinder.

        Returns:
            geo.Line3D: The centerline.
        """
        return geo.line(self.startpoint, self.endpoint)

    @property
    def center_ray(self) -> geo.Ray3D:
        """The centerline of the cylinder.

        Returns:
            geo.Ray3D: The centerline.
        """
        return geo.ray(self.startpoint, self.endpoint)

    def shorten(self, fraction: float) -> Cylinder:
        """Shorten the cylinder by moving the endpoint closer to the startpoint.

        Args:
            fraction (float): Fraction of the cylinder to keep.

        Returns:
            Cylinder: The shortened cylinder.
        """
        return Cylinder(
            startpoint=self.startpoint,
            endpoint=self.startpoint.lerp(self.endpoint, fraction),
            radius=self.radius,
        )

    def make_length(self, length: float) -> Cylinder:
        """Make the cylinder a certain length.

        Args:
            length (float): The length.

        Returns:
            Cylinder: The cylinder.
        """
        return Cylinder(
            startpoint=self.startpoint,
            endpoint=self.startpoint + self.get_direction().hat() * length,
            radius=self.radius,
        )

    def length(self) -> float:
        """The length of the cylinder.

        Returns:
            float: The length.
        """
        return (self.endpoint - self.startpoint).norm()

    def get_point(self) -> geo.Point3D:
        """Get the startpoint."""
        return self.startpoint

    def get_direction(self) -> geo.Vector3D:
        """Get the direction of the cylinder."""
        return self.endpoint - self.startpoint

    def startplane(self) -> geo.Plane:
        """Get the plane at the startpoint.

        Returns:
            geo.Plane3D: The plane.
        """
        return geo.plane(self.startpoint, self.get_direction())

    def endplane(self) -> geo.Plane:
        """Get the plane at the endpoint.

        Returns:
            geo.Plane3D: The plane.
        """
        return geo.plane(self.endpoint, -self.get_direction())

    @property
    def midpoint(self) -> geo.Point3D:
        return self.startpoint + (self.endpoint - self.startpoint) / 2

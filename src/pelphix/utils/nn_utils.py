import logging
from turtle import distance
from typing import List, overload
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from deepdrr import geo
from deepdrr.utils import listify
from scipy import ndimage

from .geo_utils import camera_point_from_index
from .geo_utils import distance_to_line
from .geo_utils import fit_line
from .geo_utils import geo_points_on_line
from .geo_utils import points_on_line

log = logging.getLogger(__name__)


def normalize_heatmap(heatmap: torch.Tensor, min_range: float = 0.0001) -> torch.Tensor:
    """Normalize the heatmap to have values between 0 and 1.

    Args:
        heatmap (torch.Tensor): A 2D array

    Returns:
        torch.Tensor: A 2D array with values between 0 and 1
    """
    hmin = torch.min(heatmap)
    hmax = torch.max(heatmap)
    return (heatmap - hmin) / torch.maximum(
        torch.tensor(min_range, dtype=heatmap.dtype, device=heatmap.device), hmax - hmin
    )


def line_heatmap_numpy(
    x1: float, y1: float, x2: float, y2: float, scale: float, size: Tuple[int, int]
) -> np.ndarray:
    """Create a heatmap for an infinite line in the image.

    Args:
        x1, y1 (float): Startpoint in image-space (i,j)
        x2, y2 (float): Endpoint in image-space (i,j)
        scale (float): Scale of the exponential distribution
        size (Tuple[int, int]): Image size as (H, W)

    Returns:
        np.ndarray: An array containing a heatmap to a line segment, as a float32.
    """
    H, W = size
    xs, ys = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    den = 2 * scale * scale
    xdiff = x2 - x1
    ydiff = y2 - y1
    distance_squared = np.square(xdiff * (y1 - ys) - ydiff * (x1 - xs)) / (
        np.square(xdiff) + np.square(ydiff)
    )
    h = np.exp(-distance_squared / den)
    return h.astype(np.float32)


def line_segment_heatmap_numpy(
    x1: float, y1: float, x2: float, y2: float, scale: float, size: Tuple[int, int]
) -> np.ndarray:
    """Create a heatmap for a line segment in the image.

    Args:
        x1, y1 (float): Startpoint in image-space (i,j)
        x2, y2 (float): Endpoint in image-space (i,j)
        scale (float): Scale of the exponential distribution
        size (Tuple[int, int]): Image size as (H, W)

    Returns:
        np.ndarray: A tensor containing a heatmap to a line segment, as a float32.
    """
    H, W = size
    xs, ys = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    den = 2 * scale * scale
    px = x2 - x1
    py = y2 - y1
    norm_sqr = px * px + py * py
    u = ((xs - x1) * px + (ys - y1) * py) / norm_sqr
    u = np.clip(u, 0, 1)
    x = x1 + u * px
    y = y1 + u * py
    dx = x - xs
    dy = y - ys
    distance_squared = np.square(dx) + np.square(dy)
    h = np.exp(-distance_squared / den)
    return h.astype(np.float32)


def line_heatmap(
    x1: float, y1: float, x2: float, y2: float, scale: float, size: Tuple[int, int]
) -> torch.Tensor:
    """Create a heatmap for an infinite line in the image.

    Args:
        x1, y1 (float): Startpoint in image-space (i,j)
        x2, y2 (float): Endpoint in image-space (i,j)
        scale (float): Scale of the exponential distribution
        size (Tuple[int, int]): Image size as (H, W)

    Returns:
        torch.Tensor: A tensor containing a heatmap to a line segment, as a float32.
    """
    H, W = size
    xs, ys = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
    )
    den = 2 * scale * scale
    xdiff = torch.tensor(x2 - x1, dtype=torch.float32)
    ydiff = torch.tensor(y2 - y1, dtype=torch.float32)
    distance_squared = torch.square(xdiff * (y1 - ys) - ydiff * (x1 - xs)) / (
        xdiff.square() + ydiff.square()
    )
    h = torch.exp(-distance_squared / den)
    return h


def line_segment_heatmap(
    x1: float, y1: float, x2: float, y2: float, scale: float, size: Tuple[int, int]
) -> torch.Tensor:
    """Create a heatmap for a line segment in the image.

    Args:
        x1, y1 (float): Startpoint in image-space (i,j)
        x2, y2 (float): Endpoint in image-space (i,j)
        scale (float): Scale of the exponential distribution
        size (Tuple[int, int]): Image size as (H, W)

    Returns:
        torch.Tensor: A tensor containing a heatmap to a line segment, as a float32.
    """
    H, W = size
    xs, ys = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
    )
    den = 2 * scale * scale
    px = torch.tensor(x2 - x1, dtype=torch.float32)
    py = torch.tensor(y2 - y1, dtype=torch.float32)
    norm_sqr = px.square() + py.square()
    u = ((xs - x1) * px + (ys - y1) * py) / norm_sqr
    u = torch.clip(u, 0, 1)
    x = x1 + u * px
    y = y1 + u * py
    dx = x - xs
    dy = y - ys
    distance_squared = dx.square() + dy.square()
    h = torch.exp(-distance_squared / den)
    return h


def heatmap_numpy(mu_x: float, mu_y: float, scale: float, size: Tuple[int, int]) -> np.ndarray:
    """Create a heatmap for a point in the image.

    Args:
        mu_x, mu_y (float): Mean of the normal distribution in image-space (i,j)
        scale (float): Scale of the normal distribution in image-space (i,j)
        size (Tuple[int, int]): Image size as (H, W)

    Returns:
        np.ndarray: An array containing a heatmap to a point, as a float32.
    """
    H, W = size
    xs, ys = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    xdiff = xs - mu_x
    ydiff = ys - mu_y
    distance_squared = np.square(xdiff) + np.square(ydiff)
    h = np.exp(-distance_squared / (2 * scale * scale))
    return h.astype(np.float32)


def heatmap(mu_x: float, mu_y: float, scale: float, size: Tuple[int, int]) -> torch.Tensor:
    """Create a single heatmap.

    Args:
        mu_x (float): Center of the bell in first axis.
        mu_y (float): Center of the bell in second axis.
        scale (float): Scale of the distribution in pixels.
        size (Tuple[int, int]): Size of the image array as (H, W).

    Returns:
        Tensor: An [H, W] array containing the heatmap.
    """
    H, W = size
    xs, ys = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
    )
    den = 2 * scale * scale
    h = torch.exp(-((xs - mu_x).pow(2) + (ys - mu_y).pow(2)) / den)
    # Sometimes normalize here by dividing by the max, but can cause nan errors.
    return h


def heatmaps(
    centers: List[Optional[Tuple[float, float]]],
    scale: Union[float, List[float]],
    size: Tuple[int, int],
) -> torch.Tensor:
    """Create C heatmaps.

    Args:
        centers (List[Tuple[float, float]]): List of centers. Elements can be None for all zero heatmap in that channel.
        scale (Union[float, List[float]]): A single scale or list of scales for each heatmap.
        size (Tuple[int, int]): Shape of the output image.

    Returns:
        Tensor: [C, H, W] shaped array of heatmaps.
    """
    channels = len(centers)
    scales = listify(scale, channels)
    return torch.stack(
        [
            np.zeros(size, np.float32)
            if centers[c] is None
            else heatmap(*centers[c], scales[c], size)
            for c in range(channels)
        ],
        axis=0,
    )


def spatial_argmax(heatmap: torch.Tensor) -> torch.Tensor:
    """Get the indices of peaks in the heatmap.

    Args:
        heatmap (torch.Tensor): A tensor with shape `[N, C, H, W]`

    Returns:
        torch.Tensor: A tensor with shape `[N, C, 2]`
    """
    N, C, H, W = heatmap.shape
    flat_indices = torch.argmax(heatmap.view(N, C, H * W), dim=2, keepdim=True)
    h_indices = flat_indices // W
    w_indices = flat_indices % W
    return torch.cat([h_indices, w_indices], dim=2)


def spatial_argmax_numpy(heatmap: np.ndarray) -> np.ndarray:
    """Get the indices of peaks in the heatmap.

    Args:
        heatmap (numpy): A 2D array

    Returns:
        :
    """
    h, w = heatmap.shape
    flat_idx = np.argmax(heatmap.flat)
    i = flat_idx // w
    j = flat_idx % w
    return np.array([i, j])


def get_points_near(expected: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Get an array of all the (i,j) coordinates withing `distance_threshold` of expected.

    Args:
        expected (np.ndarray): [description]
        distance_threshold (float): [description]

    Returns:
        np.ndarray: [description
    """
    out = []
    for di in np.arange(-distance_threshold, distance_threshold + 1, 1):
        for dj in np.arange(-distance_threshold, distance_threshold + 1, 1):
            out.append([expected[0] + di, expected[1] + dj])

    out = np.array(out)
    indices = np.linalg.norm(out - expected[np.newaxis, :], axis=1) < distance_threshold
    return out[indices].astype(np.int64)


@overload
def get_heatmap_threshold(h: np.ndarray, fraction: float) -> float:
    ...


@overload
def get_heatmap_threshold(h: torch.Tensor, fraction: float) -> torch.Tensor:
    ...


def get_heatmap_threshold(h: np.ndarray, fraction=0.5) -> float:
    """Get the threshold for a heatmap.

    Args:
        h (np.ndarray): A 2D array
        fraction (float): Fraction of the heatmap range to set the threshold at. Higher values keeps fewer pixels.

    """
    hmin = h.min()
    hmax = h.max()
    return hmin + (hmax - hmin) * fraction


def detect_landmark(
    heatmap: np.ndarray,
    hip_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Detect a landmark in the (single) heatmap.

    Args:
        heatmap (np.ndarray): A 2D array
        hip_mask (Optional[np.ndarray]): A 2D array of the same shape as `heatmap`. If not None, the landmark will be

    Returns:
        Optional[np.ndarray]: The (i,j) coordinates of the landmark, or None if no landmark was detected.

    """
    if len(heatmap.shape) != 2:
        raise ValueError("Expected a 2D heatmap. Got shape: {}".format(heatmap.shape))

    threshold = get_heatmap_threshold(heatmap)
    indices = heatmap > threshold
    if not indices.any():
        return None

    if hip_mask is not None:
        raise NotImplementedError("Hip mask not implemented yet.")

    heatmap_over_threshold = np.where(indices, heatmap, 0)
    labels, num_objects = ndimage.label(heatmap_over_threshold)
    centers = ndimage.center_of_mass(
        heatmap_over_threshold, labels=labels, index=range(1, num_objects + 1)
    )
    if isinstance(centers, tuple):
        return np.array(centers)

    elif isinstance(centers, list) and len(centers) == 1:
        return np.array(centers[0])

    elif isinstance(centers, list) and len(centers) > 0:
        if len(centers) > 30:
            log.debug("too many landmarks (model likely not trained)")
            return None
        log.debug(f"Found multiple landmarks: {centers}")
        cs = np.array(centers)
        max_c = None
        max_value = None
        for c in cs:
            i, j = c.astype(int)
            if (0 <= i < heatmap.shape[0]) and (0 <= j < heatmap.shape[1]):
                value = heatmap[i, j]
                if max_value is None or value > max_value:
                    max_value = value
                    max_c = c
        return max_c
    else:
        return None


def detect_landmarks(
    heatmaps: np.ndarray,
    order: str = "ij",
) -> np.ndarray:
    """Detect landmarks in heatmaps.

    Args:
        heatmaps (np.ndarray): A 3D array of shape `[N, H, W]`.
        order (str, optional): The format of the output. Can be "ij" or "xy". Defaults to "ij".

    Returns:
        np.ndarray: An array of shape `[N, 2]` with the (i,j) coordinates of the landmarks.
    """

    num_keypoints, H, W = heatmaps.shape
    keypoint_preds = np.zeros((num_keypoints, 2), dtype=np.float32)
    for k in range(num_keypoints):
        heatmap = heatmaps[k]
        keypoint = detect_landmark(heatmap)
        if keypoint is None:
            continue
        x, y = keypoint
        if not (0 <= x < W and 0 <= y < H):
            continue

        keypoint_pred_ij = detect_landmark(heatmap)
        if keypoint_pred_ij is None:
            continue

        if order == "xy":
            keypoint_pred = np.array([keypoint_pred_ij[1], keypoint_pred_ij[0]])
        elif order == "ij":
            keypoint_pred = keypoint_pred_ij
        else:
            raise ValueError(f"Invalid order: {order}")

        keypoint_preds[k] = keypoint_pred

    return keypoint_preds


n = 0


def get_corridor_direction(
    heatmap: np.ndarray, threshold: float = 0.5, distance_threshold: float = 200
) -> Tuple[float, float]:
    """Get the direction of the corridor starting at image coordinates (si, sj).

    TODO: select threshold more intelligently.

    Args:
        heatmap (np.ndarray): The heatmap with a corridor.
        distance_threshold: We only take points of the heatmap that are this far away from the ~approximate~ startpoint,
            as estimated by the pixel in the region furthest down.

    Returns:
        Tuple[float, float]: The (i,j) direction of the corridor (non-normalized).
    """
    # Used to be necessary due to learning bug. Shouldn't be anymore.
    # heatmap_min = heatmap.min()
    # heatmap_max = heatmap.max()
    # heatmap = (heatmap - heatmap_min) / heatmap_max

    log.debug(f"heatmap; {heatmap.min(), heatmap.max()}")
    heatmap_binary = (heatmap > threshold).astype(np.uint8)
    if not np.any(heatmap_binary):
        return None, None
    labeled, num_features = ndimage.label(heatmap_binary)

    # Selects by the biggest region
    # sizes = [np.sum(labeled == f) for f in range(1, num_features + 1)]
    # corridor_label = np.argmax(sizes) + 1

    # Selects the most confident region.
    h_values = [np.max(heatmap[labeled == f]) for f in range(1, num_features + 1)]
    corridor_label = np.argmax(h_values) + 1

    # plt.close()
    # Now remove all the pixels that are too far away, starting from the bottom of the image.
    # TODO: instead, threshold this by distance from the startpoint
    xs, ys = np.where(labeled == corridor_label)
    idx = np.argmax(xs)
    x0 = xs[idx]
    y0 = ys[idx]
    distances = np.linalg.norm([xs - x0, ys - y0], axis=0)
    indexing = distances < distance_threshold
    xs = xs[indexing]
    ys = ys[indexing]

    # Now get the line of best fit
    m, b = np.polyfit(xs, ys, 1, w=heatmap[xs, ys])
    x1 = xs.max()
    y1 = m * x1 + b
    x2 = xs.min()
    y2 = m * x2 + b
    # plt.imshow(labeled, interpolation="none")
    # plt.colorbar()
    # plt.plot(y1, x1, 'r.')
    # plt.plot(y2, x2, 'b.')
    # plt.plot(m * xs + b, xs, 'r-', linewidth=0.7)
    # global n
    # plt.savefig(f"heatmap_binary_{n:04d}.png", dpi=150)
    # n += 1
    return x2 - x1, y2 - y1


def reconstruct_corridor_direction(
    heatmap: np.ndarray,
    rescaled_depth_map: np.ndarray,
    startpoint_in_index: geo.Point2D,
    startpoint_in_camera3d: geo.Point3D,
    camera3d_from_world: geo.FrameTransform,
    index_from_world: geo.Transform,
    distance_threshold: float = 200,
) -> Optional[geo.Vector3D]:
    """Get the direction of the corridor in 3D.

    TODO: select threshold more intelligently.
    Args:
        heatmap (np.ndarray): The heatmap with a corridor.
        depth_map: the depth map, scaled back to mm
        distance_threshold: We only take points of the heatmap that are this far away from the ~approximate~ startpoint,
            as estimated by the pixel in the region furthest down.

    Returns:
        Vector3D: direction in camera of the point
    """

    threshold = get_heatmap_threshold(heatmap)
    log.debug(f"threshold: {threshold}")
    heatmap_binary = (heatmap >= threshold).astype(np.uint8)

    if not np.any(heatmap_binary):
        log.debug(f"binary heatmap empty")
        return None
    labeled, num_features = ndimage.label(heatmap_binary)

    # Take the biggest region
    sizes = np.array([np.sum(labeled == f) for f in range(1, num_features + 1)])
    log.debug(f"region sizes: {sizes}")
    corridor_labels = np.argsort(sizes)[-3:] + 1

    mask = np.isin(labeled, corridor_labels)

    points = []
    max_distance = 0
    furthest_point = None
    for i, j in zip(*np.where(np.logical_and(mask, heatmap_binary))):
        p_index = geo.point(j, i)
        distance = geo.vector(p_index - startpoint_in_index).norm()

        if distance > distance_threshold:
            continue

        d = rescaled_depth_map[i, j]
        point = camera_point_from_index(
            p_index,
            d,
            camera3d_from_world=camera3d_from_world,
            index_from_world=index_from_world,
        )
        points.append(point)

        if furthest_point is None or distance > max_distance:
            max_distance = distance
            furthest_point = point

    if not points:
        return None

    _, w = fit_line(points)
    if geo.vector(furthest_point - startpoint_in_camera3d).dot(w) < 0:
        w = -w

    return w


def reconstruct_corridor(
    heatmap: np.ndarray,
    rescaled_depth_map: np.ndarray,
    startpoint_in_index: geo.Point2D,
    startpoint_in_camera3d: geo.Point3D,
    camera3d_from_world: geo.FrameTransform,
    index_from_world: geo.Transform,
    gt_startpoint_in_camera3d: Optional[geo.Point3D] = None,
    gt_endpoint_in_camera3d: Optional[geo.Point3D] = None,
) -> Tuple[Optional[geo.Point3D], Optional[geo.Vector3D]]:
    """Get the corridor in 3D.

    Args:
        heatmap (np.ndarray): The heatmap with a corridor.
        depth_map: the depth map, scaled back to mm
        distance_threshold: We only take points of the heatmap that are this far away from the ~approximate~ startpoint,
            as estimated by the pixel in the region furthest down.

    Returns:
        Vector3D: direction in camera of the point
    """
    value_at_startpoint = heatmap[int(startpoint_in_index.y), int(startpoint_in_index.x)]
    threshold = max(np.quantile(heatmap, 0.98), get_heatmap_threshold(heatmap, fraction=0.45))
    # threshold = np.quantile(heatmap, 0.995)
    log.debug(f"threshold: {threshold}")
    heatmap_binary = (heatmap >= threshold).astype(np.uint8)
    # plt.imshow(heatmap_binary)
    # plt.savefig(f"heatmap_binary.png", dpi=150)
    # plt.close()

    # plt.imshow(heatmap)
    # plt.colorbar()
    # plt.savefig("heatmap.png", dpi=150)
    # plt.close()

    if not np.any(heatmap_binary):
        log.debug(f"binary heatmap empty")
        return None, None

    rr, cc = np.where(heatmap_binary)

    # column, row
    points_index = np.stack([cc, rr], axis=1)

    # m, b = np.polyfit(cc, rr, 1, w=heatmap[rr, cc])
    c_index, w_index = fit_line(points_index)

    # make the defining point the one closes to the startpoint.
    c_index = geo_points_on_line(startpoint_in_index, c_index, w_index)

    # Correct the startpoint
    distances = np.linalg.norm(points_index - np.array(startpoint_in_index), axis=1)
    furthest_point_idx = np.argmax(distances)
    furthest_point_index = geo.point(points_index[furthest_point_idx])
    if geo.vector(furthest_point_index - startpoint_in_index).dot(w_index) < 0:
        w_index = -w_index

    furthest_point_on_line = geo_points_on_line(furthest_point_index, c_index, w_index)
    # plt.imshow(rescaled_depth_map)
    # plt.plot(furthest_point_on_line.x, furthest_point_on_line.y, "b.")
    # plt.colorbar()

    # Only the points along the line
    # for t in np.arange(0, (furthest_point_on_line - c_index).norm(), 1):
    #     p_index = c_index + t * w_index
    #     j, i = p_index
    #     i = int(i)
    #     j = int(j)

    distances_to_line = distance_to_line(points_index, np.array(c_index), np.array(w_index))

    # all the points along in the heatmap area
    points = []
    values = []
    distances_along_line = []
    furthest_point = None
    max_distance = 0
    for idx, (i, j) in enumerate(zip(rr, cc)):
        if distances_to_line[idx] > 1:
            continue
        p_index = geo.point(j, i)

        if not (0 <= i < rescaled_depth_map.shape[0]) or not (0 <= j < rescaled_depth_map.shape[1]):
            continue
        d = rescaled_depth_map[i, j]
        point = camera_point_from_index(
            p_index,
            d,
            camera3d_from_world=camera3d_from_world,
            index_from_world=index_from_world,
        )
        values.append(heatmap[i, j])
        points.append(point)
        distances_along_line.append(distances[idx])

    if not points:
        return None, None

    values = np.array(values)
    distances_along_line = np.array(distances_along_line)

    if not points:
        return None, None

    max_distance = np.max(distances_along_line)
    # indexing = (distances_along_line < 130) & (distances_along_line > 90)
    indexing = (distances_along_line < max_distance * 0.95) & (
        distances_along_line > max_distance * 0.8
    )
    if not indexing.any():
        return None, None
    far_points = np.array(points)[indexing]
    far_values = values[indexing]
    far_values = (far_values - np.min(far_values)) / (np.max(far_values) - np.min(far_values))
    d = geo.point(np.sum(far_values[:, None] * far_points, axis=0) / np.sum(far_values))

    c = startpoint_in_camera3d
    w = geo.vector(d - c)
    log.debug(f"corridor fit distance: {w.norm()}")
    w = w.hat()

    # c, w = fit_line(points)

    # # Again, fix the direction of w to be in the same direction
    # if geo.vector(furthest_point - c).dot(w) < 0:
    #     w = -w

    if gt_startpoint_in_camera3d is not None and gt_endpoint_in_camera3d is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ps = np.array(points)
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=values, cmap="viridis", marker=",")
        # ax.scatter(
        #     far_points[:, 0],
        #     far_points[:, 1],
        #     far_points[:, 2],
        #     c="k",
        #     marker=".",
        # )
        ax.scatter(
            startpoint_in_camera3d.x,
            startpoint_in_camera3d.y,
            startpoint_in_camera3d.z,
            c="b",
            marker="o",
        )
        ax.scatter(
            gt_startpoint_in_camera3d.x,
            gt_startpoint_in_camera3d.y,
            gt_startpoint_in_camera3d.z,
            c="g",
            marker="o",
        )
        ax.scatter(
            gt_endpoint_in_camera3d.x,
            gt_endpoint_in_camera3d.y,
            gt_endpoint_in_camera3d.z,
            c="g",
            marker="o",
        )
        d = c + (gt_endpoint_in_camera3d - gt_startpoint_in_camera3d).norm() * w
        ax.scatter(d.x, d.y, d.z, c="r", marker="o")

        if mpl.get_backend() == "agg":
            global img_idx
            plt.savefig(f"{img_idx:04d}_corridor_points.png", dpi=200)
            img_idx += 1
        else:
            plt.show()
        plt.close()

    # Find the point in the corridor that is closest to the predicted startpoint.
    # c = point_on_line(startpoint_in_camera3d, c, w)
    return c, w


def refine_corridor(
    heatmap: np.ndarray, startpoint: geo.Point2D
) -> Tuple[Optional[geo.Point2D], Optional[geo.Vector2D]]:
    """
    Find the corridor that is closest to the startpoint.
    """
    # Find the closest point in the heatmap
    threshold = max(np.quantile(heatmap, 0.98), get_heatmap_threshold(heatmap, fraction=0.45))
    heatmap_binary = (heatmap >= threshold).astype(np.uint8)
    if not np.any(heatmap_binary):
        log.debug(f"binary heatmap empty")
        return None, None

    # Fit a line weighted by the heatmap
    rr, cc = np.where(heatmap_binary)

    # column, row
    points_index = np.stack([cc, rr], axis=1)

    c_index, w_index = fit_line(points_index)

    return c_index, w_index


def locate_corridor(
    startpoint_heatmap: np.ndarray, corridor_heatmap: np.ndarray
) -> Tuple[Optional[geo.Point2D], Optional[geo.Vector2D]]:
    # Get the 2D startpoint in the image
    landmark = detect_landmark(startpoint_heatmap)
    if landmark is None:
        return None, None
    else:
        pred_startpoint_in_index = geo.point(landmark[1], landmark[0])
        return refine_corridor(corridor_heatmap, pred_startpoint_in_index)


img_idx = 0

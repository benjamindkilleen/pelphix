#!/usr/bin/env python3
"""Contains generic util functions needed for your project."""
import json
import logging
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from deepdrr import geo
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

from . import mesh_utils

log = logging.getLogger(__name__)


def doublewrap(f):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


def jsonable(obj: Any):
    """Convert obj to a JSON-ready container or object.

    Args:
        obj ([type]):
    """
    if obj is None:
        return "null"
    elif isinstance(obj, (str, float, int, complex)):
        return obj
    elif isinstance(obj, Path):
        return str(obj.resolve())
    elif hasattr(obj, "get_config") and callable(obj.get_config):
        return jsonable(obj.get_config())
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map(jsonable, obj))
    elif isinstance(obj, dict):
        return dict(jsonable(list(obj.items())))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__array__"):
        return np.array(obj).tolist()
    else:
        raise ValueError(f"Unknown type for JSON: {type(obj)}")


def save_json(path: str, obj: Any, indent: int = 4, sort_keys: bool = True):
    obj = jsonable(obj)
    with open(path, "w") as file:
        json.dump(obj, file, indent=indent, sort_keys=sort_keys)


def load_json(path: str):
    with open(path, "r") as file:
        out = json.load(file)
    return out


def split_sizes(size: int, split: np.ndarray) -> np.ndarray:
    """Get a non-random split into a dataset with `size` elements.

    The minimum size for each split is 1. If size is not large enough to accommodate this, then an
    error is raised.

    Returns:
        np.ndarray: The size of each split, so that the `i`th section corresponds to
            `data[split[i]:split[i+1]]`.

    """
    split = np.array(split)
    assert np.all(split >= 0)
    assert np.sum(split) == 1, f"invalid split: {split}"
    assert size >= np.sum(split > 0)

    sizes = np.floor(size * split)
    for i in range(sizes.shape[0]):
        if sizes[i] > 0 or split[i] == 0:
            continue

        idx = np.argmax(split)
        sizes[i] += 1
        sizes[idx] -= 1

    if np.sum(sizes) != size:
        idx = np.argmax(sizes)
        sizes[idx] += size - np.sum(sizes)

    assert np.sum(sizes) == size, f"split sizes {sizes.tolist()} does not sum to {size}"
    sizes = sizes.astype(np.int64)
    return sizes


def split_indices(size: int, split: np.ndarray) -> np.ndarray:
    """Get start indices of a non-random split into a dataset with `size` elements.

    The minimum size for each split is 1. If size is not large enough to accommodate this, then an
    error is raised.

    Returns:
        np.ndarray: The start index of each split, so that the `i`th section corresponds to
            `data[split[i]:split[i+1]]`. Includes the trailing index for convenience.

    """
    sizes = split_sizes(size, split)
    indices = [0]
    for s in sizes:
        indices.append(indices[-1] + s)

    assert indices[-1] == size
    return np.array(indices)


def split_mapping(size: int, split: np.ndarray) -> np.ndarray:
    """Get the (size,) array mapping each element in a dataset to its partition.

    Args:
        size (int): The size of the dataset.
        split (np.ndarray): The split fractions. Must sum to 1.

    Returns:
        np.ndarray: The (size,) array mapping each element in a dataset to its partition.
    """
    start_indices = split_indices(size, split)
    mapping = np.zeros(size, dtype=np.int64)
    for i, start_idx in enumerate(start_indices[:-1]):
        mapping[start_idx : start_indices[i + 1]] = i
    return mapping


def one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode a mask.

    Args:
        mask (np.ndarray): The mask to encode.
        num_classes (int): The number of classes in the one-hot encoding.

    Returns:
        np.ndarray: The one-hot encoding.
    """
    return np.eye(num_classes, dtype=np.int32)[mask]


def find_pattern(x: ArrayLike, pattern: ArrayLike) -> List[int]:
    """Find the occurrences of `pattern` in `x`.

    This could be abstracted to multiple dimensions.

    Args:
        x (ArrayLike): A one-dimensional array.
        pattern (ArrayLike): The pattern to search for.

    Returns:
        List[int]: A list of the indices in `x` where `pattern` starts.
    """
    # TODO: think if there's a better algorithm than just a sliding window.
    indices = []
    x = np.array(x)
    p = np.array(pattern)
    assert p.shape[0] <= x.shape[0]
    for i in range(x.shape[0] - p.shape[0] + 1):
        if np.all(x[i : i + p.shape[0]] == p):
            indices.append(i)
    return indices


def get_drawings_dir() -> Path:
    """Get the location of the data directory, which is not tracked in the repo.

    Download the data if it is not present.

    """
    data_dir = (Path(__file__).parent / ".." / ".." / ".." / "data" / "tools").resolve()
    data_dir.mkdir(exist_ok=True, parents=True)
    return data_dir


def get_cache_dir() -> Path:
    """Get the path to the generic cache dir."""
    cache_dir = (Path(__file__).parent / ".." / ".." / ".." / "cache").resolve()
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def voxelize_tools(use_cached: bool = True, **kwargs) -> Path:
    """Voxelize the toolkit with the given density."""
    stl_dir = get_drawings_dir() / "tools"
    nifti_dir = get_cache_dir() / "nifti"
    mesh_utils.voxelize_dir(stl_dir, nifti_dir, use_cached=use_cached)
    return nifti_dir


def get_stl(d: Path) -> List[Path]:
    return list(d.glob("*.STL")) + list(d.glob("*.stl"))


def to_numpy(x: Any) -> Union[np.ndarray, dict]:
    """Get x as a numpy array, copying and detaching if need by.

    Args:
        x (Union[np.ndarray, torch.Tensor]): [description]

    Raises:
        TypeError: [description]

    Returns:
        torch.Tensor: [description]
    """
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu().numpy()
        if out.dtype in [np.float16, np.float32, np.float64]:
            out = out.astype(np.float32)
        return out
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, tuple):
        return (to_numpy(x_i) for x_i in x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        return x


def get_unbatched(x: Any):
    y = to_numpy(x)
    if isinstance(y, dict):
        n = len(list(y.values())[0])
        log.debug(f"unbatched {n} items")
        return [{k: v[i] for k, v in y.items()} for i in range(n)]
    else:
        return y


def ensure_cdim(x: np.ndarray, c: int = 1) -> np.ndarray:
    if x.ndim == 2:
        x = x[np.newaxis]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"bad input ndim: {x.shape}")

    if x.shape[0] < c:
        return np.concatenate([x] * c, axis=0)
    elif x.shape[0] == c:
        return x
    elif x.shape[2] == c:
        return x.transpose(2, 0, 1)
    else:
        raise ValueError


def combine_heatmap(
    image: Union[np.ndarray, torch.Tensor],
    heatmap: Union[np.ndarray, torch.Tensor],
    channel=0,
    normalize=True,
) -> np.ndarray:
    """Visualize a heatmap on an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): 2D float image, [H, W], or [3, H, W]
        heatmap (Union[np.ndarray, torch.Tensor]): 2D float heatmap, [H, W], or [C, H, W] array of heatmaps.
        channel (int, optional): Which channel to use for the heatmap. For an RGB image, channel 0 would render the heatmap in red.. Defaults to 0.
        normalize (bool, optional): Whether to normalize the heatmap. This can lead to all-red images if no landmark was detected. Defaults to True.

    Returns:
        np.ndarray: A [H,W,3] numpy image.
    """
    image_arr = ensure_cdim(to_numpy(image), c=3)
    heatmap_arr = ensure_cdim(to_numpy(heatmap), c=1)

    seg = False
    if heatmap_arr.dtype == bool:
        heatmap_arr = heatmap_arr.astype(np.float32)
        seg = True

    _, h, w = heatmap_arr.shape
    heat_sum = np.zeros((h, w), dtype=np.float32)
    for heat in heatmap_arr:
        heat_min = heat.min()
        heat_max = 4 if seg else heat.max()
        heat_min_minus_max = heat_max - heat_min
        heat = heat - heat_min
        if heat_min_minus_max > 1.0e-3:
            heat /= heat_min_minus_max

        heat_sum += heat

    for c in range(3):
        image_arr[c] = ((1 - heat_sum) * image_arr[c]) + (heat_sum if c == channel else 0)

    return image_arr.transpose(1, 2, 0)


def imshow_save(path: str, image: np.ndarray, interpolation="none", **kwargs):
    """Imshow the image and save the resulting figure to the path.

    Args:
        path (str): the path.
        image (np.ndarray): The image.
    """
    image = to_numpy(image)
    plt.figure()
    plt.imshow(image, cmap="viridis", interpolation=interpolation, **kwargs)
    plt.axis("off")
    plt.colorbar()
    plt.savefig(path)
    plt.close()


def quadratic_formula(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
    d = b * b - 4 * a * c

    if d < 0:
        log.debug(f"couldn't solve because d negative: {d}")
        return None, None

    # find two solutions
    x0 = (-b - np.sqrt(d)) / (2 * a)
    x1 = (-b + np.sqrt(d)) / (2 * a)
    return x0, x1


def write_metrics(path: str, metrics: Union[List[float], np.ndarray]):
    with open(path, "a") as file:
        file.writelines([f"{m}\n" for m in metrics])


def boolean_mask(x: List[int], n: int) -> np.ndarray:
    """Get a boolean mask from a list of indices."""
    y = np.zeros(n, dtype=bool)
    y[x] = True
    return y


def load_line_markup(path: Path) -> Tuple[geo.Point3D, geo.Point3D]:
    path = Path(path)
    markup = load_json(path)
    control_points = markup["markups"][0]["controlPoints"]
    points = [geo.point(cp["position"]) for cp in control_points]
    return points[0], points[1]


__all__ = [
    "get_drawings_dir",
    "get_cache_dir",
]

from datetime import datetime
from typing import List
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns
from deepdrr import geo
from deepdrr.utils.image_utils import ensure_cdim, as_uint8, as_float32


log = logging.getLogger(__name__)


# TODO: redo each of these to allow for passing in a color palette and labels, as well as a scale
# factor.


def draw_circles(
    image: np.ndarray,
    circles: np.ndarray,
    color: List[int] = [255, 0, 0],
    thickness: int = 2,
    radius: Optional[int] = None,
) -> np.ndarray:
    """Draw circles on an image.

    Args:
        image (np.ndarray): the image to draw on.
        circles (np.ndarray): the circles to draw. [N, 3] array of [x, y, r] coordinates.

    """
    color = np.array(color)
    if np.any(color < 1):
        color = color * 255
    color = color.astype(int)[:3].tolist()

    circles = np.array(circles)
    image = ensure_cdim(as_uint8(image)).copy()
    for circle in circles:
        if circles.shape[1] == 3:
            x, y, r = circle
        elif circles.shape[1] == 2:
            x, y = circle
            r = radius if radius is not None else 15
        else:
            raise ValueError(f"bad circles shape: {circles.shape}")
        if radius is not None:
            r = radius
        image = cv2.circle(image, (int(x), int(y)), int(r), color, thickness)
    return image


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.3,
    palette: str = "Spectral",
    threshold: float = 0.5,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Draw contours of masks on an image.

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [H, W, num_masks] array of masks.
    """

    image = as_float32(image)
    masks = masks.transpose(2, 0, 1)
    colors = np.array(sns.color_palette(palette, masks.shape[0]))
    if seed is not None:
        np.random.seed(seed)
    colors = colors[np.random.permutation(colors.shape[0])]
    image *= 1 - alpha
    for i, mask in enumerate(masks):
        bool_mask = mask > threshold
        image[bool_mask] = colors[i] * alpha + image[bool_mask] * (1 - alpha)

        contours, _ = cv2.findContours(
            bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image = as_float32(
            cv2.drawContours(
                as_uint8(image), contours, -1, (255 * colors[i]).tolist(), 1
            )
        )
    return (image * 255).astype(np.uint8)

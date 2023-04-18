import cv2
import numpy as np
from deepdrr.utils import image_utils
import logging

log = logging.getLogger(__name__)


def detect_circles(
    image: np.ndarray,
    min_radius: int = 40,
    max_radius: int = 100,
    min_distance: int = 20,
    blur: int = 7,
) -> np.ndarray:
    """Detect circles in an image.

    Args:
        image: (H,W) image

    Returns:
        (N,3) array of (x,y,radius,votes) for each circle detected, sorted by the number of votes.
    """
    image = image_utils.as_uint8(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur > 0:
        image = cv2.medianBlur(image, blur)
    H, W = image.shape
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        1.5,
        min_distance,  # min distance between circles
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return np.zeros((0, 3))
    else:
        return circles[0]

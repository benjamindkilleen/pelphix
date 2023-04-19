import numpy as np
from typing import List


def segmentation_to_bbox(segmentation: List[List[int]]) -> List[int]:
    """Get the bounding box of a segmentation mask.

    Args:
        segmentation (List[List[int]]): The segmentation mask in the coco style.

    Returns:
        List[int]: The bounding box as [x_min, y_min, width, height]
    """
    segmentation = np.array(sum(segmentation, [])).reshape(-1, 2)
    x = segmentation[:, 0]
    y = segmentation[:, 1]
    x_min = np.min(x)
    y_min = np.min(y)
    width = np.max(x) - x_min
    height = np.max(y) - y_min
    return [int(x_min), int(y_min), int(width), int(height)]

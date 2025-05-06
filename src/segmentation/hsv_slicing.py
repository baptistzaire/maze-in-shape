"""
Implements image segmentation based on color range slicing in HSV space.
"""

import cv2
import numpy as np
from typing import Tuple, Union

from .base import BaseSegmenter

class HSVSegmenter(BaseSegmenter):
    """
    Segments an image by isolating pixels within a specific HSV color range.

    Uses OpenCV's `cvtColor` to convert to HSV and `inRange` to create the mask.
    """

    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image based on HSV color range.

        Args:
            image: The input image as a NumPy array (assumed BGR format).
            **params: Parameters for HSV slicing.
                Required:
                    lower_hsv (Union[Tuple[int, int, int], np.ndarray]): Lower
                        bound for HSV channels (H: 0-179, S: 0-255, V: 0-255).
                    upper_hsv (Union[Tuple[int, int, int], np.ndarray]): Upper
                        bound for HSV channels (H: 0-179, S: 0-255, V: 0-255).

        Returns:
            A binary mask (uint8, 0/255) where 255 indicates pixels within the
            specified HSV range.

        Raises:
            ValueError: If required parameters `lower_hsv` or `upper_hsv` are
                        missing or invalid.
            TypeError: If the input image is not a NumPy array or bounds are
                       not tuples/arrays.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in BGR format (3 channels).")

        lower_hsv = params.get('lower_hsv')
        upper_hsv = params.get('upper_hsv')

        if lower_hsv is None or upper_hsv is None:
            raise ValueError("Parameters 'lower_hsv' and 'upper_hsv' are required.")

        # Ensure bounds are NumPy arrays
        try:
            lower_bound = np.array(lower_hsv, dtype=np.uint8)
            upper_bound = np.array(upper_hsv, dtype=np.uint8)
        except Exception as e:
            raise TypeError(f"HSV bounds must be convertible to NumPy arrays: {e}")

        if lower_bound.shape != (3,) or upper_bound.shape != (3,):
            raise ValueError("HSV bounds must have 3 elements (H, S, V).")

        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask using inRange
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        return mask

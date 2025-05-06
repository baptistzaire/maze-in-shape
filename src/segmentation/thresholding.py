"""
Implements image segmentation using simple and adaptive thresholding methods.
"""

import cv2
import numpy as np
from typing import Optional, Literal

from .base import BaseSegmenter

class ThresholdSegmenter(BaseSegmenter):
    """
    Segments an image using OpenCV's thresholding functions.

    Supports both global thresholding (cv2.threshold) and adaptive
    thresholding (cv2.adaptiveThreshold). The choice depends on the
    parameters provided.
    """

    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image using thresholding.

        Args:
            image: The input image as a NumPy array (RGB or Grayscale).
            **params: Parameters to control thresholding.
                Required:
                    method: 'global' or 'adaptive'.
                For 'global':
                    threshold_value (int): Threshold value (0-255). If None,
                        Otsu's method is used (requires threshold_type to
                        include cv2.THRESH_OTSU). Default: 127.
                    max_value (int): Value assigned to pixels above the
                        threshold. Default: 255.
                    threshold_type (int): OpenCV thresholding type
                        (e.g., cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV).
                        Can be combined with cv2.THRESH_OTSU.
                        Default: cv2.THRESH_BINARY.
                For 'adaptive':
                    adaptive_method (int): Adaptive thresholding algorithm
                        (cv2.ADAPTIVE_THRESH_MEAN_C or
                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
                        Default: cv2.ADAPTIVE_THRESH_GAUSSIAN_C.
                    block_size (int): Size of the pixel neighborhood used to
                        calculate the threshold. Must be odd and > 1.
                        Default: 11.
                    C (float): Constant subtracted from the mean or weighted
                        mean. Default: 2.
                    max_value (int): Value assigned to pixels above the
                        threshold. Default: 255.
                    threshold_type (int): Must be cv2.THRESH_BINARY or
                        cv2.THRESH_BINARY_INV. Default: cv2.THRESH_BINARY.

        Returns:
            A binary mask (uint8, 0/255) with the same dimensions as the input.

        Raises:
            ValueError: If required parameters are missing or invalid.
            TypeError: If the input image is not a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        # Ensure image is grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image
        else:
            raise ValueError("Input image must be grayscale or BGR.")

        method = params.get('method')
        if method is None:
            raise ValueError("Parameter 'method' ('global' or 'adaptive') is required.")

        max_value = params.get('max_value', 255)
        threshold_type = params.get('threshold_type', cv2.THRESH_BINARY)

        if method == 'global':
            threshold_value = params.get('threshold_value', 127)

            # Handle Otsu's method
            if threshold_value is None:
                if not (threshold_type & cv2.THRESH_OTSU):
                     # Add THRESH_OTSU if threshold_value is None and it's not already set
                     threshold_type |= cv2.THRESH_OTSU
                # For Otsu, the input threshold value is ignored, so set to 0
                threshold_value = 0
            elif threshold_type & cv2.THRESH_OTSU:
                 # If Otsu is specified but threshold_value is also given,
                 # prefer Otsu and ignore the provided value.
                 threshold_value = 0 # Otsu ignores this value

            _, mask = cv2.threshold(
                gray_image,
                threshold_value,
                max_value,
                threshold_type
            )
            return mask

        elif method == 'adaptive':
            adaptive_method = params.get('adaptive_method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            block_size = params.get('block_size', 11)
            C = params.get('C', 2.0) # C is often float in examples

            if block_size <= 1 or block_size % 2 == 0:
                raise ValueError("block_size must be an odd integer > 1.")
            if threshold_type not in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
                 raise ValueError("For adaptive thresholding, threshold_type must be cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.")


            mask = cv2.adaptiveThreshold(
                gray_image,
                max_value,
                adaptive_method,
                threshold_type,
                block_size,
                float(C) # Ensure C is float for the function call
            )
            return mask

        else:
            raise ValueError(f"Unknown thresholding method: {method}. Use 'global' or 'adaptive'.")

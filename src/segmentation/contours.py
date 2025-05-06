"""
Implements image segmentation using Canny edge detection and contour filling.
"""

import cv2
import numpy as np
from typing import Optional

from .base import BaseSegmenter

class CannyContourSegmenter(BaseSegmenter):
    """
    Segments an image by detecting edges with Canny, finding the largest
    contour, and filling it to create a mask.
    """

    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image using Canny edges and contour filling.

        Args:
            image: The input image as a NumPy array (RGB or Grayscale).
            **params: Parameters for Canny edge detection.
                Required:
                    threshold1 (float): First threshold for the hysteresis
                                        procedure in Canny.
                    threshold2 (float): Second threshold for the hysteresis
                                        procedure in Canny.
                Optional:
                    apertureSize (int): Aperture size for the Sobel operator
                                        used internally by Canny. Default: 3.
                    L2gradient (bool): Flag indicating whether to use a more
                                       accurate L2 norm for gradient magnitude.
                                       Default: False.
                    blur_ksize (int): Kernel size for Gaussian blur applied
                                      before Canny. Must be odd. If 0 or None,
                                      no blur is applied. Default: 5.
                    contour_mode (int): Contour retrieval mode for
                                        findContours (e.g., cv2.RETR_EXTERNAL).
                                        Default: cv2.RETR_EXTERNAL.
                    contour_method (int): Contour approximation method
                                          (e.g., cv2.CHAIN_APPROX_SIMPLE).
                                          Default: cv2.CHAIN_APPROX_SIMPLE.

        Returns:
            A binary mask (uint8, 0/255) with the largest contour filled.
            Returns a black mask if no contours are found.

        Raises:
            ValueError: If required parameters `threshold1` or `threshold2`
                        are missing or if blur_ksize is even.
            TypeError: If the input image is not a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        threshold1 = params.get('threshold1')
        threshold2 = params.get('threshold2')
        if threshold1 is None or threshold2 is None:
            raise ValueError("Parameters 'threshold1' and 'threshold2' are required for Canny.")

        apertureSize = params.get('apertureSize', 3)
        L2gradient = params.get('L2gradient', False)
        blur_ksize = params.get('blur_ksize', 5)
        contour_mode = params.get('contour_mode', cv2.RETR_EXTERNAL)
        contour_method = params.get('contour_method', cv2.CHAIN_APPROX_SIMPLE)

        # Ensure image is grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image
        else:
            raise ValueError("Input image must be grayscale or BGR.")

        # Optional Pre-processing: Gaussian Blur
        processed_image = gray_image
        if blur_ksize is not None and blur_ksize > 0:
            if blur_ksize % 2 == 0:
                raise ValueError("blur_ksize must be an odd integer.")
            processed_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

        # Canny Edge Detection
        edges = cv2.Canny(
            processed_image,
            threshold1,
            threshold2,
            apertureSize=apertureSize,
            L2gradient=L2gradient
        )

        # Find Contours
        contours, _ = cv2.findContours(
            edges,
            contour_mode,
            contour_method
        )

        # Create an empty mask
        mask = np.zeros_like(gray_image)

        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the largest contour filled on the mask
            cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        return mask

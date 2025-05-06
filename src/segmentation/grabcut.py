"""
Implements image segmentation using the GrabCut algorithm.
"""

import cv2
import numpy as np
from typing import Tuple

from .base import BaseSegmenter

class GrabCutSegmenter(BaseSegmenter):
    """
    Segments an image using the GrabCut algorithm (cv2.grabCut).

    Requires an initial bounding box (ROI) to guide the segmentation process.
    """

    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image using GrabCut.

        Args:
            image: The input image as a NumPy array (assumed BGR format).
            **params: Parameters for GrabCut segmentation.
                Required:
                    roi_rect (Tuple[int, int, int, int]): The region of interest
                        rectangle (x, y, width, height) containing the foreground
                        object. GrabCut uses this to initialize the model.
                    iterations (int): Number of iterations the GrabCut algorithm
                                      should run.
                Optional:
                    initial_mask (np.ndarray): An optional initial mask (uint8)
                        of the same size as the image. If provided, `roi_rect`
                        is ignored, and the mask is used for initialization
                        (cv2.GC_INIT_WITH_MASK). Values should be cv2.GC_*.
                        If None, initialization uses `roi_rect`
                        (cv2.GC_INIT_WITH_RECT). Default: None.

        Returns:
            A binary mask (uint8, 0/255) where 255 indicates the segmented
            foreground.

        Raises:
            ValueError: If required parameters are missing or invalid, or if
                        the ROI is outside image bounds.
            TypeError: If the input image or parameters have incorrect types.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in BGR format (3 channels).")

        iterations = params.get('iterations')
        if iterations is None:
            raise ValueError("Parameter 'iterations' is required for GrabCut.")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("'iterations' must be a positive integer.")

        initial_mask_param = params.get('initial_mask')
        roi_rect_param = params.get('roi_rect')

        # Initialize mask and determine mode
        mask = np.zeros(image.shape[:2], np.uint8) # Mask required by grabCut
        rect = None
        mode = cv2.GC_INIT_WITH_RECT

        if initial_mask_param is not None:
            if not isinstance(initial_mask_param, np.ndarray) or \
               initial_mask_param.shape != image.shape[:2] or \
               initial_mask_param.dtype != np.uint8:
                raise TypeError("'initial_mask' must be a uint8 NumPy array with the same height and width as the image.")
            mask = initial_mask_param.copy() # Use provided mask
            mode = cv2.GC_INIT_WITH_MASK
            # roi_rect is ignored if initial_mask is provided
        elif roi_rect_param is not None:
            if not isinstance(roi_rect_param, tuple) or len(roi_rect_param) != 4 or \
               not all(isinstance(v, int) for v in roi_rect_param):
                raise TypeError("'roi_rect' must be a tuple of 4 integers (x, y, w, h).")

            rect = roi_rect_param
            x, y, w, h = rect
            img_h, img_w = image.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or (x + w) > img_w or (y + h) > img_h:
                raise ValueError(f"ROI rectangle {rect} is outside image bounds ({img_w}x{img_h}).")
            mode = cv2.GC_INIT_WITH_RECT
            # Mask is initialized by grabCut using the rect
        else:
            raise ValueError("Either 'roi_rect' or 'initial_mask' must be provided for GrabCut.")


        # Allocate memory for background and foreground models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Run GrabCut
        try:
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, mode=mode)
        except cv2.error as e:
            # Catch potential OpenCV errors during grabCut execution
            raise RuntimeError(f"OpenCV error during grabCut: {e}")


        # Create the final binary mask
        # Values GC_FGD (1) and GC_PR_FGD (3) are considered foreground
        output_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        return output_mask

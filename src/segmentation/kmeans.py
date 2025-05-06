"""
Implements image segmentation using K-Means color clustering.
"""

import cv2
import numpy as np
from typing import Optional, List, Union

from .base import BaseSegmenter

class KMeansSegmenter(BaseSegmenter):
    """
    Segments an image by clustering pixel colors using the K-Means algorithm.

    Requires specifying which cluster index corresponds to the foreground.
    """

    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image using K-Means color clustering.

        Args:
            image: The input image as a NumPy array (assumed BGR format).
            **params: Parameters for K-Means segmentation.
                Required:
                    num_clusters (int): The number of clusters (K) to form.
                    foreground_cluster_indices (Union[int, List[int]]): The index
                        (or list of indices) of the cluster(s) representing the
                        foreground. Cluster indices range from 0 to K-1.
                Optional:
                    attempts (int): Number of times the algorithm is executed
                                    with different initial centroids. The best
                                    compactness is returned. Default: 10.
                    max_iter (int): Maximum number of iterations for K-Means.
                                    Default: 100.
                    epsilon (float): Required accuracy (epsilon) for K-Means.
                                     Default: 1.0.
                    criteria_flags (int): Flags for termination criteria
                                          (cv2.TERM_CRITERIA_EPS |
                                           cv2.TERM_CRITERIA_MAX_ITER).
                                          Default uses both EPS and MAX_ITER.

        Returns:
            A binary mask (uint8, 0/255) where 255 indicates pixels belonging
            to the specified foreground cluster(s).

        Raises:
            ValueError: If required parameters are missing or invalid.
            TypeError: If the input image is not a NumPy array or parameters
                       have incorrect types.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in BGR format (3 channels).")

        num_clusters = params.get('num_clusters')
        foreground_indices = params.get('foreground_cluster_indices')

        if num_clusters is None:
            raise ValueError("Parameter 'num_clusters' (K) is required.")
        if not isinstance(num_clusters, int) or num_clusters < 1:
            raise ValueError("'num_clusters' must be a positive integer.")

        if foreground_indices is None:
            raise ValueError("Parameter 'foreground_cluster_indices' is required.")

        # Ensure foreground_indices is a list of integers
        if isinstance(foreground_indices, int):
            foreground_indices = [foreground_indices]
        elif not isinstance(foreground_indices, list) or not all(isinstance(i, int) for i in foreground_indices):
             raise TypeError("'foreground_cluster_indices' must be an integer or a list of integers.")

        if any(idx < 0 or idx >= num_clusters for idx in foreground_indices):
            raise ValueError(f"Foreground indices must be between 0 and K-1 (K={num_clusters}).")


        attempts = params.get('attempts', 10)
        max_iter = params.get('max_iter', 100)
        epsilon = params.get('epsilon', 1.0)
        default_criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER
        criteria_flags = params.get('criteria_flags', default_criteria)
        criteria = (criteria_flags, max_iter, epsilon)

        # Reshape image into a list of pixels (N, 3) and convert to float32
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Perform K-Means clustering
        compactness, labels, centers = cv2.kmeans(
            pixel_values,
            num_clusters,
            None,  # bestLabels argument (None means use output labels)
            criteria,
            attempts,
            cv2.KMEANS_RANDOM_CENTERS # Or cv2.KMEANS_PP_CENTERS
        )

        # Create mask based on foreground cluster indices
        labels = labels.flatten() # Ensure labels is 1D
        mask = np.zeros(labels.shape, dtype=np.uint8)
        for fg_idx in foreground_indices:
            mask[labels == fg_idx] = 255

        # Reshape mask back to original image dimensions (H, W)
        mask = mask.reshape(image.shape[0], image.shape[1])

        return mask

"""
Defines the abstract base class for all segmentation algorithms.
"""

import abc
import numpy as np

class BaseSegmenter(abc.ABC):
    """
    Abstract base class for image segmentation methods.

    All segmentation algorithms should inherit from this class and implement
    the `segment` method.
    """

    @abc.abstractmethod
    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image to produce a binary mask.

        Args:
            image: The input image as a NumPy array (e.g., loaded by OpenCV).
                     Expected format might vary depending on the implementation
                     (e.g., RGB, Grayscale).
            **params: Additional parameters specific to the segmentation method.

        Returns:
            A binary mask as a NumPy array (uint8, values 0 or 255), where 255
            indicates the segmented foreground (subject) and 0 indicates the
            background. The mask should have the same height and width as the
            input image.
        """
        pass

    def __call__(self, image: np.ndarray, **params) -> np.ndarray:
        """Allows calling the segmenter instance directly."""
        return self.segment(image, **params)

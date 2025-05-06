"""
Segmentation Module Initializer and Factory.

Provides a factory function `get_segmenter` to instantiate segmentation
objects based on a string identifier.
"""

from typing import Dict, Type, Any

from .base import BaseSegmenter
from .thresholding import ThresholdSegmenter
from .hsv_slicing import HSVSegmenter
from .contours import CannyContourSegmenter
from .kmeans import KMeansSegmenter
from .grabcut import GrabCutSegmenter
from .rembg_wrapper import RembgSegmenter
from .deep_learning import DeepLearningSegmenter

# Dictionary mapping segmentation method names to their respective classes
SEGMENTER_MAP: Dict[str, Type[BaseSegmenter]] = {
    "threshold": ThresholdSegmenter,
    "hsv": HSVSegmenter,
    "canny": CannyContourSegmenter,
    "kmeans": KMeansSegmenter,
    "grabcut": GrabCutSegmenter,
    "rembg": RembgSegmenter,
    "deep_learning": DeepLearningSegmenter,
    # Add aliases if desired, e.g., "deeplab": DeepLearningSegmenter
}

def get_segmenter(name: str, **init_params: Any) -> BaseSegmenter:
    """
    Factory function to create and return a segmenter instance.

    Args:
        name: The string identifier for the desired segmentation method
              (e.g., "threshold", "rembg", "deep_learning"). Must be a key
              in SEGMENTER_MAP.
        **init_params: Keyword arguments to pass to the segmenter's
                       constructor (__init__). This is particularly relevant
                       for segmenters like DeepLearningSegmenter that require
                       initialization parameters (e.g., model_name, device).

    Returns:
        An initialized instance of the requested BaseSegmenter subclass.

    Raises:
        ValueError: If the provided `name` is not a valid key in
                    SEGMENTER_MAP.
        ImportError: If a required library for a specific segmenter
                     (e.g., rembg, torch) is not installed when trying
                     to instantiate it.
    """
    segmenter_class = SEGMENTER_MAP.get(name.lower()) # Use lower case for robustness

    if segmenter_class is None:
        raise ValueError(
            f"Unknown segmenter name: '{name}'. "
            f"Available segmenters: {list(SEGMENTER_MAP.keys())}"
        )

    try:
        # Instantiate the class, passing any provided init_params
        # If a class doesn't accept params, they will be ignored if **init_params is empty
        # or raise TypeError if unexpected params are passed.
        # DeepLearningSegmenter specifically uses init_params.
        segmenter_instance = segmenter_class(**init_params)
        return segmenter_instance
    except ImportError as e:
        # Re-raise import errors clearly indicating the segmenter and missing library
        raise ImportError(f"Failed to initialize segmenter '{name}'. Missing required library: {e}")
    except Exception as e:
        # Catch other potential instantiation errors
        raise RuntimeError(f"Error initializing segmenter '{name}' with params {init_params}: {e}")

# Expose segmenter classes and factory function for easier import
__all__ = [
    "BaseSegmenter",
    "ThresholdSegmenter",
    "HSVSegmenter",
    "CannyContourSegmenter",
    "KMeansSegmenter",
    "GrabCutSegmenter",
    "RembgSegmenter",
    "DeepLearningSegmenter",
    "get_segmenter",
    "SEGMENTER_MAP",
]

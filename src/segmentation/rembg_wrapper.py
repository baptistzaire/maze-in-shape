"""
Implements image segmentation using the 'rembg' library for background removal.

Note: Requires the 'rembg' library to be installed (pip install rembg).
Be aware of the licensing implications of rembg and its models (GPL/LGPL).
"""

import cv2 # Added import
import numpy as np
from typing import Optional

from .base import BaseSegmenter

# Attempt to import rembg and handle potential ImportError
try:
    from rembg import remove as rembg_remove
    from PIL import Image
    import io
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    # Define dummy types/functions if rembg is not installed to avoid runtime errors
    # on class definition, but raise error during instantiation or usage.
    class Image: pass # Dummy class
    def rembg_remove(*args, **kwargs): pass # Dummy function


class RembgSegmenter(BaseSegmenter):
    """
    Segments an image by removing the background using the 'rembg' library.

    This acts as a wrapper around `rembg.remove`, utilizing models like U2-Net.
    """

    def __init__(self):
        """Initializes the segmenter and checks for rembg availability."""
        if not REMBG_AVAILABLE:
            raise ImportError("The 'rembg' library is not installed. Please install it using 'pip install rembg' to use RembgSegmenter.")

    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image using rembg.remove.

        Args:
            image: The input image as a NumPy array (assumed BGR format by OpenCV,
                   but rembg expects RGB, so conversion is handled).
            **params: Parameters for rembg.remove.
                Optional:
                    model_name (str): Name of the rembg model to use
                                      (e.g., 'u2net', 'u2netp', 'silueta').
                                      Default: 'u2net'.
                    alpha_matting (bool): Enable alpha matting. Default: False.
                    alpha_matting_foreground_threshold (int): Threshold for
                        foreground detection in alpha matting. Default: 240.
                    alpha_matting_background_threshold (int): Threshold for
                        background detection in alpha matting. Default: 10.
                    alpha_matting_erode_size (int): Erode size for alpha matting.
                                                    Default: 10.
                    # Add other relevant rembg parameters as needed

        Returns:
            A binary mask (uint8, 0/255) where 255 indicates the foreground.

        Raises:
            RuntimeError: If rembg processing fails.
            TypeError: If the input image is not a NumPy array.
            ImportError: If 'rembg' library is not installed.
        """
        if not REMBG_AVAILABLE:
             # This check is redundant if __init__ succeeded, but good practice
            raise ImportError("RembgSegmenter requires the 'rembg' library.")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            # rembg might handle grayscale, but documentation implies RGB/RGBA
            raise ValueError("Input image must be in BGR format (3 channels).")

        # Get parameters for rembg
        model_name = params.get('model_name', 'u2net')
        alpha_matting = params.get('alpha_matting', False)
        fg_threshold = params.get('alpha_matting_foreground_threshold', 240)
        bg_threshold = params.get('alpha_matting_background_threshold', 10)
        erode_size = params.get('alpha_matting_erode_size', 10)

        # Convert BGR (OpenCV default) to RGB (PIL/rembg default)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(rgb_image)

        try:
            # Use rembg.remove - it returns a PIL Image in RGBA format
            output_pil_image = rembg_remove(
                pil_image,
                session=None, # Use default session management
                only_mask=False, # Get RGBA image to extract mask later
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
                alpha_matting_erode_size=erode_size,
                # Pass model name if using non-default models (rembg handles this internally now)
                # model=model_name # Parameter name might vary or be implicit
            )

            # Convert output PIL image back to NumPy array
            output_np_image = np.array(output_pil_image)

            # Extract the alpha channel as the mask
            if output_np_image.shape[2] == 4:
                alpha_mask = output_np_image[:, :, 3]
                # Threshold the alpha mask to get a binary mask (0 or 255)
                # Pixels with alpha > 0 are considered foreground
                binary_mask = np.where(alpha_mask > 0, 255, 0).astype(np.uint8)
            else:
                # Should not happen if only_mask=False, but handle defensively
                raise RuntimeError("rembg did not return an RGBA image.")

            return binary_mask

        except Exception as e:
            # Catch potential errors during rembg processing
            raise RuntimeError(f"Error during rembg processing: {e}")

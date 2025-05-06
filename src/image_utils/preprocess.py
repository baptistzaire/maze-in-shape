import cv2
import numpy as np
from typing import Optional, Tuple, Union

def resize_image(
    image: np.ndarray,
    max_dimension: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """
    Resizes an image based on a maximum dimension, target size, or scale factor.

    Only one resizing method (max_dimension, target_size, scale_factor) should be provided.
    If more than one is provided, precedence is: target_size > max_dimension > scale_factor.
    If none are provided, the original image is returned.

    Args:
        image: The input image as a NumPy array (H, W, C).
        max_dimension: The maximum allowed dimension (width or height).
                       Aspect ratio is preserved.
        target_size: A tuple (width, height) for the desired output size.
                     Aspect ratio might not be preserved.
        scale_factor: A factor by which to scale the image dimensions.
                      Aspect ratio is preserved.
        interpolation: OpenCV interpolation method flag (e.g., cv2.INTER_LINEAR,
                       cv2.INTER_AREA). Defaults to cv2.INTER_AREA, which is
                       generally good for shrinking. Use cv2.INTER_LINEAR or
                       cv2.INTER_CUBIC for enlarging.

    Returns:
        The resized image as a NumPy array.

    Raises:
        ValueError: If the input image is invalid or resizing parameters are incorrect.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    h, w = image.shape[:2]
    new_w, new_h = w, h

    if target_size:
        # Priority 1: Resize to specific target_size (width, height)
        if not (isinstance(target_size, tuple) and len(target_size) == 2 and
                all(isinstance(d, int) and d > 0 for d in target_size)):
            raise ValueError("target_size must be a tuple of two positive integers (width, height).")
        new_w, new_h = target_size
        print(f"Resizing to specific dimensions: ({new_w}, {new_h})")

    elif max_dimension:
        # Priority 2: Resize based on max_dimension, preserving aspect ratio
        if not (isinstance(max_dimension, int) and max_dimension > 0):
             raise ValueError("max_dimension must be a positive integer.")
        if w > max_dimension or h > max_dimension:
            if w > h:
                ratio = max_dimension / w
                new_w = max_dimension
                new_h = int(h * ratio)
            else:
                ratio = max_dimension / h
                new_h = max_dimension
                new_w = int(w * ratio)
            print(f"Resizing to max dimension {max_dimension}, new size: ({new_w}, {new_h})")
        else:
            # Image is already smaller than max_dimension
            print(f"Image dimensions ({w}, {h}) are within max_dimension {max_dimension}. No resize needed.")
            return image # Return original if already smaller

    elif scale_factor is not None: # Check if scale_factor was provided
        # Priority 3: Resize by scale_factor, preserving aspect ratio
        if not (isinstance(scale_factor, (int, float)) and scale_factor > 0):
             # Now this check correctly handles 0 or negative values
             raise ValueError("scale_factor must be a positive number.")
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        # Ensure dimensions are at least 1 pixel
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        print(f"Resizing by scale factor {scale_factor}, new size: ({new_w}, {new_h})")

    else:
        # No resizing parameters provided
        print("No resizing parameters provided. Returning original image.")
        return image

    # Perform the resize operation if dimensions changed
    if new_w != w or new_h != h:
        # Choose interpolation based on scaling direction if not explicitly set to AREA
        # (INTER_AREA is generally best for shrinking, LINEAR/CUBIC for enlarging)
        effective_interpolation = interpolation
        if interpolation == cv2.INTER_AREA and (new_w > w or new_h > h):
             # If default INTER_AREA is used but we are enlarging, switch to LINEAR
             effective_interpolation = cv2.INTER_LINEAR
             print(f"Switching interpolation to cv2.INTER_LINEAR for enlargement.")

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=effective_interpolation)
        return resized_image
    else:
        # Dimensions didn't change (e.g., max_dimension was larger than image)
        return image


def convert_color_space(image: np.ndarray, target_space: str) -> np.ndarray:
    """
    Converts an image to the specified target color space.

    Assumes the input image is in RGB format.

    Args:
        image: The input image as a NumPy array (H, W, C or H, W).
        target_space: The target color space ('GRAY', 'HSV', 'RGB'). Case-insensitive.

    Returns:
        The image converted to the target color space.

    Raises:
        ValueError: If the target color space is unsupported or the input image
                    has an unexpected number of channels for the conversion.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    target_space = target_space.upper()
    current_channels = image.shape[2] if image.ndim == 3 else 1

    if target_space == 'RGB':
        if current_channels == 3:
            print("Image is already in RGB format.")
            return image
        elif current_channels == 1:
            print("Converting Grayscale to RGB.")
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
             raise ValueError(f"Unsupported number of channels ({current_channels}) for conversion to RGB.")

    elif target_space == 'GRAY':
        if current_channels == 1:
            print("Image is already in Grayscale format.")
            return image
        elif current_channels == 3:
            print("Converting RGB to Grayscale.")
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # e.g., RGBA input not directly handled here, assume RGB input
            raise ValueError("Input image must be RGB (3 channels) to convert to Grayscale.")

    elif target_space == 'HSV':
        if current_channels == 3:
            print("Converting RGB to HSV.")
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError("Input image must be RGB (3 channels) to convert to HSV.")

    else:
        raise ValueError(f"Unsupported target color space: {target_space}. Supported: 'RGB', 'GRAY', 'HSV'.")

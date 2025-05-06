import cv2
import numpy as np
import os
from typing import Optional

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Loads an image from the specified file path and converts it to RGB format.

    Args:
        image_path: The path to the image file.

    Returns:
        A NumPy array representing the image in RGB format,
        or None if the image cannot be loaded.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    # Load the image using OpenCV
    # cv2.imread loads in BGR format by default
    img = cv2.imread(image_path)

    if img is None:
        # This might happen if the file exists but is corrupted or not a valid image format
        print(f"Warning: Could not load image from path: {image_path}. File might be corrupted or an unsupported format.")
        return None

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Saves the given image (NumPy array) to the specified file path.

    Args:
        image: The NumPy array representing the image (expected in RGB format).
        output_path: The path where the image should be saved.

    Returns:
        True if the image was saved successfully, False otherwise.
    """
    if image is None:
        print("Error: Cannot save a None image.")
        return False

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Convert the image from RGB to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Save the image using OpenCV
        success = cv2.imwrite(output_path, img_bgr)
        if not success:
            print(f"Error: Failed to save image to {output_path}")
            return False
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

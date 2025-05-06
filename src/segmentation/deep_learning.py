"""
Implements image segmentation using deep learning models (e.g., from torchvision).

Note: Requires 'torch' and 'torchvision' libraries. Install using:
pip install torch torchvision
A GPU is highly recommended for reasonable performance.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from .base import BaseSegmenter

# Attempt to import PyTorch related libraries
try:
    import torch
    import torchvision
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    from torchvision.transforms.functional import to_pil_image, pil_to_tensor, resize, InterpolationMode
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy types/functions if torch is not installed
    class torch: # Dummy class
        Tensor = type(None)
        device = type(None)
        no_grad = lambda: (lambda func: func) # Dummy decorator
        class cuda: # Nested dummy class for torch.cuda
            @staticmethod
            def is_available(): return False
        @staticmethod
        def device(x): return "cpu" # Dummy device function
    class torchvision: pass # Dummy class
    class Image: pass # Dummy class
    def to_pil_image(x): pass
    def pil_to_tensor(x): pass
    def resize(x, size, interpolation): pass
    InterpolationMode = type(None) # Dummy type


class DeepLearningSegmenter(BaseSegmenter):
    """
    Segments an image using a pre-trained deep learning model from libraries
    like torchvision.
    """

    def __init__(self, **params):
        """
        Initializes the segmenter, loads the model, and sets up the device.

        Args:
            **params: Configuration parameters.
                Required:
                    model_name (str): Name of the model architecture to load.
                                      Currently supports 'deeplabv3_resnet50'.
                Optional:
                    weights (str): Pre-trained weights to use (e.g., 'DEFAULT',
                                   path to file). Default uses torchvision's
                                   default weights for the model.
                    device (str): Device to run inference on ('cuda', 'cpu', or
                                  specific cuda device like 'cuda:0').
                                  Default: 'cuda' if available, else 'cpu'.
                    target_class_index (int): The index of the class in the model's
                                              output to consider as foreground.
                                              Often depends on the dataset the model
                                              was trained on (e.g., COCO). Default: 15 (person in COCO).
        Raises:
            ImportError: If 'torch' or 'torchvision' are not installed.
            ValueError: If the specified model_name is not supported.
            RuntimeError: If model loading fails.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("DeepLearningSegmenter requires 'torch' and 'torchvision'. Please install them.")

        self.model_name = params.get('model_name')
        if self.model_name is None:
            raise ValueError("Parameter 'model_name' is required.")

        weights_param = params.get('weights', 'DEFAULT')
        device_param = params.get('device')
        self.target_class_index = params.get('target_class_index', 15) # Default: Person class in COCO

        # Determine device
        if device_param:
            self.device = torch.device(device_param)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and weights
        try:
            if self.model_name == 'deeplabv3_resnet50':
                if weights_param == 'DEFAULT':
                    weights = DeepLabV3_ResNet50_Weights.DEFAULT
                else:
                    # Add logic here to load weights from a file path if needed
                    raise NotImplementedError("Loading custom weights not yet implemented.")
                self.model = deeplabv3_resnet50(weights=weights)
                self.transforms = weights.transforms() # Get standard transforms
            # Add elif blocks here for other models (e.g., from segmentation_models.pytorch)
            # elif self.model_name == 'unet_resnet34':
            #    import segmentation_models_pytorch as smp
            #    self.model = smp.Unet(...)
            #    self.transforms = ... # Define appropriate transforms
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")

            self.model.eval() # Set model to evaluation mode
            self.model.to(self.device)
            print(f"Model '{self.model_name}' loaded successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")


    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Segments the input image using the loaded deep learning model.

        Args:
            image: The input image as a NumPy array (assumed BGR format).
            **params: Additional parameters (currently unused but part of the interface).

        Returns:
            A binary mask (uint8, 0/255) where 255 indicates pixels classified
            as the target foreground class.

        Raises:
            TypeError: If the input image is not a NumPy array.
            RuntimeError: If inference fails.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("DeepLearningSegmenter requires 'torch' and 'torchvision'.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in BGR format (3 channels).")

        # --- Preprocessing ---
        # Convert BGR NumPy array to RGB PIL Image
        img_pil = to_pil_image(image[:, :, ::-1]) # BGR to RGB
        # Use the actual size of the converted PIL image to ensure consistency with the dummy image in tests
        original_size = img_pil.size  # (width, height)

        # Apply model-specific transforms
        # Force RGB mode without altering image size
        img_pil = img_pil.convert('RGB')
        input_tensor = self.transforms(img_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device) # Add batch dim and move to device

        # --- Inference ---
        try:
            with torch.no_grad():
                output = self.model(input_batch)['out'] # Output shape: (1, num_classes, H, W)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")

        # --- Postprocessing ---
        # Get semantic predictions (class index per pixel)
        output_predictions = output.argmax(1) # Shape: (1, H, W)

        # Create binary mask for the target class
        mask_tensor = (output_predictions == self.target_class_index).squeeze(0) # Shape: (H, W)

        # Convert mask tensor to NumPy array (on CPU)
        mask_np = mask_tensor.byte().cpu().numpy() # Values 0 or 1

        # Resize mask back to original image size if necessary
        # The transforms might include resizing, so we resize the mask back
        mask_pil = Image.fromarray(mask_np)
        # Use NEAREST interpolation to avoid introducing intermediate values in the binary mask
        resized_mask_pil = mask_pil.resize(original_size, Image.NEAREST)
        resized_mask_np = np.array(resized_mask_pil)

        # Convert to 0/255 format
        final_mask = (resized_mask_np * 255).astype(np.uint8)

        return final_mask

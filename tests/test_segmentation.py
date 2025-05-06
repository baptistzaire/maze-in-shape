"""
Unit tests for the segmentation module.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, ANY
from PIL import Image # Import real Image

# Assume src is importable (e.g., project root added to PYTHONPATH or using pytest structure)
from src.segmentation import (
    get_segmenter,
    BaseSegmenter,
    ThresholdSegmenter,
    HSVSegmenter,
    CannyContourSegmenter,
    KMeansSegmenter,
    GrabCutSegmenter,
    RembgSegmenter,
    DeepLearningSegmenter,
    SEGMENTER_MAP
)

# --- Fixtures ---

@pytest.fixture
def sample_image_gray():
    """A simple grayscale image (e.g., black square on white background)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 200 # A gray square
    return img

@pytest.fixture
def sample_image_bgr():
    """A simple BGR image (e.g., blue square on black background)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75, 0] = 255 # Blue channel full
    return img

@pytest.fixture
def expected_mask_square():
    """Expected mask for the square in sample images."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask

# --- Helper Function ---

def check_mask(mask: np.ndarray, expected_shape: tuple):
    """Basic checks for a binary mask."""
    assert isinstance(mask, np.ndarray)
    assert mask.shape == expected_shape
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, [0, 255])) # Mask should only contain 0 or 255

# --- Test Cases ---

# 1. Base Class (ensure it's abstract)
def test_base_segmenter_abstract():
    with pytest.raises(TypeError):
        BaseSegmenter() # Cannot instantiate abstract class

# 2. ThresholdSegmenter
def test_threshold_segmenter_global(sample_image_gray, expected_mask_square):
    segmenter = ThresholdSegmenter()
    params = {'method': 'global', 'threshold_value': 100, 'threshold_type': cv2.THRESH_BINARY}
    mask = segmenter.segment(sample_image_gray.copy(), **params)
    check_mask(mask, sample_image_gray.shape)
    # Check if the square is segmented (approximate check)
    assert np.sum(mask[25:75, 25:75] == 255) > 2000 # Most pixels in square should be 255
    assert np.sum(mask[:25, :25] == 0) > 600 # Corner should be 0

def test_threshold_segmenter_otsu(sample_image_gray, expected_mask_square):
    segmenter = ThresholdSegmenter()
    # Otsu ignores threshold_value if None or if THRESH_OTSU is set
    params = {'method': 'global', 'threshold_value': None, 'threshold_type': cv2.THRESH_BINARY}
    mask = segmenter.segment(sample_image_gray.copy(), **params)
    check_mask(mask, sample_image_gray.shape)
    assert np.sum(mask[25:75, 25:75] == 255) > 2000

def test_threshold_segmenter_adaptive(sample_image_gray):
    segmenter = ThresholdSegmenter()
    params = {'method': 'adaptive', 'adaptive_method': cv2.ADAPTIVE_THRESH_MEAN_C, 'block_size': 11, 'C': 2}
    mask = segmenter.segment(sample_image_gray.copy(), **params)
    check_mask(mask, sample_image_gray.shape)
    # Adaptive results are harder to predict exactly, just check format

def test_threshold_segmenter_invalid_params(sample_image_gray):
    segmenter = ThresholdSegmenter()
    with pytest.raises(ValueError): # Missing method
        segmenter.segment(sample_image_gray.copy())
    with pytest.raises(ValueError): # Invalid block_size
        params = {'method': 'adaptive', 'block_size': 10}
        segmenter.segment(sample_image_gray.copy(), **params)

# 3. HSVSegmenter
def test_hsv_segmenter(sample_image_bgr, expected_mask_square):
    segmenter = HSVSegmenter()
    # Define broad HSV range for blue
    # OpenCV HSV: H: 0-179, S: 0-255, V: 0-255. Blue is around H=120.
    lower_blue = (100, 100, 100)
    upper_blue = (140, 255, 255)
    params = {'lower_hsv': lower_blue, 'upper_hsv': upper_blue}
    mask = segmenter.segment(sample_image_bgr.copy(), **params)
    check_mask(mask, sample_image_bgr.shape[:2])
    assert np.all(mask[25:75, 25:75] == 255) # Square should be perfectly masked
    assert np.all(mask[:25, :25] == 0) # Corner should be 0

def test_hsv_segmenter_invalid_params(sample_image_bgr):
    segmenter = HSVSegmenter()
    with pytest.raises(ValueError): # Missing bounds
        segmenter.segment(sample_image_bgr.copy())
    with pytest.raises(ValueError): # Wrong image format
        segmenter.segment(np.zeros((10, 10)), lower_hsv=(0,0,0), upper_hsv=(1,1,1))

# 4. CannyContourSegmenter
def test_canny_contour_segmenter(sample_image_gray, expected_mask_square):
    segmenter = CannyContourSegmenter()
    # Thresholds might need tuning for real images, but should work for simple square
    params = {'threshold1': 50, 'threshold2': 150, 'blur_ksize': 0}
    mask = segmenter.segment(sample_image_gray.copy(), **params)
    check_mask(mask, sample_image_gray.shape)
    # Check if the square area is filled
    assert np.sum(mask[30:70, 30:70] == 255) > 1500 # Inner part should be filled

def test_canny_contour_segmenter_no_contours(sample_image_gray):
    segmenter = CannyContourSegmenter()
    # Use thresholds that likely won't find edges
    params = {'threshold1': 500, 'threshold2': 600, 'blur_ksize': 0}
    # Create a blank image
    blank_image = np.zeros_like(sample_image_gray)
    mask = segmenter.segment(blank_image, **params)
    check_mask(mask, blank_image.shape)
    assert np.all(mask == 0) # Should return black mask if no contours

# 5. KMeansSegmenter
def test_kmeans_segmenter(sample_image_bgr):
    segmenter = KMeansSegmenter()
    # Expect 2 clusters: black background, blue square
    # Need to identify which cluster index is blue (might vary)
    # We'll run it and check if *one* of the clusters roughly matches the square
    params = {'num_clusters': 2, 'foreground_cluster_indices': [0]} # Guess index 0 is foreground
    try:
        mask0 = segmenter.segment(sample_image_bgr.copy(), **params)
        check_mask(mask0, sample_image_bgr.shape[:2])
    except Exception as e:
        pytest.fail(f"KMeans failed for index 0: {e}")

    params = {'num_clusters': 2, 'foreground_cluster_indices': [1]} # Guess index 1 is foreground
    try:
        mask1 = segmenter.segment(sample_image_bgr.copy(), **params)
        check_mask(mask1, sample_image_bgr.shape[:2])
    except Exception as e:
        pytest.fail(f"KMeans failed for index 1: {e}")

    # Check if one of the masks correctly identifies the square
    square_pixels_mask0 = np.sum(mask0[25:75, 25:75] == 255)
    square_pixels_mask1 = np.sum(mask1[25:75, 25:75] == 255)
    assert square_pixels_mask0 > 2000 or square_pixels_mask1 > 2000

def test_kmeans_segmenter_invalid_params(sample_image_bgr):
    segmenter = KMeansSegmenter()
    with pytest.raises(ValueError): # Missing num_clusters
        segmenter.segment(sample_image_bgr.copy(), foreground_cluster_indices=[0])
    with pytest.raises(ValueError): # Missing foreground_cluster_indices
        segmenter.segment(sample_image_bgr.copy(), num_clusters=2)
    with pytest.raises(ValueError): # Invalid index
        segmenter.segment(sample_image_bgr.copy(), num_clusters=2, foreground_cluster_indices=[2])

# 6. GrabCutSegmenter
def test_grabcut_segmenter(sample_image_bgr, expected_mask_square):
    segmenter = GrabCutSegmenter()
    # Define ROI around the square
    roi = (20, 20, 60, 60) # x, y, w, h
    params = {'roi_rect': roi, 'iterations': 5}
    mask = segmenter.segment(sample_image_bgr.copy(), **params)
    check_mask(mask, sample_image_bgr.shape[:2])
    # GrabCut should be quite accurate for this simple case
    assert np.sum(mask[25:75, 25:75] == 255) > 2400 # Expect most of the square

def test_grabcut_segmenter_invalid_params(sample_image_bgr):
    segmenter = GrabCutSegmenter()
    with pytest.raises(ValueError): # Missing roi_rect and initial_mask
        segmenter.segment(sample_image_bgr.copy(), iterations=5)
    with pytest.raises(ValueError): # Missing iterations
        segmenter.segment(sample_image_bgr.copy(), roi_rect=(0,0,10,10))
    with pytest.raises(ValueError): # Invalid ROI
        segmenter.segment(sample_image_bgr.copy(), roi_rect=(90, 90, 20, 20), iterations=5)

# 7. RembgSegmenter (Requires Mocking)
def test_rembg_segmenter(sample_image_bgr, expected_mask_square):
    # Use nested context managers instead of decorators
    with patch('src.segmentation.rembg_wrapper.REMBG_AVAILABLE', True) as mock_rembg_available_flag, \
         patch('src.segmentation.rembg_wrapper.rembg_remove') as mock_rembg_remove, \
         patch('src.segmentation.rembg_wrapper.Image') as mock_pil_image_class, \
         patch('src.segmentation.rembg_wrapper.cv2') as mock_cv2, \
         patch('src.segmentation.rembg_wrapper.np.array') as mock_np_array: # Also mock np.array here

        # --- Mock Setup ---
        mock_pil_image_instance = MagicMock()
        mock_pil_image_class.fromarray.return_value = mock_pil_image_instance

        fake_rgba_output = np.zeros((100, 100, 4), dtype=np.uint8)
        fake_rgba_output[:, :, :3] = sample_image_bgr
        fake_rgba_output[:, :, 3] = expected_mask_square
        mock_rembg_remove.return_value = MagicMock(
            tobytes=fake_rgba_output.tobytes(),
            size=fake_rgba_output.shape[1::-1],
            mode='RGBA'
        )
        mock_np_array.return_value = fake_rgba_output # Mock the result of np.array

        # --- Test Execution ---
        segmenter = RembgSegmenter()
        params = {'model_name': 'u2net'}
        mask = segmenter.segment(sample_image_bgr.copy(), **params)

    # --- Assertions ---
    check_mask(mask, expected_mask_square.shape)
    np.testing.assert_array_equal(mask, expected_mask_square)
    # Check cvtColor call manually due to numpy array comparison issues in assert_called_with
    assert mock_cv2.cvtColor.call_count == 1
    call_args, call_kwargs = mock_cv2.cvtColor.call_args
    np.testing.assert_array_equal(call_args[0], sample_image_bgr) # Check image argument
    assert call_args[1] == mock_cv2.COLOR_BGR2RGB # Check color code argument
    # mock_cv2.cvtColor.assert_called_once_with(sample_image_bgr, mock_cv2.COLOR_BGR2RGB) # Original failing line
    mock_pil_image_class.fromarray.assert_called_once()
    mock_rembg_remove.assert_called_once_with(
        mock_pil_image_instance,
            session=None,
            only_mask=False,
            alpha_matting=False,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
    mock_np_array.assert_called_once() # Check np.array was called (Corrected indentation)

def test_rembg_segmenter_not_available():
    with patch('src.segmentation.rembg_wrapper.REMBG_AVAILABLE', False):
        with pytest.raises(ImportError, match="rembg' library is not installed"):
            RembgSegmenter()

# 8. DeepLearningSegmenter (Requires Mocking)
def test_deep_learning_segmenter(sample_image_bgr, expected_mask_square):
    # Use nested context managers
    with patch('src.segmentation.deep_learning.TORCH_AVAILABLE', True) as mock_torch_available_flag, \
         patch('src.segmentation.deep_learning.torch') as mock_torch, \
         patch('src.segmentation.deep_learning.torchvision') as mock_torchvision, \
         patch('src.segmentation.deep_learning.Image') as mock_pil_image_class, \
         patch('src.segmentation.deep_learning.to_pil_image') as mock_to_pil, \
         patch('src.segmentation.deep_learning.resize') as mock_resize, \
         patch('src.segmentation.deep_learning.np.array') as mock_np_array:

        # --- Mock Setup ---
        mock_torch.device.return_value = "cpu"
        mock_torch.cuda.is_available.return_value = False
        mock_device = "cpu"

        # Mock the model instance returned by the torchvision function
        mock_model_instance = MagicMock()
        mock_torchvision.models.segmentation.deeplabv3_resnet50.return_value = mock_model_instance

        # Mock the weights object and its transforms() method
        mock_weights_obj = MagicMock()
        mock_transforms_callable = MagicMock() # This will represent the self.transforms callable
        mock_weights_obj.transforms.return_value = mock_transforms_callable
        mock_torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT = mock_weights_obj

        # Mock the input tensor that the transforms callable should return
        mock_input_tensor = MagicMock()
        mock_input_tensor.unsqueeze.return_value.to.return_value = mock_input_tensor
        mock_transforms_callable.return_value = mock_input_tensor # When self.transforms is called, return this

        # Mock the PIL image conversion
        dummy_pil_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), 'RGB')
        mock_to_pil.return_value = dummy_pil_image

        # Mock the model's output tensor processing
        mock_output_tensor = MagicMock()
        fake_logits = np.zeros((1, 2, 100, 100), dtype=np.float32)
        fake_logits[0, 1, 25:75, 25:75] = 1.0
        fake_logits[0, 0, :, :] = 0.5
        mock_argmax_result_tensor = MagicMock()
        mock_output_tensor.argmax.return_value = mock_argmax_result_tensor

        mock_squeezed_tensor = MagicMock()
        mock_argmax_result_tensor.squeeze.return_value = mock_squeezed_tensor

        mock_byte_tensor = MagicMock()
        mock_cpu_tensor = MagicMock()
        mock_cpu_tensor.numpy.return_value = (fake_logits.argmax(axis=1).squeeze(0) == 1).astype(np.uint8)
        mock_byte_tensor.cpu.return_value = mock_cpu_tensor
        mock_squeezed_tensor.byte.return_value = mock_byte_tensor

        # Ensure the mocked model instance returns the mocked output tensor
        mock_model_instance.return_value = {'out': mock_output_tensor}

        mock_mask_pil = MagicMock()
        mock_pil_image_class.fromarray.return_value = mock_mask_pil
        mock_resized_pil = MagicMock()
        mock_mask_pil.resize.return_value = mock_resized_pil
        mock_np_array.return_value = expected_mask_square # Mock final np.array call

        # --- Test Execution ---
        init_params = {'model_name': 'deeplabv3_resnet50', 'target_class_index': 1}
        segmenter = DeepLearningSegmenter(**init_params)
        mask = segmenter.segment(sample_image_bgr.copy())

        # --- Assertions ---
        check_mask(mask, expected_mask_square.shape)
        np.testing.assert_array_equal(mask, expected_mask_square)
        mock_torchvision.models.segmentation.deeplabv3_resnet50.assert_called_once_with(weights=mock_weights_obj) # Check correct weights obj used
        mock_model_instance.eval.assert_called_once()
        mock_model_instance.to.assert_called_with(mock_device) # Check model moved to device
        mock_to_pil.assert_called_once()
        mock_transforms_callable.assert_called_once_with(dummy_pil_image) # Check transforms called
        mock_input_tensor.unsqueeze.assert_called_once_with(0)
        mock_input_tensor.unsqueeze.return_value.to.assert_called_once_with(mock_device)
        mock_model_instance.assert_called_once_with(mock_input_tensor) # Check model called with tensor
        mock_pil_image_class.fromarray.assert_called_once()
        # Use ANY for the interpolation mode as it might be an enum value
        mock_mask_pil.resize.assert_called_once_with((100,100), ANY)
        mock_np_array.assert_called_once() # Check final np.array call

def test_deep_learning_segmenter_not_available():
    with patch('src.segmentation.deep_learning.TORCH_AVAILABLE', False):
        with pytest.raises(ImportError, match="requires 'torch' and 'torchvision'"):
            DeepLearningSegmenter(model_name='deeplabv3_resnet50')

# 9. Factory Function (get_segmenter)
def test_get_segmenter_valid():
    for name, klass in SEGMENTER_MAP.items():
        # Skip DL/rembg if deps not installed (mocks aren't active here)
        if name == "rembg":
            try: import rembg # type: ignore
            except ImportError: continue
        if name == "deep_learning":
            try: import torch # type: ignore
            except ImportError: continue

        # Provide necessary init params only for DL
        init_params = {}
        if name == "deep_learning":
            # Need to mock the model loading within the factory call scope
            with patch('src.segmentation.deep_learning.DeepLearningSegmenter.__init__', return_value=None):
                 init_params = {'model_name': 'deeplabv3_resnet50'} # Dummy param
                 segmenter = get_segmenter(name, **init_params)
                 assert isinstance(segmenter, klass)
        else:
            segmenter = get_segmenter(name, **init_params)
            assert isinstance(segmenter, klass)


def test_get_segmenter_invalid_name():
    with pytest.raises(ValueError, match="Unknown segmenter name: 'invalid_name'"):
         get_segmenter("invalid_name")

# Helper class for testing init errors
class _ErrorOnInitSegmenter(BaseSegmenter):
    def __init__(self, **kwargs):
        raise RuntimeError("Simulated init failure")
    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        # Needs to be implemented due to BaseSegmenter being abstract
        return np.zeros_like(image[:,:,0] if len(image.shape) == 3 else image)

@patch.dict(SEGMENTER_MAP, {"test_error": _ErrorOnInitSegmenter})
def test_get_segmenter_init_error():
     with pytest.raises(RuntimeError, match="Error initializing segmenter 'test_error'"):
         # The factory will now call _ErrorOnInitSegmenter.__init__ which raises the error
         get_segmenter("test_error")

# Test passing init_params to factory (specifically for DeepLearningSegmenter)
def test_get_segmenter_with_init_params():
    init_params = {'model_name': 'deeplabv3_resnet50', 'device': 'cpu', 'target_class_index': 10}
    # Use context managers
    with patch('src.segmentation.deep_learning.TORCH_AVAILABLE', True), \
         patch('src.segmentation.deep_learning.DeepLearningSegmenter.__init__', return_value=None) as mock_dl_init:
        segmenter = get_segmenter("deep_learning", **init_params)
        assert isinstance(segmenter, DeepLearningSegmenter)
        # Check if __init__ was called with the correct params
        mock_dl_init.assert_called_once_with(**init_params)

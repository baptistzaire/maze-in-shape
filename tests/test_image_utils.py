import pytest
import numpy as np
import cv2
import os

# Assume src is in the Python path or adjust imports accordingly
# For example, by running pytest from the project root directory
from src.image_utils.io import load_image, save_image
from src.image_utils.preprocess import resize_image, convert_color_space

# --- Fixtures ---

@pytest.fixture
def dummy_rgb_image() -> np.ndarray:
    """Creates a simple 10x20 RGB image."""
    return np.random.randint(0, 256, size=(10, 20, 3), dtype=np.uint8)

@pytest.fixture
def dummy_gray_image() -> np.ndarray:
    """Creates a simple 10x20 Grayscale image."""
    return np.random.randint(0, 256, size=(10, 20), dtype=np.uint8)

@pytest.fixture
def temp_image_file(tmp_path, dummy_rgb_image):
    """Creates a temporary image file for loading tests."""
    # OpenCV expects BGR, so convert fixture before saving
    img_bgr = cv2.cvtColor(dummy_rgb_image, cv2.COLOR_RGB2BGR)
    file_path = tmp_path / "test_image.png"
    cv2.imwrite(str(file_path), img_bgr)
    return str(file_path)

# --- Test io.py ---

def test_load_image_success(temp_image_file, dummy_rgb_image):
    """Test successful loading and conversion to RGB."""
    loaded_img = load_image(temp_image_file)
    assert loaded_img is not None
    assert loaded_img.shape == dummy_rgb_image.shape
    assert loaded_img.dtype == np.uint8
    # Check if it's roughly the same image (allowing for compression artifacts)
    assert np.allclose(loaded_img, dummy_rgb_image, atol=10) # Allow some tolerance

def test_load_image_not_found():
    """Test FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_path.jpg")

def test_load_image_corrupted(tmp_path, mocker):
    """Test handling of corrupted/invalid image files."""
    # Create an empty file
    corrupted_file = tmp_path / "corrupted.png"
    corrupted_file.touch()
    # Mock cv2.imread to simulate failure
    mocker.patch('cv2.imread', return_value=None)
    # Expect None to be returned and a warning printed (check capsys if needed)
    assert load_image(str(corrupted_file)) is None

def test_save_image_success(tmp_path, dummy_rgb_image):
    """Test successful saving of an image."""
    output_path = tmp_path / "output_image.jpg"
    success = save_image(dummy_rgb_image, str(output_path))
    assert success is True
    assert output_path.exists()
    # Optionally load it back and check
    loaded_back = cv2.imread(str(output_path))
    assert loaded_back is not None
    assert loaded_back.shape == dummy_rgb_image.shape

def test_save_image_none():
    """Test saving a None image."""
    success = save_image(None, "some_path.png")
    assert success is False

def test_save_image_creates_dir(tmp_path, dummy_rgb_image):
    """Test that save_image creates the output directory if it doesn't exist."""
    output_path = tmp_path / "new_dir" / "output_image.png"
    success = save_image(dummy_rgb_image, str(output_path))
    assert success is True
    assert output_path.exists()
    assert (tmp_path / "new_dir").is_dir()

# --- Test preprocess.py ---

def test_resize_image_max_dimension(dummy_rgb_image):
    """Test resizing based on max_dimension."""
    h, w = dummy_rgb_image.shape[:2] # 10, 20
    max_dim = 15
    resized = resize_image(dummy_rgb_image, max_dimension=max_dim)
    assert resized is not None
    new_h, new_w = resized.shape[:2]
    assert new_w == max_dim # Width was larger dimension
    assert new_h == int(h * (max_dim / w)) # Preserves aspect ratio
    assert new_h < max_dim

def test_resize_image_max_dimension_no_change(dummy_rgb_image):
    """Test resizing when image is smaller than max_dimension."""
    max_dim = 30 # Larger than image dimensions (10x20)
    resized = resize_image(dummy_rgb_image, max_dimension=max_dim)
    assert resized is not None
    assert resized.shape == dummy_rgb_image.shape
    # Check if it returns the *same* object if no resize needed
    assert np.array_equal(resized, dummy_rgb_image)

def test_resize_image_target_size(dummy_rgb_image):
    """Test resizing to a specific target size."""
    target_w, target_h = 5, 8
    resized = resize_image(dummy_rgb_image, target_size=(target_w, target_h))
    assert resized is not None
    assert resized.shape == (target_h, target_w, 3)

def test_resize_image_scale_factor(dummy_rgb_image):
    """Test resizing by a scale factor."""
    h, w = dummy_rgb_image.shape[:2] # 10, 20
    scale = 0.5
    resized = resize_image(dummy_rgb_image, scale_factor=scale)
    assert resized is not None
    assert resized.shape == (int(h * scale), int(w * scale), 3)

def test_resize_image_precedence(dummy_rgb_image):
    """Test parameter precedence: target_size > max_dimension > scale_factor."""
    target_w, target_h = 5, 8
    max_dim = 15
    scale = 0.5
    # Provide all, expect target_size to be used
    resized = resize_image(dummy_rgb_image, target_size=(target_w, target_h), max_dimension=max_dim, scale_factor=scale)
    assert resized.shape == (target_h, target_w, 3)

def test_resize_image_invalid_input():
    """Test ValueError for invalid input image."""
    with pytest.raises(ValueError, match="Input image cannot be None"):
        resize_image(None, max_dimension=100)

def test_resize_image_invalid_params(dummy_rgb_image):
    """Test ValueError for invalid resizing parameters."""
    with pytest.raises(ValueError, match="target_size must be a tuple"):
        resize_image(dummy_rgb_image, target_size=(-5, 10))
    with pytest.raises(ValueError, match="max_dimension must be a positive integer"):
        resize_image(dummy_rgb_image, max_dimension=-10)
    with pytest.raises(ValueError, match="scale_factor must be a positive number"):
        resize_image(dummy_rgb_image, scale_factor=0)

def test_convert_color_space_rgb_to_gray(dummy_rgb_image):
    """Test RGB to Grayscale conversion."""
    gray_img = convert_color_space(dummy_rgb_image, 'GRAY')
    assert gray_img is not None
    assert gray_img.ndim == 2 # Grayscale has no channel dimension
    assert gray_img.shape == dummy_rgb_image.shape[:2]
    assert gray_img.dtype == np.uint8

def test_convert_color_space_rgb_to_hsv(dummy_rgb_image):
    """Test RGB to HSV conversion."""
    hsv_img = convert_color_space(dummy_rgb_image, 'HSV')
    assert hsv_img is not None
    assert hsv_img.ndim == 3
    assert hsv_img.shape == dummy_rgb_image.shape
    assert hsv_img.dtype == np.uint8

def test_convert_color_space_gray_to_rgb(dummy_gray_image):
    """Test Grayscale to RGB conversion."""
    rgb_img = convert_color_space(dummy_gray_image, 'RGB')
    assert rgb_img is not None
    assert rgb_img.ndim == 3
    assert rgb_img.shape == (dummy_gray_image.shape[0], dummy_gray_image.shape[1], 3)
    assert rgb_img.dtype == np.uint8
    # Check if all channels are the same (as expected from GRAY->RGB)
    assert np.all(rgb_img[:, :, 0] == rgb_img[:, :, 1])
    assert np.all(rgb_img[:, :, 1] == rgb_img[:, :, 2])

def test_convert_color_space_no_change(dummy_rgb_image, dummy_gray_image):
    """Test conversion when already in target space."""
    rgb_img = convert_color_space(dummy_rgb_image, 'RGB')
    assert np.array_equal(rgb_img, dummy_rgb_image)
    gray_img = convert_color_space(dummy_gray_image, 'GRAY')
    assert np.array_equal(gray_img, dummy_gray_image)

def test_convert_color_space_invalid_target(dummy_rgb_image):
    """Test ValueError for unsupported target space."""
    with pytest.raises(ValueError, match="Unsupported target color space"):
        convert_color_space(dummy_rgb_image, 'YUV')

def test_convert_color_space_invalid_channels(dummy_gray_image):
    """Test ValueError for invalid input channels for a conversion."""
    with pytest.raises(ValueError, match="Input image must be RGB .* to convert to HSV"):
        convert_color_space(dummy_gray_image, 'HSV')
    # Create a 4-channel image (like RGBA) - not directly supported
    rgba_like = np.random.randint(0, 256, size=(10, 20, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Input image must be RGB .* to convert to Grayscale"):
        convert_color_space(rgba_like, 'GRAY')
    with pytest.raises(ValueError, match="Unsupported number of channels .* for conversion to RGB"):
        convert_color_space(rgba_like, 'RGB')

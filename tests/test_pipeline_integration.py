"""
Integration tests for the main maze generation pipeline.

Verifies that the `generate_maze_from_image` function runs end-to-end
with different configurations and produces valid output images.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import cv2 # For creating dummy image

# Assuming the pipeline function is importable from src
from src.main_pipeline import generate_maze_from_image

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def dummy_input_image() -> np.ndarray:
    """Creates a simple NumPy array image (black with white rectangle) for testing."""
    height, width = 60, 80
    img_np = np.zeros((height, width, 3), dtype=np.uint8) # Black background
    # Add a white rectangle in the middle
    cv2.rectangle(img_np, (15, 10), (width - 15, height - 10), (255, 255, 255), -1)
    return img_np

@pytest.fixture
def base_config() -> dict:
    """Provides a base configuration dictionary for tests."""
    return {
        'segmentation': {
            'method': 'threshold',
            'params': {
                'method': 'global', # Specify the thresholding method
                'threshold_value': 128 # Threshold for the white rectangle
            }
        },
        'grid': {'cell_size': 5},
        'maze': {'algorithm': 'dfs'}, # Default algorithm
        'solve': {'enabled': False}, # Don't solve by default for speed
        'rendering': {
            'style': 'silhouette', # Simpler to verify than overlay
            'linewidth': 1,
        }
    }

# --- Test Cases ---

def test_pipeline_threshold_dfs_silhouette(dummy_input_image, base_config, tmp_path):
    """
    Tests the pipeline with threshold segmentation, DFS maze, silhouette rendering.
    Verifies successful execution and output type/saving.
    """
    test_config = base_config.copy()
    # No changes needed for DFS/silhouette, they are defaults in base_config

    output_file = tmp_path / "test_thresh_dfs_sil.png"

    try:
        result_image = generate_maze_from_image(
            image_source=dummy_input_image,
            config_dict=test_config,
            output_path=output_file
        )

        # Assertions
        assert isinstance(result_image, Image.Image), "Output should be a PIL Image"
        assert result_image.size[0] > 0, "Output image width should be positive"
        assert result_image.size[1] > 0, "Output image height should be positive"
        assert output_file.exists(), "Output file should have been created"
        assert output_file.is_file(), "Output path should be a file"
        # Optionally check file size or try to load it
        assert output_file.stat().st_size > 100 # Check if file has reasonable size

    except Exception as e:
        pytest.fail(f"Pipeline execution failed unexpectedly: {e}")

def test_pipeline_threshold_prim_overlay(dummy_input_image, base_config, tmp_path):
    """
    Tests the pipeline with threshold segmentation, Prim's maze, overlay rendering.
    """
    test_config = base_config.copy()
    test_config['maze']['algorithm'] = 'prim'
    test_config['rendering']['style'] = 'overlay'
    test_config['solve']['enabled'] = True # Test solving as well
    test_config['rendering']['bg_color'] = (30, 30, 30, 255) # Add a BG for overlay

    output_file = tmp_path / "test_thresh_prim_over.png"

    try:
        result_image = generate_maze_from_image(
            image_source=dummy_input_image,
            config_dict=test_config,
            output_path=output_file
        )

        # Assertions
        assert isinstance(result_image, Image.Image), "Output should be a PIL Image"
        assert result_image.mode == "RGBA", "Overlay should produce RGBA image with BG"
        assert output_file.exists(), "Output file should have been created"

    except Exception as e:
        pytest.fail(f"Pipeline execution failed unexpectedly: {e}")

def test_pipeline_no_output_path(dummy_input_image, base_config):
    """
    Tests the pipeline runs correctly when no output_path is provided.
    """
    test_config = base_config.copy()

    try:
        result_image = generate_maze_from_image(
            image_source=dummy_input_image,
            config_dict=test_config,
            output_path=None # Explicitly None
        )
        # Assertions
        assert isinstance(result_image, Image.Image), "Output should be a PIL Image"

    except Exception as e:
        pytest.fail(f"Pipeline execution failed unexpectedly: {e}")

def test_pipeline_invalid_config_raises_error(dummy_input_image):
    """
    Tests that the pipeline raises ValueError for invalid configuration.
    """
    invalid_config = {
        'grid': {'cell_size': -5}, # Invalid cell size
        'segmentation': {'method': 'threshold'},
        'maze': {'algorithm': 'dfs'},
        'rendering': {'style': 'silhouette'},
    }
    with pytest.raises((ValueError, TypeError)): # Config validation raises ValueError or TypeError
        generate_maze_from_image(
            image_source=dummy_input_image,
            config_dict=invalid_config,
            output_path=None
        )

def test_pipeline_empty_mask_raises_error(base_config):
    """
    Tests that the pipeline raises ValueError if segmentation results in an empty mask
    that leads to no passable cells.
    """
    # Create an all-black image that will result in an empty mask after thresholding
    height, width = 60, 80
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    test_config = base_config.copy()
    # Ensure thresholding finds nothing
    test_config['segmentation']['params']['threshold_value'] = 250

    # Expecting a ValueError either from empty mask check or no passable cells check
    with pytest.raises(ValueError, match=r"(empty mask|No passable cells)"):
        generate_maze_from_image(
            image_source=black_image,
            config_dict=test_config,
            output_path=None
        )

# --- Placeholder for tests requiring external dependencies ---

# Example: Test with rembg (requires rembg installed and potentially models)
# @pytest.mark.skipif(not pytest.importorskip("rembg"), reason="rembg library not found")
# def test_pipeline_rembg_dfs(dummy_input_image, base_config, tmp_path):
#     """Tests the pipeline with rembg segmentation."""
#     test_config = base_config.copy()
#     test_config['segmentation']['method'] = 'rembg'
#     # Rembg might not need specific params here, depends on wrapper implementation
#     test_config['segmentation'].pop('params', None)
#
#     output_file = tmp_path / "test_rembg_dfs.png"
#
#     try:
#         result_image = generate_maze_from_image(
#             image_source=dummy_input_image, # Rembg might work better on real images
#             config_dict=test_config,
#             output_path=output_file
#         )
#         assert isinstance(result_image, Image.Image)
#         assert output_file.exists()
#     except Exception as e:
#         # Rembg can have specific setup errors (e.g., model download)
#         pytest.fail(f"Pipeline with rembg failed: {e}")

"""
Main pipeline orchestrator for the Maze-in-Shape Generator.

Connects the different stages: image loading, segmentation, grid creation,
maze generation, solving (optional), and rendering.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, Set, List, Tuple

import numpy as np
import cv2 # Import OpenCV
from PIL import Image, ImageDraw # Add PIL imports

# --- Import necessary components from the project ---
from .config import MazeConfig # Use the dataclass for config handling
from .image_utils.io import load_image, save_image
from .image_utils.preprocess import resize_image, convert_color_space
from .segmentation import get_segmenter, BaseSegmenter
from .grid.creation import create_grid_from_mask, MazeGrid
from .maze.factory import create_maze_generator
from .maze.start_end import get_start_end_points
from .maze.solve import solve_maze
from .rendering.draw import (
    render_maze_silhouette,
    render_maze_overlay,
    START_COLOR, END_COLOR, WALL_COLOR, BACKGROUND_COLOR,
    SOLUTION_COLOR, SHAPE_COLOR # Import default colors
)
from .maze.types import MazeData, Point, Wall, Cell, Maze # Import necessary types

# Define ImageType as PIL Image, as that's what rendering produces and expects for masks
ImageType = Image.Image

def _calculate_existing_walls(maze_grid: MazeGrid, passages: Set[Wall]) -> Set[Wall]:
    """
    Helper function to determine walls that *exist* (were not removed).

    Iterates through all potential walls between adjacent passable cells in the grid
    and returns the set of walls that are *not* in the `passages` set.

    Args:
        maze_grid: The MazeGrid object representing the valid maze area.
        passages: A set of Walls (pairs of cells) representing removed walls (paths).

    Returns:
        A set of Walls representing the walls that should be drawn.
    """
    all_possible_walls: Set[Wall] = set()
    for r in range(maze_grid.height):
        for c in range(maze_grid.width):
            # Only consider walls originating from passable cells
            if not maze_grid.is_passable(r, c):
                continue

            cell = Cell((r, c))

            # Check horizontal neighbor (down)
            if r + 1 < maze_grid.height:
                neighbor = Cell((r + 1, c))
                # Add potential wall if neighbor is also passable
                if maze_grid.is_passable(r + 1, c):
                     # Ensure consistent ordering for the set (smaller point first)
                     wall = tuple(sorted((cell, neighbor)))
                     all_possible_walls.add(Wall(wall))

            # Check vertical neighbor (right)
            if c + 1 < maze_grid.width:
                neighbor = Cell((r, c + 1))
                 # Add potential wall if neighbor is also passable
                if maze_grid.is_passable(r, c + 1):
                    wall = tuple(sorted((cell, neighbor)))
                    all_possible_walls.add(Wall(wall))

    # Walls that exist are all potential walls minus the passages carved out
    existing_walls = all_possible_walls - passages
    return existing_walls

def generate_maze_from_image(
    image_source: Union[str, Path, np.ndarray], # Accept np.ndarray (RGB) too
    config_dict: Dict[str, Any], # Accept dict for flexibility
    output_path: Optional[Union[str, Path]] = None
) -> ImageType: # Return PIL Image
    """
    Generates a maze within the shape of the main subject of an input image.

    Orchestrates the full pipeline: Load -> Preprocess -> Segment -> Grid ->
    Maze -> Start/End -> Solve (Optional) -> Render -> Save (Optional).

    Args:
        image_source: Path to the input image, a URL (if loader supports it),
                      or a pre-loaded image (NumPy array in RGB format).
        config_dict: A dictionary containing parameters for each stage.
                     See MazeConfig for expected structure and defaults.
                     Example keys: 'preprocessing', 'segmentation', 'grid',
                     'maze', 'solve', 'rendering'.
        output_path: Optional path to save the final generated maze image (as PNG).
                     If None, the image is returned but not saved.

    Returns:
        The final generated maze image as a PIL Image object.

    Raises:
        FileNotFoundError: If the image_source path does not exist.
        ValueError: If configuration is invalid, image loading/processing fails,
                    or a pipeline stage encounters an unrecoverable error (e.g.,
                    empty grid, no passable cells, invalid start/end points).
        ImportError: If a required dependency for a selected method is missing.
        TypeError: If input types are incorrect.
    """
    print(f"--- Starting Maze Generation Pipeline ---")

    # --- 0. Configuration Setup ---
    print("Step 0: Processing Configuration...")
    try:
        # Create MazeConfig instance from the input dictionary.
        # We manually map dictionary keys to MazeConfig fields for robustness.
        # This allows the input dict to be partial or have extra keys.
        config = MazeConfig(
            # Grid settings
            cell_size=config_dict.get('grid', {}).get('cell_size', MazeConfig.cell_size),
            # Segmentation settings
            segmentation_method=config_dict.get('segmentation', {}).get('method', MazeConfig.segmentation_method),
            threshold_value=config_dict.get('segmentation', {}).get('params', {}).get('threshold_value', MazeConfig.threshold_value),
            # Maze settings
            maze_algorithm=config_dict.get('maze', {}).get('algorithm', MazeConfig.maze_algorithm),
            start_point=config_dict.get('maze', {}).get('start_point', MazeConfig.start_point),
            end_point=config_dict.get('maze', {}).get('end_point', MazeConfig.end_point),
            # Rendering settings
            linewidth=config_dict.get('rendering', {}).get('linewidth', MazeConfig.linewidth),
            rendering_style=config_dict.get('rendering', {}).get('style', MazeConfig.rendering_style),
        )
        # MazeConfig's __post_init__ handles basic validation
        print(f"Using validated configuration: {config}")
    except (ValueError, TypeError) as e:
        # Catch specific config validation errors from MazeConfig.__post_init__
        raise ValueError(f"Pipeline Configuration Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during config processing
        raise ValueError(f"Unexpected error processing configuration: {e}")


    # --- 1. Load and Preprocess Image ---
    print("Step 1: Load & Preprocess Image...")
    try:
        if isinstance(image_source, (str, Path)):
            image_np = load_image(str(image_source)) # Returns NumPy array (RGB)
            if image_np is None:
                 # load_image might print a warning, but we raise a clear error here
                 raise ValueError(f"Image file loaded as None from path: {image_source}. Check file format/corruption.")
            print(f"Loaded image from path: {image_source}")
        elif isinstance(image_source, np.ndarray):
            # Assume input numpy array is already in RGB format
            # Check the image_source directly before assigning to image_np
            if image_source.ndim != 3 or image_source.shape[2] != 3:
                 raise ValueError(f"Input NumPy image must be 3-channel (RGB). Got shape {image_source.shape}")
            image_np = image_source.copy() # Work on a copy
            print("Using pre-loaded NumPy image.")
        else:
            # Catch incorrect input type early
            raise TypeError(f"Unsupported image_source type: {type(image_source)}. Must be str, Path, or NumPy array.")

        # --- Preprocessing (applied to NumPy array) ---
        preproc_config = config_dict.get('preprocessing', {})
        # Apply resizing if specified
        if 'resize' in preproc_config and preproc_config['resize']:
            print(f"Resizing image with params: {preproc_config['resize']}")
            image_np = resize_image(image_np, **preproc_config['resize']) # Can raise ValueError

        # Apply color space conversion if specified
        if 'colorspace' in preproc_config and preproc_config['colorspace']:
            target_space = preproc_config['colorspace']
            print(f"Converting image to colorspace: {target_space}")
            image_np = convert_color_space(image_np, target_space) # Can raise ValueError

        preprocessed_image_np = image_np # Final image after this stage
        print(f"Image loaded & preprocessed. Shape: {preprocessed_image_np.shape}, Type: {preprocessed_image_np.dtype}")

    except FileNotFoundError as e:
        # Specific error for missing file
        raise FileNotFoundError(f"Image Load Error: Input file not found at {image_source}. Details: {e}")
    except (ValueError, TypeError) as e:
        # Catch errors from load_image, resize, convert_color_space, or type checks
        raise ValueError(f"Image Load/Preprocessing Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors in this stage
        raise RuntimeError(f"Unexpected error during image loading/preprocessing: {e}")


    # --- 2. Segment Subject ---
    print(f"Step 2: Segment Subject using '{config.segmentation_method}'...")
    try:
        seg_params = config_dict.get('segmentation', {}).get('params', {})
        seg_init_params = config_dict.get('segmentation', {}).get('init_params', {})

        # get_segmenter can raise ValueError (unknown name) or ImportError
        segmenter: BaseSegmenter = get_segmenter(config.segmentation_method, **seg_init_params)

        # segment method can raise various errors depending on implementation
        mask_np = segmenter.segment(preprocessed_image_np, **seg_params)

        # --- Validate Mask Output ---
        if mask_np is None:
             raise ValueError("Segmentation failed: segment method returned None.")
        if mask_np.ndim != 2:
             raise ValueError(f"Segmentation Error: Expected 2D mask, but got {mask_np.ndim} dimensions.")
        if mask_np.dtype != np.uint8:
             # Attempt conversion if possible, otherwise raise error
             print(f"Warning: Segmentation mask has unexpected dtype {mask_np.dtype}. Attempting conversion to uint8.")
             try:
                 mask_np = mask_np.astype(np.uint8)
             except Exception as conv_e:
                 raise ValueError(f"Segmentation Error: Mask dtype is {mask_np.dtype}, and conversion to uint8 failed: {conv_e}")

        # Ensure mask is binary (0 or 255) for consistency downstream
        # Use a small threshold > 0 to binarize potential non-exact masks
        _, mask_np = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

        # Check if mask is all black (segmentation found nothing)
        if not np.any(mask_np):
            print("Warning: Segmentation resulted in an entirely black mask (no subject found).")
            # Depending on desired behavior, could raise error or continue with empty maze
            # raise ValueError("Segmentation Error: Resulting mask is empty (all black).")

        print(f"Segmentation complete. Mask shape: {mask_np.shape}, Type: {mask_np.dtype}, Non-zero pixels: {np.count_nonzero(mask_np)}")

    except (ValueError, ImportError, RuntimeError, TypeError, AttributeError) as e:
         # Catch errors from factory, segment method, or validation
         # Re-raise as ValueError for pipeline consistency, preserving original message
         raise ValueError(f"Segmentation Error ({config.segmentation_method}): {e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"Unexpected error during segmentation: {e}")


    # --- 3. Convert Mask to Grid ---
    print(f"Step 3: Convert Mask to Grid (cell size: {config.cell_size})...")
    try:
        # create_grid_from_mask expects uint8 mask (0 or non-zero)
        grid_array = create_grid_from_mask(mask_np, config.cell_size) # Can raise errors if mask/cell_size invalid
        maze_grid = MazeGrid(grid_array) # Encapsulates the boolean grid array

        # --- Validate Grid Output ---
        if maze_grid.height == 0 or maze_grid.width == 0:
            # This indicates the mask was too small for the cell size or completely empty
            raise ValueError(f"Grid creation resulted in an empty grid ({maze_grid.height}x{maze_grid.width}). "
                             f"Check cell_size ({config.cell_size}) relative to mask dimensions and content.")

        print(f"Grid created. Dimensions: {maze_grid.height}x{maze_grid.width}")

        # Extract the set of passable cells (Points) for maze generation/solving
        passable_cells: Set[Point] = {
            (r, c) for r in range(maze_grid.height) for c in range(maze_grid.width)
            if maze_grid.is_passable(r, c)
        }
        # --- Validate Passable Cells ---
        if not passable_cells:
             # This means the mask contained shape, but no cell centers landed on it
             raise ValueError("Grid Creation Error: No passable cells found in the grid. "
                              "The shape might be too thin or small for the chosen cell_size.")

        print(f"Found {len(passable_cells)} passable cells in the grid.")

    except (ValueError, TypeError) as e:
        # Catch errors from create_grid_from_mask or validation
        raise ValueError(f"Grid Creation Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"Unexpected error during grid creation: {e}")


    # --- 4. Generate Maze ---
    print(f"Step 4: Generate Maze using '{config.maze_algorithm}'...")
    try:
        # create_maze_generator can raise ValueError
        generator = create_maze_generator(config.maze_algorithm)
        # The generate method takes the set of passable cells (Points)
        # It can raise errors if the grid is invalid or generation fails
        passages: Maze = generator.generate(passable_cells) # passages is Set[Wall]

        # --- Validate Maze Output ---
        # Check if passages is a set (basic type check)
        if not isinstance(passages, set):
             raise TypeError(f"Maze generation Error: Expected a set of passages, got {type(passages)}.")
        # A valid maze on a grid with passable cells should usually have passages
        # (unless it's a 1x1 grid or completely disconnected)
        if not passages and len(passable_cells) > 1:
             print(f"Warning: Maze generation resulted in zero passages for {len(passable_cells)} passable cells. "
                   f"The resulting maze will have no paths.")
             # Consider if this should be an error depending on requirements

        print(f"Maze generated. Number of passages (removed walls): {len(passages)}")

        # Calculate the set of *existing* walls needed for rendering/solving
        existing_walls = _calculate_existing_walls(maze_grid, passages)
        print(f"Calculated number of existing walls: {len(existing_walls)}")

    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        # Catch errors from factory or generate method
        raise ValueError(f"Maze Generation Error ({config.maze_algorithm}): {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during maze generation: {e}")


    # --- 5. Select Start/End Points ---
    print("Step 5: Select Start/End Points...")
    try:
        # get_start_end_points needs passable cells, passages, and MazeConfig
        # It can raise ValueError for various reasons (invalid specified points, no path found for auto-selection)
        start_cell, end_cell = get_start_end_points(passable_cells, passages, config)

        # --- Validate Start/End Output ---
        if start_cell is None or end_cell is None:
             raise ValueError("Start/End selection failed to return valid points (returned None).")
        if start_cell not in passable_cells:
             raise ValueError(f"Selected start point {start_cell} is not a passable cell in the grid.")
        if end_cell not in passable_cells:
             raise ValueError(f"Selected end point {end_cell} is not a passable cell in the grid.")

        print(f"Selected Start point: {start_cell}, End point: {end_cell}")

    except ValueError as e:
        # Catch specific errors from get_start_end_points or validation
        raise ValueError(f"Start/End Point Selection Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during start/end point selection: {e}")


    # --- Prepare MazeData dictionary for Solving and Rendering ---
    # This step itself is unlikely to fail if inputs are validated
    maze_data: MazeData = {
        'width': maze_grid.width,
        'height': maze_grid.height,
        'walls': existing_walls, # Use the calculated existing walls
        'start': start_cell,
        'end': end_cell,
        'grid_mask': passable_cells # Provide the set of passable cells
    }


    # --- 6. Solve Maze (Optional) ---
    solution_path: Optional[List[Point]] = None
    solve_config = config_dict.get('solve', {})
    if solve_config.get('enabled', False):
        print("Step 6: Solve Maze...")
        try:
            # solve_maze takes the MazeData dictionary
            solution_path = solve_maze(maze_data) # Can return empty list if no solution

            # --- Validate Solution Output ---
            if solution_path is None:
                 # Should return list, even if empty. None indicates potential issue.
                 print("Warning: Maze solver returned None instead of a list. Treating as no solution.")
                 solution_path = []
            elif not isinstance(solution_path, list):
                 print(f"Warning: Maze solver returned type {type(solution_path)} instead of list. Treating as no solution.")
                 solution_path = []

            if solution_path:
                # Basic checks on the path
                if solution_path[0] != start_cell or solution_path[-1] != end_cell:
                     print(f"Warning: Solver path does not start/end at the expected points ({start_cell} -> {end_cell}). Path: {solution_path[:2]}...{solution_path[-2:]}")
                     # Don't necessarily discard, but log it.
                print(f"Maze solved. Path length: {len(solution_path)} steps.")
            else:
                # This is a valid outcome, not an error
                print("Maze could not be solved (no path found between start and end).")

        except Exception as e:
            # Treat solving failure as non-fatal, just print a warning
            print(f"Warning: Error encountered during maze solving: {e}")
            solution_path = None # Ensure solution_path is None if solving failed


    # --- 7. Render Maze ---
    print(f"Step 7: Render Maze (style: '{config.rendering_style}')...")
    try:
        render_config = config_dict.get('rendering', {})
        common_render_args = {
            'maze_data': maze_data,
            'cell_size': config.cell_size,
            'linewidth': config.linewidth,
            'start_marker_color': render_config.get('start_color', START_COLOR),
            'end_marker_color': render_config.get('end_color', END_COLOR),
            'wall_color': render_config.get('wall_color', WALL_COLOR),
            'solution_path': solution_path, # Pass the found solution path (or None)
            'solution_color': render_config.get('solution_color', SOLUTION_COLOR),
        }

        if config.rendering_style == 'silhouette':
            final_image = render_maze_silhouette(
                **common_render_args,
                bg_color=render_config.get('bg_color', BACKGROUND_COLOR)
            )
        elif config.rendering_style == 'overlay':
            # Overlay needs the original shape mask (as PIL Image)
            mask_pil = Image.fromarray(mask_np).convert('L')

            # Ensure mask size matches the expected rendering size
            expected_width = maze_grid.width * config.cell_size
            expected_height = maze_grid.height * config.cell_size
            if mask_pil.size != (expected_width, expected_height):
                 print(f"Resizing segmentation mask from {mask_pil.size} to {(expected_width, expected_height)} for overlay rendering.")
                 mask_pil = mask_pil.resize((expected_width, expected_height), Image.Resampling.NEAREST)

            final_image = render_maze_overlay(
                **common_render_args,
                shape_mask=mask_pil, # Pass the correctly sized PIL mask
                shape_color=render_config.get('shape_color', SHAPE_COLOR),
                bg_color=render_config.get('bg_color', None) # Default to transparent BG for overlay
            )
        else:
            # Invalid style string provided
            raise ValueError(f"Unsupported rendering style: '{config.rendering_style}'. Choose 'silhouette' or 'overlay'.")

        # --- Validate Render Output ---
        if not isinstance(final_image, Image.Image):
             raise TypeError(f"Rendering function did not return a PIL Image object (got {type(final_image)}).")

        print(f"Rendering complete. Final image size: {final_image.size}, Mode: {final_image.mode}")

    except (ValueError, TypeError) as e:
         # Catch specific errors from rendering functions or validation
         raise ValueError(f"Rendering Error: {e}")
    except Exception as e:
        # Rendering errors are critical to the final output
        raise RuntimeError(f"Unexpected error during maze rendering: {e}")


    # --- 8. Save Output (Optional) ---
    if output_path:
        output_path = Path(output_path) # Ensure it's a Path object
        print(f"Step 8: Save Output to {output_path}...")
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the final PIL Image (e.g., as PNG)
            final_image.save(output_path) # Can raise exceptions (IOError, etc.)
            print(f"Image successfully saved.")
        except Exception as e:
            # Log error but still return the image if rendering succeeded
            # Consider if saving failure should be fatal? For now, just warn.
            print(f"Warning: Failed to save image to {output_path}: {e}")


    print("--- Maze Generation Pipeline Finished Successfully ---")
    return final_image


# --- Example Usage Block ---
if __name__ == '__main__':
    print("\n=== Running Pipeline Test ===")
    # Define a test configuration dictionary
    test_config_dict: Dict[str, Any] = {
        'preprocessing': {
            # 'resize': {'max_dimension': 100} # Optional: Resize for faster testing
        },
        'segmentation': {
            'method': 'threshold',
            'params': {'threshold_value': 50} # Threshold for the white square
        },
        'grid': {'cell_size': 5}, # Smaller cell size for more detail
        'maze': {
            'algorithm': 'dfs',
            # 'start_point': (0, 0), # Optional: Specify start/end
            # 'end_point': (10, 10),
            },
        'solve': {'enabled': True}, # Attempt to solve the maze
        'rendering': {
            'style': 'overlay', # 'silhouette' or 'overlay'
            'linewidth': 1,
            'shape_color': (220, 220, 220, 180), # Semi-transparent grey shape
            'bg_color': (50, 50, 80, 255),      # Dark blue background
            'solution_color': 'yellow',
            }
    }

    # Create a dummy numpy image (e.g., 60x40)
    height, width = 40, 60
    dummy_image_np = np.zeros((height, width, 3), dtype=np.uint8) # Black background
    # Add a white rectangle in the middle for segmentation
    cv2.rectangle(dummy_image_np, (10, 5), (width - 10, height - 5), (255, 255, 255), -1)

    # Define output path
    test_output_dir = Path("test_output")
    test_output_path = test_output_dir / "pipeline_test_maze.png"

    try:
        # --- Run the pipeline with the NumPy array ---
        print(f"\nGenerating maze from NumPy array ({height}x{width})...")
        result_image = generate_maze_from_image(
            image_source=dummy_image_np,
            config_dict=test_config_dict,
            output_path=test_output_path
        )
        print(f"\nPipeline executed. Result image size: {result_image.size}")
        if test_output_path.exists():
             print(f"Output image saved to: {test_output_path}")
             # Optional: Open the image
             # import os
             # if os.name == 'nt': # Windows
             #     os.startfile(test_output_path)
             # elif os.name == 'posix': # macOS/Linux
             #     os.system(f'open "{test_output_path}"') # macOS
                 # os.system(f'xdg-open "{test_output_path}"') # Linux

    except (ImportError, FileNotFoundError, ValueError, RuntimeError, TypeError) as e:
        print(f"\n--- Pipeline Execution Failed ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
    finally:
        print("\n=== Pipeline Test Finished ===")
        # Clean up test output? (Optional)
        # if test_output_path.exists():
        #     test_output_path.unlink()
        # if test_output_dir.exists() and not any(test_output_dir.iterdir()):
        #      test_output_dir.rmdir()

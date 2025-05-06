"""
Unit tests for maze rendering functions.
"""

import pytest
from PIL import Image, ImageDraw

# Assume src is importable from the root directory
from src.maze.types import MazeData, Point
from src.rendering.draw import render_maze_silhouette, render_maze_overlay
from src.maze.solve import solve_maze # Needed for solution path testing

# --- Test Data Fixture ---

@pytest.fixture(scope="module")
def sample_maze_data() -> MazeData:
    """Provides a consistent, simple MazeData dictionary for tests."""
    # A simple 3x3 maze with a known path
    maze_data: MazeData = {
        'width': 3,
        'height': 3,
        'start': (0, 0),
        'end': (2, 2),
        'walls': { # Walls that EXIST (removed walls create paths)
            # Start with all possible internal walls
            # Horizontal walls between (r, c) and (r+1, c)
            ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)),
            ((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (2, 2)),
            # Vertical walls between (r, c) and (r, c+1)
            ((0, 0), (0, 1)), ((0, 1), (0, 2)),
            ((1, 0), (1, 1)), ((1, 1), (1, 2)),
            ((2, 0), (2, 1)), ((2, 1), (2, 2)),
        }
    }
    # Remove walls to create a simple path: (0,0) -> (1,0) -> (1,1) -> (2,1) -> (2,2)
    walls_to_remove = {
        ((0, 0), (1, 0)), # Horizontal between (0,0) and (1,0)
        ((1, 0), (1, 1)), # Vertical between (1,0) and (1,1)
        ((1, 1), (2, 1)), # Horizontal between (1,1) and (2,1)
        ((2, 1), (2, 2)), # Vertical between (2,1) and (2,2)
    }
    # Remove these walls from the set of existing walls
    for wall in walls_to_remove:
        # Check both orderings in case the set contains tuples in reverse
        if wall in maze_data['walls']:
            maze_data['walls'].remove(wall)
        elif (wall[1], wall[0]) in maze_data['walls']:
             maze_data['walls'].remove((wall[1], wall[0]))

    return maze_data

@pytest.fixture(scope="module")
def sample_solution_path(sample_maze_data: MazeData) -> list[Point]:
    """Provides the solution path for the sample_maze_data."""
    # Use the solver to find the path in the now-solvable maze
    path = solve_maze(sample_maze_data)
    assert path, "Solver failed for sample maze data - check maze/solver logic or maze data"
    return path

@pytest.fixture(scope="module")
def sample_shape_mask() -> Image.Image:
    """Provides a simple mask matching the sample maze dimensions."""
    cell_size = 10 # Must match cell_size used in tests
    width = 3 * cell_size
    height = 3 * cell_size
    mask = Image.new('L', (width, height), 0) # Black background
    draw = ImageDraw.Draw(mask)
    # Draw a white rectangle covering most of it
    draw.rectangle([(5, 5), (width - 6, height - 6)], fill=255)
    return mask

# --- Test Parameters ---
CELL_SIZE = 10
LINEWIDTH = 1

# --- Tests for render_maze_silhouette ---

def test_render_silhouette_runs(sample_maze_data: MazeData):
    """Test if silhouette rendering executes without errors."""
    try:
        img = render_maze_silhouette(sample_maze_data, CELL_SIZE, LINEWIDTH)
        assert img is not None
    except Exception as e:
        pytest.fail(f"render_maze_silhouette raised an exception: {e}")

def test_render_silhouette_output_type(sample_maze_data: MazeData):
    """Test if silhouette rendering returns a PIL Image."""
    img = render_maze_silhouette(sample_maze_data, CELL_SIZE, LINEWIDTH)
    assert isinstance(img, Image.Image)

def test_render_silhouette_output_size(sample_maze_data: MazeData):
    """Test if silhouette rendering returns an image of the correct size."""
    img = render_maze_silhouette(sample_maze_data, CELL_SIZE, LINEWIDTH)
    expected_width = sample_maze_data['width'] * CELL_SIZE
    expected_height = sample_maze_data['height'] * CELL_SIZE
    assert img.size == (expected_width, expected_height)

def test_render_silhouette_with_solution(sample_maze_data: MazeData, sample_solution_path: list[Point]):
    """Test silhouette rendering with a solution path."""
    try:
        img = render_maze_silhouette(
            sample_maze_data, CELL_SIZE, LINEWIDTH, solution_path=sample_solution_path
        )
        assert img is not None
        assert isinstance(img, Image.Image) # Check type again
    except Exception as e:
        pytest.fail(f"render_maze_silhouette with solution raised an exception: {e}")

# --- Tests for render_maze_overlay ---

def test_render_overlay_runs(sample_maze_data: MazeData, sample_shape_mask: Image.Image):
    """Test if overlay rendering executes without errors."""
    try:
        img = render_maze_overlay(sample_maze_data, sample_shape_mask, CELL_SIZE, LINEWIDTH)
        assert img is not None
    except Exception as e:
        pytest.fail(f"render_maze_overlay raised an exception: {e}")

def test_render_overlay_output_type(sample_maze_data: MazeData, sample_shape_mask: Image.Image):
    """Test if overlay rendering returns a PIL Image."""
    img = render_maze_overlay(sample_maze_data, sample_shape_mask, CELL_SIZE, LINEWIDTH)
    assert isinstance(img, Image.Image)

def test_render_overlay_output_size(sample_maze_data: MazeData, sample_shape_mask: Image.Image):
    """Test if overlay rendering returns an image of the correct size."""
    img = render_maze_overlay(sample_maze_data, sample_shape_mask, CELL_SIZE, LINEWIDTH)
    expected_width = sample_maze_data['width'] * CELL_SIZE
    expected_height = sample_maze_data['height'] * CELL_SIZE
    assert img.size == (expected_width, expected_height)
    assert img.mode == "RGBA" # Overlay should be RGBA

def test_render_overlay_with_solution(sample_maze_data: MazeData, sample_shape_mask: Image.Image, sample_solution_path: list[Point]):
    """Test overlay rendering with a solution path."""
    try:
        img = render_maze_overlay(
            sample_maze_data, sample_shape_mask, CELL_SIZE, LINEWIDTH, solution_path=sample_solution_path
        )
        assert img is not None
        assert isinstance(img, Image.Image) # Check type again
    except Exception as e:
        pytest.fail(f"render_maze_overlay with solution raised an exception: {e}")

def test_render_overlay_invalid_mask_size(sample_maze_data: MazeData):
    """Test overlay rendering raises ValueError with incorrect mask size."""
    wrong_size_mask = Image.new('L', (10, 10)) # Clearly wrong size
    with pytest.raises(ValueError, match="Shape mask size"):
        render_maze_overlay(sample_maze_data, wrong_size_mask, CELL_SIZE, LINEWIDTH)

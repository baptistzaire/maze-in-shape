"""
Tests for maze start and end point selection functionality.
"""

import numpy as np
import pytest
from src.grid.creation import MazeGrid
from src.config import MazeConfig
# Import internal BFS for testing purposes
from src.maze.start_end import (
    find_boundary_cells, find_farthest_points, _bfs_distances,
    validate_start_end_points, get_start_end_points
)
from src.maze.types import Cell, Wall

def test_find_boundary_cells_simple():
    """Test boundary cell detection with a simple 3x3 grid."""
    # Create a 3x3 grid with all cells passable
    grid = np.ones((3, 3), dtype=bool)
    maze_grid = MazeGrid(grid)
    
    # All cells should be boundary cells since it's a 3x3 grid
    boundary = find_boundary_cells(maze_grid)
    assert len(boundary) == 9
    
def test_find_boundary_cells_with_obstacle():
    """Test boundary cell detection with internal obstacles."""
    # Create a 4x4 grid
    grid = np.ones((4, 4), dtype=bool)
    # Add an obstacle in the middle
    grid[1:3, 1:3] = False
    maze_grid = MazeGrid(grid)
    
    boundary = find_boundary_cells(maze_grid)
    
    # Expected boundary cells:
    # - All edge cells that are passable (12)
    # - The cells adjacent to the central obstacle (8)
    # Some cells are counted twice (corner cells next to obstacle)
    expected_boundary = {
        # Edge cells
        (0, 0), (0, 1), (0, 2), (0, 3),  # Top edge
        (3, 0), (3, 1), (3, 2), (3, 3),  # Bottom edge
        (1, 0), (2, 0),  # Left edge
        (1, 3), (2, 3),  # Right edge
    }
    
    assert set(boundary) == expected_boundary
    assert len(boundary) == 12

def test_find_boundary_cells_empty():
    """Test boundary cell detection with empty grid."""
    grid = np.zeros((3, 3), dtype=bool)
    maze_grid = MazeGrid(grid)
    
    boundary = find_boundary_cells(maze_grid)
    assert len(boundary) == 0

def test_find_boundary_cells_complex_shape():
    """Test boundary cell detection with a more complex shape (plus sign)."""
    # Create a 5x5 grid
    grid = np.zeros((5, 5), dtype=bool)
    # Create a plus shape
    grid[2, :] = True  # Horizontal bar
    grid[:, 2] = True  # Vertical bar
    maze_grid = MazeGrid(grid)

    boundary = find_boundary_cells(maze_grid)

    # Note: The definition considers cells adjacent to *any* orthogonally impassable cell as boundary.
    # In this plus shape, the center cell (2,2) has only passable orthogonal neighbors,
    # so it's not considered a boundary cell by the function.
    expected_boundary = {
        (2, 0), (2, 1), (2, 3), (2, 4), # Horizontal arms (excluding center)
        (0, 2), (1, 2), (3, 2), (4, 2)  # Vertical arms (excluding center)
    }

    assert set(boundary) == expected_boundary
    assert len(boundary) == 8 # The 8 cells forming the arms, excluding the center

def test_find_boundary_cells_single_path():
    """Test boundary cell detection with a single path through grid."""
    grid = np.zeros((3, 3), dtype=bool)
    # Create a horizontal path in the middle
    grid[1, :] = True
    maze_grid = MazeGrid(grid)
    
    boundary = find_boundary_cells(maze_grid)
    expected = [(1, 0), (1, 1), (1, 2)]  # All path cells are boundary cells
    assert set(boundary) == set(expected)

def test_find_farthest_points_linear():
    """Test finding farthest points in a simple linear path."""
    # Create a 1x3 grid with all cells connected
    grid = {Cell((0, i)) for i in range(3)}
    passages = {
        Wall((Cell((0, 0)), Cell((0, 1)))),
        Wall((Cell((0, 1)), Cell((0, 2))))
    }
    
    start, end = find_farthest_points(grid, passages)
    # Should return the endpoints of the path, ordered by coordinate
    assert {start, end} == {Cell((0, 0)), Cell((0, 2))}
    assert start < end  # Verify ordering (smaller coordinate first)

def test_find_farthest_points_branching():
    """Test finding farthest points in a grid with multiple possible paths."""
    # Create a T-shaped path:
    #  0 1 2
    # +-----
    # |x x x 0
    # |  x   1
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 1))
    }
    passages = {
        Wall((Cell((0, 0)), Cell((0, 1)))),
        Wall((Cell((0, 1)), Cell((0, 2)))),
        Wall((Cell((0, 1)), Cell((1, 1))))
    }
    
    start, end = find_farthest_points(grid, passages)
    # Should find a pair with the maximum path distance (which is 2)
    # Valid pairs are {(0,0), (0,2)} and {(0,2), (1,1)} and {(0,0), (1,1)}
    # Let's check the distance is correct using the internal BFS helper
    distances = _bfs_distances(grid, passages, start)
    assert distances[end] == 2 
    # Check if the returned pair is one of the valid ones with max distance
    valid_pairs = [
        {Cell((0, 0)), Cell((0, 2))}, 
        {Cell((0, 2)), Cell((1, 1))},
        {Cell((0, 0)), Cell((1, 1))} 
    ]
    assert {start, end} in valid_pairs

def test_find_farthest_points_square_loop():
    """Test finding farthest points on a 2x2 square loop."""
    # Create a 2x2 square loop
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 0)), Cell((1, 1))}
    passages = {
        Wall((Cell((0, 0)), Cell((0, 1)))),
        Wall((Cell((0, 1)), Cell((1, 1)))),
        Wall((Cell((1, 1)), Cell((1, 0)))),
        Wall((Cell((1, 0)), Cell((0, 0)))),
    }

    start, end = find_farthest_points(grid, passages)
    # Farthest points should be diagonally opposite corners
    assert {start, end} == {Cell((0, 0)), Cell((1, 1))} or \
           {start, end} == {Cell((0, 1)), Cell((1, 0))}
    # Verify distance is 2
    distances = _bfs_distances(grid, passages, start)
    assert distances[end] == 2

def test_find_farthest_points_with_candidates():
    """Test finding farthest points when given specific candidate cells."""
    # Create a 3x3 grid with a path around the edge
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 2)),
        Cell((2, 0)), Cell((2, 1)), Cell((2, 2))
    }
    passages = {
        Wall((Cell((0, 0)), Cell((0, 1)))),
        Wall((Cell((0, 1)), Cell((0, 2)))),
        Wall((Cell((0, 2)), Cell((1, 2)))),
        Wall((Cell((1, 2)), Cell((2, 2)))),
        Wall((Cell((2, 1)), Cell((2, 2)))),
        Wall((Cell((2, 0)), Cell((2, 1)))),
        Wall((Cell((1, 0)), Cell((2, 0)))),
        Wall((Cell((0, 0)), Cell((1, 0))))
    }
    
    # Only use corner cells as candidates
    candidates = [Cell((0, 0)), Cell((0, 2)), Cell((2, 0)), Cell((2, 2))]
    start, end = find_farthest_points(grid, passages, candidates)
    # Should find opposite corners with longest path
    assert {start, end} == {Cell((0, 0)), Cell((2, 2))} or \
           {start, end} == {Cell((0, 2)), Cell((2, 0))}

def test_find_farthest_points_empty_grid():
    """Test that empty grid raises ValueError."""
    with pytest.raises(ValueError, match="Grid cannot be empty"):
        find_farthest_points(set(), set())

def test_find_farthest_points_empty_candidates():
    """Test that empty candidates list raises ValueError."""
    grid = {Cell((0, 0)), Cell((0, 1))}
    passages = {Wall((Cell((0, 0)), Cell((0, 1))))}
    with pytest.raises(ValueError, match="Candidates list cannot be empty"):
        find_farthest_points(grid, passages, [])

def test_find_farthest_points_no_paths():
    """Test finding farthest points in a grid with no passages."""
    grid = {Cell((0, 0)), Cell((0, 1))}
    passages = set()  # No passages between cells
    # Update the expected error message based on the actual implementation
    with pytest.raises(ValueError, match="No valid paths found between candidates from starting point"):
        find_farthest_points(grid, passages)

def test_validate_start_end_points_empty_grid():
    """Test that validation fails with empty grid."""
    with pytest.raises(ValueError, match="Grid cannot be empty"):
        validate_start_end_points(set(), start=(0, 0), end=(1, 1))

def test_validate_start_end_points_out_of_bounds():
    """Test validation with out-of-bounds points."""
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 0)), Cell((1, 1))}
    
    # Point beyond grid bounds
    with pytest.raises(ValueError, match="start_point .* is outside grid bounds"):
        validate_start_end_points(grid, start=(-1, 0))
        
    with pytest.raises(ValueError, match="end_point .* is outside grid bounds"):
        validate_start_end_points(grid, end=(2, 0))

def test_validate_start_end_points_impassable():
    """Test validation with points in impassable cells."""
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 1))}  # (1,0) is impassable
    
    with pytest.raises(ValueError, match="start_point .* is not in a passable cell"):
        validate_start_end_points(grid, start=(1, 0))
        
    with pytest.raises(ValueError, match="end_point .* is not in a passable cell"):
        validate_start_end_points(grid, end=(1, 0))

def test_validate_start_end_points_valid():
    """Test validation with valid points."""
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 1))}
    
    # Should not raise any exceptions
    validate_start_end_points(grid, start=(0, 0), end=(1, 1))
    validate_start_end_points(grid, start=(0, 1))  # Test with only one point
    validate_start_end_points(grid)  # Test with no points

def test_get_start_end_points_from_config():
    """Test getting points from config."""
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 0)), Cell((1, 1))}
    passages = {
        Wall((Cell((0, 0)), Cell((0, 1)))),
        Wall((Cell((0, 0)), Cell((1, 0)))),
    }
    
    config = MazeConfig(start_point=(0, 0), end_point=(1, 1))
    start, end = get_start_end_points(grid, passages, config)
    assert start == Cell((0, 0))
    assert end == Cell((1, 1))

def test_get_start_end_points_auto_select():
    """Test auto-selection of points when not specified in config."""
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 0)), Cell((1, 1))}
    passages = {
        Wall((Cell((0, 0)), Cell((0, 1)))),
        Wall((Cell((0, 1)), Cell((1, 1)))),
        Wall((Cell((1, 0)), Cell((1, 1)))),
    }
    
    config = MazeConfig()  # No points specified
    start, end = get_start_end_points(grid, passages, config)
    # Should choose boundary cells with maximum path distance
    # All cells are boundary cells in this 2x2 grid
    # Maximum path has length 3, e.g.:
    # (0,0) -> (0,1) -> (1,1) -> (1,0)
    distances = _bfs_distances(grid, passages, start)
    assert distances[end] == 3  # Check that we found a maximum distance path

def test_get_start_end_points_invalid_config():
    """Test that invalid config points are caught."""
    grid = {Cell((0, 0)), Cell((0, 1))}
    passages = {Wall((Cell((0, 0)), Cell((0, 1))))}
    
    config = MazeConfig(start_point=(2, 2), end_point=(3, 3))  # Out of bounds
    with pytest.raises(ValueError, match="outside grid bounds"):
        get_start_end_points(grid, passages, config)

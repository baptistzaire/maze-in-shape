import numpy as np
import pytest
from src.grid.creation import create_grid_from_mask, MazeGrid

def test_create_grid_simple_square():
    # Create a mask with a centered square
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    cell_size = 5
    grid = create_grid_from_mask(mask, cell_size)
    # Ensure grid dimensions are correct
    expected_shape = (50 // cell_size, 50 // cell_size)
    assert grid.shape == expected_shape

    # Check each cell based on the center pixel classification
    for i in range(expected_shape[0]):
        for j in range(expected_shape[1]):
            center_y = i * cell_size + cell_size // 2
            center_x = j * cell_size + cell_size // 2
            expected = (10 <= center_y < 40) and (10 <= center_x < 40)
            assert grid[i, j] == expected

def test_create_grid_empty_mask():
    # Mask with all zeros should yield a grid with all False
    mask = np.zeros((30, 30), dtype=np.uint8)
    cell_size = 3
    grid = create_grid_from_mask(mask, cell_size)
    assert np.all(grid == False)

def test_create_grid_full_mask():
    # Mask with all ones should yield a grid with all True
    mask = np.ones((30, 30), dtype=np.uint8)
    cell_size = 3
    grid = create_grid_from_mask(mask, cell_size)
    assert np.all(grid == True)

def test_maze_grid_properties():
    # Use a mask with a centered square region to test MazeGrid properties
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    cell_size = 2
    grid = create_grid_from_mask(mask, cell_size)
    maze_grid = MazeGrid(grid)
    # The grid dimensions should be 20 // 2 by 20 // 2
    assert maze_grid.height == 10
    assert maze_grid.width == 10
    # Check that a cell in the center is passable
    center_row, center_col = 3, 3
    assert maze_grid.is_passable(center_row, center_col) is True

def test_get_neighbors():
    # Create a full mask and generate the grid
    mask = np.ones((10, 10), dtype=np.uint8)
    cell_size = 2
    grid = create_grid_from_mask(mask, cell_size)
    maze_grid = MazeGrid(grid)
    
    # Test neighbors for a central cell; should have 4 neighbors (up, down, left, right)
    neighbors = maze_grid.get_neighbors(2, 2)
    assert len(neighbors) == 4
    # Test neighbors for a corner cell (0, 0); should have only two valid neighbors
    neighbors_corner = maze_grid.get_neighbors(0, 0)
    expected_neighbors = {(1, 0), (0, 1)}
    assert set(neighbors_corner) == expected_neighbors

if __name__ == "__main__":
    pytest.main()

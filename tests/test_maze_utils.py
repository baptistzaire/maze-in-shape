"""Tests for maze generation utility functions."""
import pytest
from typing import Set

from src.maze.types import Cell
from src.maze.utils import (
    get_valid_neighbors,
    get_all_walls,
    get_grid_dimensions,
    is_valid_cell
)

def test_get_valid_neighbors_empty():
    """Test getting neighbors with empty grid."""
    grid: Set[Cell] = set()
    neighbors = get_valid_neighbors(grid, 0, 0)
    assert len(neighbors) == 0

def test_get_valid_neighbors_single():
    """Test getting neighbors with single cell grid."""
    grid = {Cell((0, 0))}
    neighbors = get_valid_neighbors(grid, 0, 0)
    assert len(neighbors) == 0

def test_get_valid_neighbors_full():
    """Test getting neighbors in fully connected grid."""
    grid = {
        Cell((0, 0)), Cell((0, 1)),
        Cell((1, 0)), Cell((1, 1))
    }
    neighbors = get_valid_neighbors(grid, 0, 0)
    assert len(neighbors) == 2
    assert Cell((0, 1)) in neighbors
    assert Cell((1, 0)) in neighbors

def test_get_valid_neighbors_edge():
    """Test getting neighbors at grid edge."""
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((0, 2)),
        Cell((1, 0)), Cell((1, 1)), Cell((1, 2))
    }
    # Test corner cell
    corner_neighbors = get_valid_neighbors(grid, 0, 0)
    assert len(corner_neighbors) == 2
    assert Cell((0, 1)) in corner_neighbors
    assert Cell((1, 0)) in corner_neighbors
    
    # Test edge cell
    edge_neighbors = get_valid_neighbors(grid, 0, 1)
    assert len(edge_neighbors) == 3
    assert Cell((0, 0)) in edge_neighbors
    assert Cell((0, 2)) in edge_neighbors
    assert Cell((1, 1)) in edge_neighbors

def test_get_all_walls_empty():
    """Test getting walls with empty grid."""
    grid: Set[Cell] = set()
    walls = get_all_walls(grid)
    assert len(walls) == 0

def test_get_all_walls_basic():
    """Test getting walls in simple grid."""
    grid = {Cell((0, 0)), Cell((0, 1))}
    walls = get_all_walls(grid)
    assert len(walls) == 1
    wall = walls[0]
    assert wall == (Cell((0, 0)), Cell((0, 1)))

def test_get_all_walls_ordering():
    """Test wall cell ordering is consistent."""
    grid = {Cell((1, 1)), Cell((0, 1))}  # Deliberately unordered
    walls = get_all_walls(grid)
    assert len(walls) == 1
    wall = walls[0]
    # Should always order cells consistently
    assert wall == (Cell((0, 1)), Cell((1, 1)))

def test_get_grid_dimensions_empty():
    """Test grid dimensions with empty grid."""
    grid: Set[Cell] = set()
    rows, cols = get_grid_dimensions(grid)
    assert rows == 0
    assert cols == 0

def test_get_grid_dimensions_single():
    """Test grid dimensions with single cell."""
    grid = {Cell((0, 0))}
    rows, cols = get_grid_dimensions(grid)
    assert rows == 1
    assert cols == 1

def test_get_grid_dimensions_sparse():
    """Test grid dimensions with sparse grid."""
    grid = {Cell((0, 0)), Cell((2, 3))}  # Only two corners
    rows, cols = get_grid_dimensions(grid)
    assert rows == 3  # 0-2 inclusive
    assert cols == 4  # 0-3 inclusive

def test_is_valid_cell():
    """Test cell validity checking."""
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 0))}
    
    # Valid cells
    assert is_valid_cell(grid, 0, 0)
    assert is_valid_cell(grid, 0, 1)
    assert is_valid_cell(grid, 1, 0)
    
    # Invalid cells
    assert not is_valid_cell(grid, 1, 1)
    assert not is_valid_cell(grid, 2, 2)
    assert not is_valid_cell(grid, -1, 0)

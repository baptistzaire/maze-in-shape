"""Integration tests for maze generation algorithms.

Tests all implemented maze generators with various grid configurations and
verifies their output properties.
"""
import pytest
from typing import Set, List

from src.maze.types import Cell, Wall, Maze
from src.maze.factory import create_maze_generator, GENERATORS

def create_test_grid(rows: int, cols: int) -> Set[Cell]:
    """Create a rectangular grid of cells for testing.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        Set of cells forming a rectangular grid.
    """
    return {Cell((r, c)) for r in range(rows) for c in range(cols)}

def bfs_reachable_cells(maze: Maze, start: Cell, grid: Set[Cell]) -> Set[Cell]:
    """Find all cells reachable from start using BFS traversal.
    
    Args:
        maze: Set of removed walls forming the maze paths
        start: Starting cell for traversal
        grid: Set of valid cells in the maze
        
    Returns:
        Set of cells reachable from the start cell.
    """
    reachable = {start}
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        # Check all walls for connections
        for wall in maze:
            cell1, cell2 = wall
            if cell1 == current and cell2 not in reachable:
                reachable.add(cell2)
                queue.append(cell2)
            elif cell2 == current and cell1 not in reachable:
                reachable.add(cell1)
                queue.append(cell1)
    
    return reachable

def verify_maze_properties(maze: Maze, grid: Set[Cell]) -> None:
    """Verify common maze properties that should hold for all generators.
    
    Args:
        maze: Generated maze (set of removed walls)
        grid: Original grid cells
    
    Raises:
        AssertionError: If any property is violated
    """
    # Empty grid should produce empty maze
    if not grid:
        assert not maze, "Empty grid should produce empty maze"
        return
    
    # Single cell should produce empty maze
    if len(grid) == 1:
        assert not maze, "Single cell grid should produce empty maze"
        return
    
    # Get any cell to start traversal
    start = next(iter(grid))
    reachable = bfs_reachable_cells(maze, start, grid)
    
    # All cells should be reachable (maze is fully connected)
    assert reachable == grid, "Maze must connect all cells"
    
    # Perfect maze should have exactly n-1 walls removed
    assert len(maze) == len(grid) - 1, "Perfect maze must have n-1 removed walls"
    
    # Verify all walls connect adjacent cells
    for wall in maze:
        cell1, cell2 = wall
        row1, col1 = cell1
        row2, col2 = cell2
        assert abs(row1 - row2) + abs(col1 - col2) == 1, "Walls must connect adjacent cells"

@pytest.mark.parametrize('algorithm', sorted(GENERATORS.keys()))
def test_empty_grid(algorithm: str):
    """Test all generators with empty grid."""
    generator = create_maze_generator(algorithm)
    grid: Set[Cell] = set()
    maze = generator.generate(grid)
    verify_maze_properties(maze, grid)

@pytest.mark.parametrize('algorithm', sorted(GENERATORS.keys()))
def test_single_cell(algorithm: str):
    """Test all generators with single cell grid."""
    generator = create_maze_generator(algorithm)
    grid = {Cell((0, 0))}
    maze = generator.generate(grid)
    verify_maze_properties(maze, grid)

@pytest.mark.parametrize('algorithm', sorted(GENERATORS.keys()))
def test_2x2_grid(algorithm: str):
    """Test all generators with 2x2 grid."""
    generator = create_maze_generator(algorithm)
    grid = create_test_grid(2, 2)
    maze = generator.generate(grid)
    verify_maze_properties(maze, grid)

@pytest.mark.parametrize('algorithm', sorted(GENERATORS.keys()))
def test_3x3_grid(algorithm: str):
    """Test all generators with 3x3 grid."""
    generator = create_maze_generator(algorithm)
    grid = create_test_grid(3, 3)
    maze = generator.generate(grid)
    verify_maze_properties(maze, grid)

@pytest.mark.parametrize('algorithm', sorted(GENERATORS.keys()))
def test_rectangular_grid(algorithm: str):
    """Test all generators with rectangular grid."""
    generator = create_maze_generator(algorithm)
    grid = create_test_grid(2, 4)  # 2x4 grid
    maze = generator.generate(grid)
    verify_maze_properties(maze, grid)

@pytest.mark.parametrize('algorithm', sorted(GENERATORS.keys()))
def test_sparse_grid(algorithm: str):
    """Test all generators with sparse (disconnected) grid."""
    generator = create_maze_generator(algorithm)
    grid = {
        Cell((0, 0)), Cell((0, 1)),  # Connected pair
        Cell((2, 2))                  # Isolated cell
    }
    maze = generator.generate(grid)
    
    # Can't verify standard properties as grid is disconnected,
    # but can verify basic adjacency
    for wall in maze:
        cell1, cell2 = wall
        row1, col1 = cell1
        row2, col2 = cell2
        assert abs(row1 - row2) + abs(col1 - col2) == 1

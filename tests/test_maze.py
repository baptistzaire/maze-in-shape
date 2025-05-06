"""Tests for maze generation and representation."""
from src.maze.types import Cell, Wall, Maze

def test_cell_coordinates():
    """Test creation and usage of Cell coordinates."""
    cell = Cell((0, 1))
    assert cell == (0, 1)
    row, col = cell
    assert row == 0
    assert col == 1

def test_wall_representation():
    """Test creation and usage of Wall representation."""
    cell1 = Cell((0, 0))
    cell2 = Cell((0, 1))
    wall = Wall((cell1, cell2))
    assert wall == ((0, 0), (0, 1))

def test_maze_structure():
    """Test creation and usage of Maze structure."""
    # Create a simple maze with one removed wall
    cell1 = Cell((0, 0))
    cell2 = Cell((0, 1))
    wall = Wall((cell1, cell2))
    maze: Maze = {wall}
    
    # Verify maze contains the wall
    assert wall in maze
    assert len(maze) == 1
    
    # Add another wall and verify
    cell3 = Cell((1, 1))
    wall2 = Wall((cell2, cell3))
    maze.add(wall2)
    assert len(maze) == 2
    assert wall2 in maze

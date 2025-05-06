"""Tests for maze generation functionality."""
import pytest
from typing import Set

from src.maze.types import Cell, Wall, Maze
from src.maze.base_generator import BaseMazeGenerator

class DummyGenerator(BaseMazeGenerator):
    """A concrete implementation for testing the base generator."""
    def generate(self, grid: Set[Cell]) -> Maze:
        """Simple implementation that connects first two cells it finds."""
        cells = list(grid)
        if len(cells) < 2:
            return set()
        return {self._create_wall(cells[0], cells[1])}

def test_base_generator_abstract():
    """Test that BaseMazeGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseMazeGenerator()

def test_concrete_generator():
    """Test that a concrete implementation can be instantiated."""
    generator = DummyGenerator()
    assert isinstance(generator, BaseMazeGenerator)

def test_get_valid_neighbors():
    """Test the _get_valid_neighbors helper method."""
    generator = DummyGenerator()
    grid = {
        Cell((0, 0)), Cell((0, 1)), Cell((1, 0)),
        # Note: (1, 1) deliberately missing to test invalid neighbor
    }
    
    neighbors = generator._get_valid_neighbors(Cell((0, 0)), grid)
    assert len(neighbors) == 2
    assert Cell((0, 1)) in neighbors
    assert Cell((1, 0)) in neighbors
    assert Cell((1, 1)) not in neighbors  # Invalid neighbor

def test_create_wall():
    """Test the _create_wall helper method."""
    generator = DummyGenerator()
    
    # Test wall creation is consistent regardless of cell order
    cell1, cell2 = Cell((0, 0)), Cell((0, 1))
    wall1 = generator._create_wall(cell1, cell2)
    wall2 = generator._create_wall(cell2, cell1)
    
    assert wall1 == wall2
    assert wall1 == Wall((cell1, cell2))  # Cells should be ordered

def test_generate_simple_maze():
    """Test generating a simple maze with the dummy implementation."""
    generator = DummyGenerator()
    grid = {Cell((0, 0)), Cell((0, 1)), Cell((1, 0))}
    
    maze = generator.generate(grid)
    
    assert len(maze) == 1  # Dummy impl connects first two cells only
    wall = list(maze)[0]
    assert isinstance(wall, Wall)
    assert all(cell in grid for cell in wall)

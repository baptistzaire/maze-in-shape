"""Tests for maze generator factory."""
import pytest

from src.maze.factory import create_maze_generator, GENERATORS
from src.maze.base_generator import BaseMazeGenerator
from src.maze.dfs import DFSMazeGenerator
from src.maze.prim import PrimMazeGenerator
from src.maze.kruskal import KruskalMazeGenerator
from src.maze.wilson import WilsonMazeGenerator

def test_create_dfs_generator():
    """Test creating DFS maze generator."""
    generator = create_maze_generator('dfs')
    assert isinstance(generator, DFSMazeGenerator)
    assert isinstance(generator, BaseMazeGenerator)

def test_create_prim_generator():
    """Test creating Prim's maze generator."""
    generator = create_maze_generator('prim')
    assert isinstance(generator, PrimMazeGenerator)
    assert isinstance(generator, BaseMazeGenerator)

def test_create_kruskal_generator():
    """Test creating Kruskal's maze generator."""
    generator = create_maze_generator('kruskal')
    assert isinstance(generator, KruskalMazeGenerator)
    assert isinstance(generator, BaseMazeGenerator)

def test_create_wilson_generator():
    """Test creating Wilson's maze generator."""
    generator = create_maze_generator('wilson')
    assert isinstance(generator, WilsonMazeGenerator)
    assert isinstance(generator, BaseMazeGenerator)

def test_case_insensitive():
    """Test that algorithm names are case insensitive."""
    assert isinstance(create_maze_generator('DFS'), DFSMazeGenerator)
    assert isinstance(create_maze_generator('Prim'), PrimMazeGenerator)
    assert isinstance(create_maze_generator('KRUSKAL'), KruskalMazeGenerator)
    assert isinstance(create_maze_generator('Wilson'), WilsonMazeGenerator)

def test_invalid_algorithm():
    """Test error handling for invalid algorithm names."""
    with pytest.raises(ValueError) as exc_info:
        create_maze_generator('invalid')
    
    # Verify error message includes valid options
    error_msg = str(exc_info.value)
    assert 'invalid' in error_msg
    for algorithm in GENERATORS.keys():
        assert algorithm in error_msg

def test_all_generators_registered():
    """Test that all implemented generators are available in factory."""
    expected_algorithms = {'dfs', 'prim', 'kruskal', 'wilson'}
    assert set(GENERATORS.keys()) == expected_algorithms

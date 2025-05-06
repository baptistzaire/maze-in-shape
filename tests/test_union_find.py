"""Tests for Union-Find data structure."""
import pytest
from src.maze.union_find import UnionFind

def test_init():
    """Test initialization of Union-Find structure."""
    uf = UnionFind[int]()
    assert uf.num_sets == 0

def test_make_set():
    """Test make_set operation."""
    uf = UnionFind[int]()
    uf.make_set(1)
    assert uf.num_sets == 1
    # Making same set twice shouldn't change count
    uf.make_set(1)
    assert uf.num_sets == 1

def test_find():
    """Test find operation."""
    uf = UnionFind[str]()
    uf.make_set("a")
    assert uf.find("a") == "a"
    # Find should create set if doesn't exist
    assert uf.find("b") == "b"
    assert uf.num_sets == 2

def test_union():
    """Test union operation."""
    uf = UnionFind[int]()
    
    # Test basic union
    uf.make_set(1)
    uf.make_set(2)
    assert uf.num_sets == 2
    
    assert uf.union(1, 2)  # Should succeed
    assert uf.num_sets == 1
    
    assert not uf.union(1, 2)  # Already unified
    assert uf.num_sets == 1

def test_connected():
    """Test connected check."""
    uf = UnionFind[str]()
    
    # Initially separate
    uf.make_set("a")
    uf.make_set("b")
    uf.make_set("c")
    assert not uf.connected("a", "b")
    
    # After union
    uf.union("a", "b")
    assert uf.connected("a", "b")
    assert not uf.connected("b", "c")
    
    # Transitive connection
    uf.union("b", "c")
    assert uf.connected("a", "c")

def test_path_compression():
    """Test path compression optimization."""
    uf = UnionFind[int]()
    
    # Create a chain: 1 -> 2 -> 3 -> 4
    for i in range(1, 5):
        uf.make_set(i)
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(3, 4)
    
    # Find should compress path
    root = uf.find(1)
    # All elements should now point directly to root
    assert uf.find(2) == root
    assert uf.find(3) == root
    assert uf.find(4) == root

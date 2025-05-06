"""Union-Find data structure implementation.

This module provides a Union-Find (disjoint set) data structure implementation
used by Kruskal's algorithm to efficiently track connected components.
"""
from typing import Dict, TypeVar, Generic, Set

T = TypeVar('T')

class UnionFind(Generic[T]):
    """Union-Find data structure with path compression and union by rank.
    
    This implementation uses path compression during find operations and
    union by rank during merge operations to achieve near-constant time
    operations.
    """
    
    def __init__(self):
        """Initialize an empty Union-Find data structure."""
        self._parent: Dict[T, T] = {}  # item -> parent
        self._rank: Dict[T, int] = {}  # root -> rank
        self._sets: int = 0  # number of disjoint sets
    
    def make_set(self, x: T) -> None:
        """Create a new set containing a single element.
        
        Args:
            x: The element to create a set for.
        """
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._sets += 1
    
    def find(self, x: T) -> T:
        """Find the representative element (root) for a set.
        
        Uses path compression to optimize future lookups.
        
        Args:
            x: The element to find the representative for.
        
        Returns:
            The representative element of x's set.
        """
        if x not in self._parent:
            self.make_set(x)
        
        # Path compression: point all nodes directly to root
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]
    
    def union(self, x: T, y: T) -> bool:
        """Merge the sets containing x and y if they are different.
        
        Uses union by rank to keep trees shallow.
        
        Args:
            x: First element.
            y: Second element.
        
        Returns:
            True if sets were merged, False if already in same set.
        """
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return False
        
        # Union by rank: attach smaller rank tree under root of larger rank
        if self._rank[x_root] < self._rank[y_root]:
            self._parent[x_root] = y_root
        elif self._rank[x_root] > self._rank[y_root]:
            self._parent[y_root] = x_root
        else:
            self._parent[y_root] = x_root
            self._rank[x_root] += 1
        
        self._sets -= 1
        return True
    
    def connected(self, x: T, y: T) -> bool:
        """Check if two elements are in the same set.
        
        Args:
            x: First element.
            y: Second element.
            
        Returns:
            True if elements are in the same set, False otherwise.
        """
        return self.find(x) == self.find(y)
    
    @property
    def num_sets(self) -> int:
        """Get the current number of disjoint sets.
        
        Returns:
            Number of disjoint sets.
        """
        return self._sets

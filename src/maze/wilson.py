"""Wilson's algorithm maze generator implementation.

This module implements a maze generator using Wilson's algorithm, which creates
unbiased uniform spanning trees using loop-erased random walks.
"""
import random
from typing import Set, Dict, List, Optional
from collections import defaultdict

from .types import Cell, Wall, Maze
from .base_generator import BaseMazeGenerator

class WilsonMazeGenerator(BaseMazeGenerator):
    """Maze generator using Wilson's algorithm.
    
    This implementation generates truly uniform spanning trees by performing
    loop-erased random walks. While slower than other algorithms, it produces
    unbiased results.
    """
    
    def _random_walk(self, start: Cell, grid: Set[Cell], 
                    in_maze: Set[Cell], max_attempts: int = 1000) -> List[Cell]:
        """Perform a loop-erased random walk until hitting a cell in the maze.
        
        Args:
            start: Starting cell for the random walk
            grid: Set of valid cells that can be visited
            in_maze: Set of cells already in the maze
            max_attempts: Maximum number of steps before restarting walk
            
        Returns:
            List of cells in the walk (including start and end points)
        """
        attempts = 0
        current = start
        path = [current]
        
        while current not in in_maze and attempts < max_attempts:
            attempts += 1
            # Get valid neighbors that are in the grid
            neighbors = self._get_valid_neighbors(current, grid)
            if not neighbors:
                # If stuck, restart walk from start
                current = start
                path = [current]
                attempts = 0
                continue
            
            # Choose random neighbor and add to path
            next_cell = random.choice(neighbors)
            current = next_cell
            
            # If we've seen this cell before, erase the loop
            if next_cell in path:
                loop_start = path.index(next_cell)
                path = path[:loop_start + 1]
                attempts = 0  # Reset attempts after loop erasure
            else:
                path.append(next_cell)
        
        # If we hit max attempts, fall back to connecting to nearest in_maze cell
        if current not in in_maze:
            nearest = min(in_maze, key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1]))
            path.append(nearest)
        
        return path
    
    def _find_connected_region(self, cell: Cell, grid: Set[Cell]) -> Set[Cell]:
        """Find all cells in grid that can be reached from the given cell.
        
        Args:
            cell: Starting cell
            grid: Set of valid cells
            
        Returns:
            Set of cells that can be reached through valid moves.
        """
        connected = {cell}
        to_visit = [cell]
        
        while to_visit:
            current = to_visit.pop()
            for neighbor in self._get_valid_neighbors(current, grid):
                if neighbor not in connected:
                    connected.add(neighbor)
                    to_visit.append(neighbor)
        
        return connected

    def generate(self, grid: Set[Cell]) -> Maze:
        """Generate a maze using Wilson's algorithm.
        
        Args:
            grid: Set of valid cells where the maze can be carved.
            
        Returns:
            A set of walls that have been removed to create the maze paths.
        """
        if not grid:
            return set()
        
        # Start with random cell in the maze
        start = random.choice(list(grid))
        # Find all cells reachable from start
        region = self._find_connected_region(start, grid)
        
        in_maze: Set[Cell] = {start}
        removed_walls: Maze = set()
        
        # Process only cells in the connected region
        remaining = region - in_maze
        while remaining:
            # Pick random unvisited cell to start walk
            start = random.choice(list(remaining))
            
            # Perform loop-erased random walk
            path = self._random_walk(start, grid, in_maze)
            
            # Add path to maze
            for i in range(len(path) - 1):
                cell1, cell2 = path[i], path[i + 1]
                if cell1 in grid and cell2 in grid:  # Extra safety check
                    removed_walls.add(self._create_wall(cell1, cell2))
            
            # Mark all path cells as in maze
            for cell in path:
                if cell in grid:  # Extra safety check
                    in_maze.add(cell)
            
            # Update remaining cells in this region
            remaining = region - in_maze
        
        # Process other regions independently if they exist
        unprocessed = grid - region
        while unprocessed:
            # Start new region
            start = random.choice(list(unprocessed))
            region = self._find_connected_region(start, unprocessed)
            
            in_region: Set[Cell] = {start}
            remaining = region - in_region
            
            # Process this region
            while remaining:
                start = random.choice(list(remaining))
                path = self._random_walk(start, region, in_region)
                
                for i in range(len(path) - 1):
                    cell1, cell2 = path[i], path[i + 1]
                    if cell1 in region and cell2 in region:
                        removed_walls.add(self._create_wall(cell1, cell2))
                
                for cell in path:
                    if cell in region:
                        in_region.add(cell)
                
                remaining = region - in_region
            
            # Remove processed region from unprocessed cells
            unprocessed -= region
        
        return removed_walls

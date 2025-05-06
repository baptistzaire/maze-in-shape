"""
Module for determining start and end points in maze generation.
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import deque
from ..config import MazeConfig
import numpy as np
from ..grid.creation import MazeGrid
from .types import Cell, Wall
from .utils import get_valid_neighbors

def _bfs_distances(grid: Set[Cell], passages: Set[Wall], start: Cell) -> Dict[Cell, int]:
    """
    Perform BFS from start cell to find distances to all reachable cells.
    
    Args:
        grid: Set of valid/passable cells
        passages: Set of walls that have been removed (passages between cells)
        start: Starting cell for BFS
        
    Returns:
        Dictionary mapping each reachable cell to its distance from start
    """
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        
        # Check neighbors we can move to through passages
        for neighbor in get_valid_neighbors(grid, current[0], current[1]):
            # Check if there's a passage between cells
            wall = (current, neighbor) if current < neighbor else (neighbor, current)
            if wall in passages and neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances

def find_boundary_cells(maze_grid: MazeGrid) -> List[Tuple[int, int]]:
    """
    Find all valid cells that could serve as maze entry/exit points.
    
    This function identifies potential candidates for maze entry/exit points by locating
    cells that are:
    1. Passable (True in the grid)
    2. And any of:
       - Adjacent to at least one impassable cell
       - Located on the edge of the grid
       - Part of a small grid (3x3 or smaller) where all cells are potential entry/exit points
    
    Parameters:
        maze_grid (MazeGrid): The maze grid to analyze
        
    Returns:
        List[Tuple[int, int]]: List of (row, col) coordinates for boundary cells
    """
    boundary_cells = []
    
    for row in range(maze_grid.height):
        for col in range(maze_grid.width):
            # Skip if current cell is not passable
            if not maze_grid.is_passable(row, col):
                continue
                
            # Check if cell is on grid edge
            is_edge = (row == 0 or row == maze_grid.height - 1 or 
                      col == 0 or col == maze_grid.width - 1)
            
            # Check all possible neighbors (including out of bounds)
            potential_neighbors = [
                (row-1, col), (row+1, col),
                (row, col-1), (row, col+1)
            ]
            
            # Count non-passable or out-of-bounds neighbors
            non_passable_count = sum(
                1 for r, c in potential_neighbors
                if not maze_grid.is_passable(r, c)
            )
            
            # Add to boundary cells if:
            # - Grid is small (3x3 or smaller) OR
            # - Cell is on edge OR
            # - Has any non-passable/out-of-bounds neighbors
            is_small_grid = maze_grid.height <= 3 and maze_grid.width <= 3
            if is_small_grid or is_edge or non_passable_count > 0:
                boundary_cells.append((row, col))
    
    return boundary_cells

def find_farthest_points(grid: Set[Cell], passages: Set[Wall], candidates: List[Cell] = None) -> Tuple[Cell, Cell]:
    """
    Find the pair of cells that have the longest path between them through the maze.
    
    This uses a two-pass BFS algorithm:
    1. First pass from an arbitrary start point to find the farthest reachable point
    2. Second pass from that point to find the true farthest point
    
    The resulting points are guaranteed to have the longest possible path between them
    in the maze (the maze diameter).
    
    Args:
        grid: Set of valid/passable cells in the maze
        passages: Set of walls that have been removed (passages between cells)
        candidates: Optional list of cells to consider as start/end points.
                   If None, all passable cells are considered.
    
    Returns:
        Tuple of (start_cell, end_cell) representing the two farthest points
    
    Raises:
        ValueError: If grid is empty or no valid path exists between cells
    """
    if not grid:
        raise ValueError("Grid cannot be empty")
        
    # If no candidates provided, use all passable cells
    if candidates is None:
        candidates = list(grid)
    elif not candidates:
        raise ValueError("Candidates list cannot be empty")
    
    # Standard two-pass BFS algorithm to find maze diameter endpoints within candidates
    
    # --- First Pass ---
    # Start BFS from the first candidate
    start_node = candidates[0]
    distances1 = _bfs_distances(grid, passages, start_node)
    
    if len(distances1) <= 1 and len(grid) > 1: # Check if only start node is reachable
         # Check if any candidate is reachable at all
        reachable_candidates = [c for c in candidates if c in distances1]
        if not reachable_candidates or (len(reachable_candidates) == 1 and reachable_candidates[0] == start_node):
             raise ValueError("No valid paths found between candidates from starting point")

    # Find the node 'u' within candidates that is farthest from start_node
    u = None
    max_dist1 = -1
    for candidate in candidates:
        if candidate in distances1:
            dist = distances1[candidate]
            if dist > max_dist1:
                max_dist1 = dist
                u = candidate
            # Tie-breaking: prefer candidate with larger coordinates if distances are equal
            elif dist == max_dist1 and candidate > u:
                 u = candidate

    if u is None:
         # This might happen if the start_node is isolated from all other candidates
         raise ValueError("Could not find a reachable candidate in the first BFS pass")

    # --- Second Pass ---
    # Start BFS from node 'u'
    distances2 = _bfs_distances(grid, passages, u)

    # Find the node 'v' within candidates that is farthest from 'u'
    v = None
    max_dist2 = -1
    for candidate in candidates:
        if candidate in distances2:
            dist = distances2[candidate]
            if dist > max_dist2:
                max_dist2 = dist
                v = candidate
            # Tie-breaking: prefer candidate with larger coordinates if distances are equal
            elif dist == max_dist2 and candidate > v:
                 v = candidate
                 
    if v is None:
         # This should theoretically not happen if u was found and is a candidate
         raise ValueError("Could not find a reachable candidate in the second BFS pass")

    # Return the pair (u, v) ordered consistently
    if u > v:
        u, v = v, u
    return (u, v)

def validate_start_end_points(
    grid: Set[Cell], 
    start: Optional[Tuple[int, int]] = None, 
    end: Optional[Tuple[int, int]] = None
) -> None:
    """
    Validate that the specified start/end points are valid for the given grid.
    
    Valid points must be:
    1. Within grid bounds (determined by the coordinates of cells in the grid)
    2. In passable cells (cells that exist in the grid)
    
    Args:
        grid: Set of valid/passable cells
        start: Optional start point as (row, col)
        end: Optional end point as (row, col)
        
    Raises:
        ValueError: If either point is invalid
    """
    if not grid:
        raise ValueError("Grid cannot be empty")
    
    # Only validate points that are provided
    points_to_check = []
    if start is not None:
        points_to_check.append(('start', start))
    if end is not None:
        points_to_check.append(('end', end))
        
    if not points_to_check:
        return
        
    # Find grid bounds
    min_row = min(cell[0] for cell in grid)
    max_row = max(cell[0] for cell in grid)
    min_col = min(cell[1] for cell in grid)
    max_col = max(cell[1] for cell in grid)
    
    # Check each point
    for point_name, point in points_to_check:
        row, col = point
        
        # Check bounds
        if not (min_row <= row <= max_row and min_col <= col <= max_col):
            raise ValueError(
                f"{point_name}_point ({row}, {col}) is outside grid bounds "
                f"[{min_row}..{max_row}, {min_col}..{max_col}]"
            )
        
        # Check if cell is passable
        if Cell((row, col)) not in grid:
            raise ValueError(
                f"{point_name}_point ({row}, {col}) is not in a passable cell"
            )

def get_start_end_points(
    grid: Set[Cell],
    passages: Set[Wall],
    config: MazeConfig
) -> Tuple[Cell, Cell]:
    """
    Get start and end points for the maze, either from config or automatically.
    
    If start_point and end_point are specified in config, validates and returns them.
    Otherwise, finds suitable points automatically using find_farthest_points.
    
    Args:
        grid: Set of valid/passable cells
        passages: Set of walls that have been removed (passages between cells)
        config: Configuration object that may contain start/end points
        
    Returns:
        Tuple of (start_cell, end_cell)
        
    Raises:
        ValueError: If specified points are invalid
    """
    # If points specified in config, validate and use them
    if config.start_point is not None and config.end_point is not None:
        # This will raise ValueError if points are invalid
        validate_start_end_points(grid, config.start_point, config.end_point)
        return (Cell(config.start_point), Cell(config.end_point))
    
    # Otherwise, find boundary cells and use them as candidates
    # First create a numpy grid from the set of cells
    if not grid:
        raise ValueError("Grid cannot be empty")
        
    # Find grid dimensions
    max_row = max(cell[0] for cell in grid)
    max_col = max(cell[1] for cell in grid)
    
    # Create numpy array and mark passable cells
    np_grid = np.zeros((max_row + 1, max_col + 1), dtype=bool)
    for cell in grid:
        np_grid[cell[0], cell[1]] = True
    
    # Get boundary cells using MazeGrid
    maze_grid = MazeGrid(np_grid)
    boundary_cells = [Cell((r, c)) for r, c in find_boundary_cells(maze_grid)]
    return find_farthest_points(grid, passages, boundary_cells)

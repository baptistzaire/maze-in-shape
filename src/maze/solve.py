"""
Functions for solving the generated maze (finding a path from start to end).
"""

from collections import deque
from typing import List, Optional, Set, Tuple

from .types import MazeData, Point # Import the newly defined types

def solve_maze(maze_data: MazeData) -> List[Point]:
    """
    Solves the maze using Breadth-First Search (BFS) to find the shortest path.

    Args:
        maze_data: A dictionary containing maze details:
            - width (int): Number of columns in the grid.
            - height (int): Number of rows in the grid.
            - walls (Set[Tuple[Point, Point]]): A set of pairs of adjacent
              cells ((r1, c1), (r2, c2)) that have a wall BETWEEN them.
            - start (Point): The (row, col) of the start cell.
            - end (Point): The (row, col) of the end cell.
            - grid_mask (Optional[Set[Point]]): Optional set of valid cells.
              If provided, only cells in the mask are considered traversable.

    Returns:
        A list of (row, col) tuples representing the shortest path from
        start to end, including both start and end points. Returns an
        empty list if no solution is found.
    """
    width = maze_data['width']
    height = maze_data['height']
    walls = maze_data['walls']
    start = maze_data['start']
    end = maze_data['end']
    grid_mask = maze_data.get('grid_mask') # Optional: Use if provided

    if not start or not end:
        return [] # Cannot solve without start and end points

    # Check if start/end are valid cells if grid_mask is provided
    if grid_mask and (start not in grid_mask or end not in grid_mask):
        return [] # Start or end is outside the valid maze area

    # Initialize queue for BFS: stores tuples of (current_point, path_to_current)
    queue = deque([(start, [start])])
    visited: Set[Point] = {start}

    while queue:
        current_point, path = queue.popleft()

        if current_point == end:
            return path # Solution found

        cr, cc = current_point

        # Explore neighbors (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = cr + dr, cc + dc
            neighbor: Point = (nr, nc)

            # 1. Check bounds
            if not (0 <= nr < height and 0 <= nc < width):
                continue

            # 2. Check if neighbor is valid (if mask exists)
            if grid_mask and neighbor not in grid_mask:
                continue

            # 3. Check for wall between current and neighbor
            wall_exists = (current_point, neighbor) in walls or \
                          (neighbor, current_point) in walls
            if wall_exists:
                continue

            # 4. Check if visited
            if neighbor in visited:
                continue

            # If neighbor is valid, reachable, and unvisited:
            visited.add(neighbor)
            new_path = path + [neighbor]
            queue.append((neighbor, new_path))

    return [] # No solution found if queue empties

# Example Usage (requires a MazeData structure)
if __name__ == '__main__':
    # --- Test 1: Standard Rectangular Maze ---
    # Use the same dummy maze from draw.py for consistency
    # (Assuming it's solvable)
    dummy_maze_solve: MazeData = {
        'width': 5,
        'height': 4,
        'start': (0, 0),
        'end': (3, 4), # Adjusted in draw.py example, adjust here too
        'walls': {
            # Horizontal walls
            ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 3), (1, 3)), ((0, 4), (1, 4)),
            ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((1, 3), (2, 3)),
            ((2, 0), (3, 0)), ((2, 2), (3, 2)), ((2, 4), (3, 4)),
            # Vertical walls
            ((0, 0), (0, 1)), ((0, 2), (0, 3)), ((0, 3), (0, 4)),
            ((1, 0), (1, 1)), ((1, 1), (1, 2)),
            ((2, 1), (2, 2)), ((2, 3), (2, 4)),
            ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (3, 3)), ((3, 3), (3, 4)),
        },
        'grid_mask': None
    }
    # Adjust end point if necessary (matching draw.py example)
    if dummy_maze_solve['end'][1] >= dummy_maze_solve['width']:
         dummy_maze_solve['end'] = (dummy_maze_solve['end'][0], dummy_maze_solve['width'] - 1)

    print(f"Solving dummy maze from {dummy_maze_solve['start']} to {dummy_maze_solve['end']}...")
    solution_path = solve_maze(dummy_maze_solve)

    if solution_path:
        print("Solution found:")
        print(solution_path)
        print(f"Path length: {len(solution_path)} steps")
    else:
        print("No solution found for the dummy maze.")

    # Example with grid_mask (make a simple L-shape mask)
    masked_maze = dummy_maze_solve.copy() # Shallow copy is okay here
    masked_maze['grid_mask'] = {
        (0,0), (1,0), (2,0), (3,0), # Left column
        (3,1), (3,2), (3,3), (3,4)  # Bottom row
    }
    # Adjust start/end if they fall outside the new mask
    if masked_maze['start'] not in masked_maze['grid_mask']:
        print("Adjusting start for masked maze")
        masked_maze['start'] = (0,0) # Keep start if valid
    if masked_maze['end'] not in masked_maze['grid_mask']:
         print(f"Adjusting end {masked_maze['end']} for masked maze")
         masked_maze['end'] = (3, 4) # Keep end if valid (adjust if needed)
         if masked_maze['end'][1] >= masked_maze['width']:
             masked_maze['end'] = (masked_maze['end'][0], masked_maze['width'] - 1)
         # Check again if the adjusted end point is valid
         if masked_maze['end'] not in masked_maze['grid_mask']:
             print(f"Adjusted end point {masked_maze['end']} is still outside the mask. Expecting no solution.")

    print(f"\nSolving masked maze from {masked_maze['start']} to {masked_maze['end']}...")
    masked_solution_path = solve_maze(masked_maze)

    if masked_solution_path:
        print("Solution found for masked maze:")
        print(masked_solution_path)
        print(f"Path length: {len(masked_solution_path)} steps")
    else:
        print("No solution found for the masked maze.")

"""
Module for grid creation from binary mask for maze generation.
"""

import numpy as np

def create_grid_from_mask(mask: np.ndarray, cell_size: int) -> np.ndarray:
    """
    Create a discrete grid from a binary mask using center-point classification.

    The function divides the binary mask into cells of size (cell_size x cell_size)
    and determines the cell's value by checking the mask value at the center of each cell.
    A cell is set to True if the center pixel value is non-zero, otherwise False.

    Parameters:
        mask (np.ndarray): A 2D binary NumPy array where non-zero values represent the subject.
        cell_size (int): The size of each cell in the grid.

    Returns:
        np.ndarray: A 2D boolean NumPy array representing the grid.
    """
    height, width = mask.shape
    # Determine grid dimensions (using floor division to ignore partial cells)
    grid_rows = height // cell_size
    grid_cols = width // cell_size

    grid = np.zeros((grid_rows, grid_cols), dtype=bool)

    for i in range(grid_rows):
        for j in range(grid_cols):
            center_y = i * cell_size + cell_size // 2
            center_x = j * cell_size + cell_size // 2

            # Ensure the computed center is within bounds
            if center_y >= height or center_x >= width:
                continue

            grid[i, j] = bool(mask[center_y, center_x])
    
    return grid

class MazeGrid:
    """
    Encapsulates the grid for maze generation and provides grid-aware helper methods.
    
    Attributes:
        grid (np.ndarray): A 2D boolean array representing the maze grid.
                          True indicates a passable cell, False indicates an obstacle.
    """

    def __init__(self, grid: np.ndarray):
        """
        Initializes the MazeGrid with a given grid.
        
        Parameters:
            grid (np.ndarray): A 2D boolean NumPy array representing the grid.
        """
        self.grid = grid

    @property
    def height(self) -> int:
        """Returns the number of rows in the grid."""
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        """Returns the number of columns in the grid."""
        return self.grid.shape[1]

    def is_passable(self, row: int, col: int) -> bool:
        """
        Determines if the cell at (row, col) is passable.
        
        Parameters:
            row (int): Row index.
            col (int): Column index.
            
        Returns:
            bool: True if the cell is passable, False otherwise.
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            return bool(self.grid[row, col])
        return False

    def get_neighbors(self, row: int, col: int) -> list:
        """
        Returns valid neighboring cells (up, down, left, right) for a given cell.
        
        Parameters:
            row (int): Row index.
            col (int): Column index.
            
        Returns:
            list: List of (row, col) tuples for valid neighbor cells.
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

if __name__ == "__main__":
    # Example usage:
    # Create a dummy binary mask (for instance, a 100x100 grid with a simple shape)
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[30:70, 30:70] = 1  # Create a square in the middle

    cell_size = 10
    grid_array = create_grid_from_mask(dummy_mask, cell_size)
    print("Generated grid shape:", grid_array.shape)
    print(grid_array)

    # Example usage of MazeGrid class
    maze_grid = MazeGrid(grid_array)
    print("MazeGrid height:", maze_grid.height)
    print("MazeGrid width:", maze_grid.width)
    print("Passable at (2,2):", maze_grid.is_passable(2, 2))
    print("Neighbors of (2,2):", maze_grid.get_neighbors(2, 2))

"""
Configuration module for the Maze-in-Shape Generator.

Defines a dataclass to hold tunable parameters for the maze generation pipeline.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

# Define allowed literal types for configuration options
SegmentationMethod = Literal["threshold", "rembg"] # Add more as implemented
MazeAlgorithm = Literal["dfs", "prim"] # Add more as implemented
RenderingStyle = Literal["silhouette", "overlay"]

@dataclass
class MazeConfig:
    """
    Configuration settings for the maze generation pipeline.

    Attributes:
        cell_size: The size of each cell in the grid (pixels).
        segmentation_method: The method used for subject segmentation.
        maze_algorithm: The algorithm used for maze generation.
        linewidth: The width of the maze walls in the rendered output (pixels).
        rendering_style: The style used for rendering the final maze image.
        # Add more parameters as needed, e.g., threshold values, start/end points
    """
    cell_size: int = 10
    segmentation_method: SegmentationMethod = "threshold"
    maze_algorithm: MazeAlgorithm = "dfs"
    linewidth: int = 1
    rendering_style: RenderingStyle = "overlay"

    # Start and end points for the maze (row, col), if None will be auto-selected
    start_point: Optional[Tuple[int, int]] = None
    end_point: Optional[Tuple[int, int]] = None
    
    # Parameters for specific methods
    threshold_value: int = 128 # Used only if segmentation_method is 'threshold'

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.linewidth <= 0:
            raise ValueError("linewidth must be positive")
        
        # If only one point is specified, both must be specified
        if bool(self.start_point) != bool(self.end_point):
            raise ValueError("Both start_point and end_point must be specified if either is provided")
        
        # Validate start/end point format if provided
        if self.start_point is not None:
            if not isinstance(self.start_point, tuple) or len(self.start_point) != 2:
                raise ValueError("start_point must be a tuple of (row, col)")
            if not all(isinstance(x, int) for x in self.start_point):
                raise ValueError("start_point coordinates must be integers")
                
        if self.end_point is not None:
            if not isinstance(self.end_point, tuple) or len(self.end_point) != 2:
                raise ValueError("end_point must be a tuple of (row, col)")
            if not all(isinstance(x, int) for x in self.end_point):
                raise ValueError("end_point coordinates must be integers")

# Example of how to create a config instance with defaults
# default_config = MazeConfig()

# Example of how to create a config instance with overrides
# custom_config = MazeConfig(cell_size=5, maze_algorithm="prim")

def load_default_config() -> MazeConfig:
    """
    Loads and returns the default MazeConfig.

    Returns:
        An instance of MazeConfig with default values.
    """
    return MazeConfig()

"""
Tests for configuration validation.
"""

import pytest
from src.config import MazeConfig

def test_config_valid_start_end_points():
    """Test MazeConfig with valid start/end points."""
    config = MazeConfig(start_point=(0, 0), end_point=(1, 1))
    assert config.start_point == (0, 0)
    assert config.end_point == (1, 1)

def test_config_missing_one_point():
    """Test MazeConfig with only one of start/end points specified."""
    with pytest.raises(ValueError, match="Both start_point and end_point must be specified if either is provided"):
        MazeConfig(start_point=(0, 0))
    
    with pytest.raises(ValueError, match="Both start_point and end_point must be specified if either is provided"):
        MazeConfig(end_point=(1, 1))

def test_config_invalid_point_format():
    """Test MazeConfig with invalid point formats."""
    # Wrong type
    with pytest.raises(ValueError, match=r"start_point must be a tuple of \(row, col\)"):
        MazeConfig(start_point=[0, 0], end_point=(1, 1))
        
    # Wrong size
    with pytest.raises(ValueError, match=r"end_point must be a tuple of \(row, col\)"):
        MazeConfig(start_point=(0, 0), end_point=(1, 1, 1))
        
    # Wrong element type
    with pytest.raises(ValueError, match="start_point coordinates must be integers"):
        MazeConfig(start_point=(0.5, 0), end_point=(1, 1))

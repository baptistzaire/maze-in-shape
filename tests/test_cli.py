# tests/test_cli.py
"""
Tests for the Command Line Interface (src/cli.py).

Uses click.testing.CliRunner to invoke the CLI commands in-process and
mocks the main pipeline function to isolate CLI parsing logic.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

# Ensure the main function from cli.py can be imported
# Adjust path if necessary based on test execution context
try:
    from src.cli import main as cli_main
except ImportError:
    # If running pytest from the root directory, this might be needed
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.cli import main as cli_main

# --- Fixtures ---

@pytest.fixture
def runner():
    """Provides a CliRunner instance."""
    return CliRunner()

@pytest.fixture
def mock_pipeline():
    """Mocks the generate_maze_from_image function."""
    # Patch the function *where it's looked up* in the cli module
    with patch('src.cli.generate_maze_from_image') as mock_func:
        # Configure the mock to return True by default (simulating success)
        # The actual pipeline returns a PIL image, but for CLI testing,
        # we only care if it runs without raising an exception handled by the CLI.
        # Returning None or a simple value is fine.
        mock_func.return_value = None
        yield mock_func

@pytest.fixture
def temp_input_file(tmp_path):
    """Creates a temporary dummy input file."""
    input_file = tmp_path / "input.png"
    input_file.touch() # Create an empty file
    return str(input_file) # Return path as string

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provides a temporary directory path for output."""
    output_dir = tmp_path / "output"
    # Don't create it here, let the CLI/pipeline handle it if needed
    return str(output_dir)

# --- Test Cases ---

def test_cli_help(runner):
    """Test the --help option."""
    result = runner.invoke(cli_main, ['--help'])
    assert result.exit_code == 0
    assert "Usage: main [OPTIONS] INPUT_PATH OUTPUT_PATH" in result.output
    assert "--segmentation-method" in result.output
    assert "--cell-size" in result.output
    assert "--help" in result.output

def test_cli_basic_success(runner, mock_pipeline, temp_input_file, tmp_path):
    """Test basic successful execution with required arguments."""
    output_file = tmp_path / "output.png"
    args = [temp_input_file, str(output_file)]
    result = runner.invoke(cli_main, args)

    print(f"CLI Output:\n{result.output}")
    print(f"CLI Exception:\n{result.exception}")
    if result.exception:
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exc_info[2])

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}"
    assert f"Successfully generated and saved maze to: {output_file}" in result.output
    mock_pipeline.assert_called_once()
    # Check basic args passed to the mocked pipeline
    call_args, call_kwargs = mock_pipeline.call_args
    assert call_kwargs['image_source'] == temp_input_file
    assert call_kwargs['output_path'] == str(output_file)
    # Check default config values were passed in the dict
    assert call_kwargs['config_dict']['grid']['cell_size'] == 10 # Default
    assert call_kwargs['config_dict']['segmentation']['method'] == 'threshold' # Default
    assert call_kwargs['config_dict']['solve']['enabled'] is False # Default

def test_cli_with_options(runner, mock_pipeline, temp_input_file, tmp_path):
    """Test execution with various options specified."""
    output_file = tmp_path / "output_options.png"
    start_coords = "5,10"
    end_coords = "20,30"
    args = [
        temp_input_file,
        str(output_file),
        '--segmentation-method', 'rembg',
        '--cell-size', '5',
        '--maze-algo', 'prim',
        '--linewidth', '3',
        '--style', 'silhouette',
        '--start-point', start_coords,
        '--end-point', end_coords,
        '--show-solution'
    ]
    result = runner.invoke(cli_main, args)

    assert result.exit_code == 0
    assert f"Successfully generated and saved maze to: {output_file}" in result.output
    mock_pipeline.assert_called_once()

    # Check specific options passed to the mocked pipeline
    call_args, call_kwargs = mock_pipeline.call_args
    config = call_kwargs['config_dict']
    assert config['segmentation']['method'] == 'rembg'
    assert config['grid']['cell_size'] == 5
    assert config['maze']['algorithm'] == 'prim'
    assert config['rendering']['linewidth'] == 3
    assert config['rendering']['style'] == 'silhouette'
    assert config['maze']['start_point'] == (5, 10)
    assert config['maze']['end_point'] == (20, 30)
    assert config['solve']['enabled'] is True

def test_cli_missing_output_arg(runner, temp_input_file):
    """Test error when output path argument is missing."""
    args = [temp_input_file]
    result = runner.invoke(cli_main, args)
    assert result.exit_code != 0 # Should fail
    assert "Error: Missing argument 'OUTPUT_PATH'." in result.output

def test_cli_missing_input_arg(runner, tmp_path):
    """Test error when input path argument is missing."""
    output_file = tmp_path / "output.png"
    args = [str(output_file)] # Only provide output
    result = runner.invoke(cli_main, args)
    assert result.exit_code != 0 # Should fail
    # Click's behavior might vary slightly, check for general error message
    assert "Error: Missing argument" in result.output or "Error:" in result.output

def test_cli_invalid_input_path(runner, tmp_path):
    """Test error when input path does not exist."""
    non_existent_input = tmp_path / "non_existent.png"
    output_file = tmp_path / "output.png"
    args = [str(non_existent_input), str(output_file)]
    result = runner.invoke(cli_main, args)
    assert result.exit_code != 0 # Should fail
    # Check for key parts of the error message, avoiding exact path string comparison
    assert "Error: Invalid value for 'INPUT_PATH':" in result.output
    assert "File '" in result.output
    assert non_existent_input.name in result.output # Check filename part
    assert "' does not exist." in result.output

def test_cli_invalid_option_value(runner, temp_input_file, tmp_path):
    """Test error when an option has an invalid value (e.g., non-integer cell size)."""
    output_file = tmp_path / "output.png"
    args = [temp_input_file, str(output_file), '--cell-size', 'abc']
    result = runner.invoke(cli_main, args)
    assert result.exit_code != 0 # Should fail
    # Error message format depends on click version and type validation
    assert "Error: Invalid value for '--cell-size'" in result.output
    assert "'abc' is not a valid integer" in result.output or "'abc' is not a valid integer range" in result.output


def test_cli_invalid_coords_format(runner, temp_input_file, tmp_path):
    """Test error when coordinate format is wrong."""
    output_file = tmp_path / "output.png"
    args = [temp_input_file, str(output_file), '--start-point', '5_10'] # Invalid separator
    result = runner.invoke(cli_main, args)
    assert result.exit_code != 0 # Should fail
    # Adjust assertion to include the detail from the callback's exception
    assert "Error: Invalid value for '--start-point': Error parsing coordinates: Coordinates must be in 'row,col' format." in result.output

def test_cli_pipeline_error(runner, mock_pipeline, temp_input_file, tmp_path):
    """Test CLI handling when the mocked pipeline raises an error."""
    output_file = tmp_path / "output_error.png"
    error_message = "Simulated pipeline failure!"
    mock_pipeline.side_effect = ValueError(error_message) # Simulate a ValueError

    args = [temp_input_file, str(output_file)]
    result = runner.invoke(cli_main, args)

    assert result.exit_code != 0 # Should fail
    assert f"Pipeline Error: {error_message}" in result.output
    mock_pipeline.assert_called_once() # Ensure it was called despite the error

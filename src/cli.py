# src/cli.py
"""
Command Line Interface for the Maze-in-Shape Generator.

Uses click to parse arguments and configure the maze generation pipeline.
"""

import click
import sys
import os
from typing import Tuple, Optional

# Ensure the src directory is in the Python path
# This allows importing modules like src.config when running cli.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import the main pipeline function and potentially config defaults if needed later
    from src.main_pipeline import generate_maze_from_image
    # Import config defaults for showing help text accurately
    from src.config import MazeConfig
except ImportError as e:
    click.echo(f"Error importing project modules: {e}", err=True)
    click.echo("Please ensure you are running the script from the project root directory", err=True)
    click.echo("or that the 'src' directory is in your PYTHONPATH.", err=True)
    sys.exit(1)

# --- Helper Function for Coordinate Parsing ---
def parse_coords(ctx, param, value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Callback to parse 'row,col' string into (int, int) tuple."""
    if value is None:
        return None
    try:
        parts = value.split(',')
        if len(parts) != 2:
            raise click.BadParameter("Coordinates must be in 'row,col' format.")
        coords = (int(parts[0].strip()), int(parts[1].strip()))
        return coords
    except ValueError:
        raise click.BadParameter("Coordinates must be integers in 'row,col' format.")
    except Exception as e:
        raise click.BadParameter(f"Error parsing coordinates: {e}")

# --- Click Command Definition ---
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('output_path', type=click.Path(dir_okay=False, writable=True))
@click.option(
    '--segmentation-method',
    type=click.Choice(["threshold", "rembg"], case_sensitive=False), # TODO: Update choices as more methods are added
    default=MazeConfig.segmentation_method, # Default from MazeConfig
    show_default=True,
    help='Subject segmentation algorithm.'
)
@click.option(
    '--cell-size',
    type=click.IntRange(min=1),
    default=MazeConfig.cell_size,
    show_default=True,
    help='Approximate size (pixels) of each maze cell.'
)
@click.option(
    '--maze-algo',
    type=click.Choice(["dfs", "prim"], case_sensitive=False), # TODO: Update choices as more methods are added
    default=MazeConfig.maze_algorithm, # Default from MazeConfig
    show_default=True,
    help='Maze generation algorithm.'
)
@click.option(
    '--linewidth',
    type=click.IntRange(min=1),
    default=MazeConfig.linewidth,
    show_default=True,
    help='Thickness of maze walls (pixels).'
)
@click.option(
    '--style',
    type=click.Choice(["silhouette", "overlay"], case_sensitive=False), # TODO: Update choices as more methods are added
    default=MazeConfig.rendering_style, # Default from MazeConfig
    show_default=True,
    help='Rendering style.'
)
@click.option(
    '--start-point',
    type=click.STRING,
    callback=parse_coords,
    help="Manual start coordinates 'row,col'. If omitted, uses automatic selection."
)
@click.option(
    '--end-point',
    type=click.STRING,
    callback=parse_coords,
    help="Manual end coordinates 'row,col'. If omitted, uses automatic selection."
)
@click.option(
    '--show-solution',
    is_flag=True,
    default=False,
    help='Draw the solution path on the maze.'
)
@click.option(
    '--threshold-value', # Example of a method-specific parameter
    type=click.IntRange(min=0, max=255),
    default=MazeConfig.threshold_value,
    show_default=True,
    help='Threshold value (0-255) used for "threshold" segmentation.'
)
# TODO: Add --config option later if needed to load from YAML
# @click.option('--config', type=click.Path(exists=True, dir_okay=False), help='Path to YAML config file.')
def main(
    input_path: str,
    output_path: str,
    segmentation_method: str, # Click returns string
    cell_size: int,
    maze_algo: str, # Click returns string
    linewidth: int,
    style: str, # Click returns string
    start_point: Optional[Tuple[int, int]],
    end_point: Optional[Tuple[int, int]],
    show_solution: bool,
    threshold_value: int,
    # config_file: Optional[str], # For YAML config file
):
    """
    Generates a maze constrained within the shape of the main subject found in an INPUT_PATH image
    and saves it to OUTPUT_PATH.
    """
    click.echo(f"Starting maze generation for: {input_path}")

    # --- Configuration Setup ---
    # if config:
        # Load base config from YAML file
        # click.echo(f"Loading configuration from: {config}")
        # maze_config = load_config_from_yaml(config) # Need to implement this
        # Override with CLI arguments where provided (or decide on precedence)
        # ...
    # else: # If not loading from file
        # Create config dictionary directly from CLI arguments
    config_dict = {
        'segmentation': {
            'method': segmentation_method,
            'params': {
                # Add method-specific params here based on segmentation_method
                'threshold_value': threshold_value if segmentation_method == 'threshold' else None,
                # Add other params like 'model_name' for 'rembg' if needed
            },
            # 'init_params': {} # Add if segmenters need init params
        },
        'grid': {
            'cell_size': cell_size,
        },
        'maze': {
            'algorithm': maze_algo,
            'start_point': start_point,
            'end_point': end_point,
        },
        'solve': {
            'enabled': show_solution,
        },
        'rendering': {
            'style': style,
            'linewidth': linewidth,
            # Add color options if exposed via CLI later
        },
        # 'preprocessing': {} # Add if preprocessing options are added to CLI
    }
    # Clean up None values from segmentation params if not applicable
    if config_dict['segmentation']['params'].get('threshold_value') is None:
        del config_dict['segmentation']['params']['threshold_value']

    click.echo(f"Using configuration derived from CLI arguments: {config_dict}")

    # --- Run Pipeline ---
    try:
        click.echo("Processing image and generating maze...")
        # generate_maze_from_image handles saving internally if output_path is provided
        # It returns the PIL image, but we don't need it here unless we want to display it.
        # Success is indicated by the absence of exceptions.
        _ = generate_maze_from_image(
            image_source=input_path,
            config_dict=config_dict,
            output_path=output_path
        )
        # If we reach here, it succeeded
        click.echo(click.style(f"Successfully generated and saved maze to: {output_path}", fg='green'))

    except FileNotFoundError as e:
        click.echo(f"Error: Input file not found - {e}", err=True)
        sys.exit(1)
    except (ValueError, TypeError) as e:
        # Catch configuration, validation, or processing errors from the pipeline
        click.echo(f"Pipeline Error: {e}", err=True)
        sys.exit(1)
    except ImportError as e:
         click.echo(f"Error: Missing optional dependency for selected method? {e}", err=True)
         sys.exit(1)
    except Exception as e:
        # Catch-all for other pipeline errors
        click.echo(f"An unexpected error occurred during maze generation: {e}", err=True)
        # Consider adding traceback logging here for debugging
        # import traceback
        # click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

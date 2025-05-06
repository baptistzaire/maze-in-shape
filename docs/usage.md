# Usage Guide

This guide explains how to use the Maze-in-Shape Generator, both via the Command Line Interface (CLI) and the Python API.

## Command Line Interface (CLI)

*(Adapt this section based on your final CLI implementation using libraries like `argparse` or `click`)*

The primary way to use the tool is through the `cli.py` script.

**Basic Syntax:**

```bash
python src/cli.py --input <path/to/image.png> --output <path/to/maze.png> [OPTIONS]

Required Arguments:
--input <path>: Path to the input image file (e.g., images/cat.jpg).
--output <path>: Path where the generated maze image will be saved (e.g., output/cat_maze.png).
Common Options:
--segmentation-method <method>: Choose the subject segmentation algorithm.
Options: threshold, hsv, canny, kmeans, grabcut, rembg, unet, maskrcnn (Adjust based on implemented methods)
Default: rembg (Example default)
See Segmentation Options for details.
--cell-size <int>: The approximate size (in pixels of the original image) that each maze cell represents. Smaller values mean higher maze resolution and detail, but slower generation. (Default: 10)
--maze-algo <algorithm>: The maze generation algorithm.
Options: dfs, prim, kruskal, wilson (Adjust based on implemented methods)
Default: dfs
--linewidth <int>: Thickness of the maze walls in pixels. (Default: 2)
--style <style>: Rendering style.
Options: silhouette, overlay
Default: silhouette
--start-point <x,y>: Manually specify start coordinates (relative to grid).
--end-point <x,y>: Manually specify end coordinates (relative to grid). (If not specified, uses automatic farthest points).
--show-solution: If set, draws the solution path on the maze.
--config <path>: Path to a YAML configuration file with settings.
Examples:
Generate maze using default settings (rembg segmentation, dfs maze):
python src/cli.py --input data/examples/sample_cat.png --output output/cat_maze_default.png
Use code with caution.
Bash
Use GrabCut segmentation (requires interactive bbox or pre-defined rect) and Prim's algorithm:
python src/cli.py --input data/examples/sample_tree.png --output output/tree_maze_grabcut.png --segmentation-method grabcut --maze-algo prim --cell-size 15
Use code with caution.
Bash
(Note: GrabCut might need modifications for non-interactive use or a way to pass the bounding box via CLI)
Generate a high-resolution maze with thin lines and show the solution:
python src/cli.py --input data/examples/sample_complex.jpg --output output/complex_maze_solved.png --cell-size 5 --linewidth 1 --show-solution
Use code with caution.
Bash
Python API
You can integrate the maze generation pipeline into your own Python scripts.
(Adapt this section based on your final API design)
Core Usage:
from src.main_pipeline import generate_maze_from_image
from src.config import MazeConfig # Assuming a config class/dict

# 1. Configure the generation pipeline
config = MazeConfig(
    segmentation_method='rembg', # Or 'grabcut', 'threshold', 'unet', etc.
    segmentation_params={'model_name': 'u2net'}, # Params specific to method
    cell_size=10,
    maze_algorithm='dfs', # Or 'prim', etc.
    start_end_method='farthest', # Or 'manual'
    start_point=None, # Specify if start_end_method='manual'
    end_point=None,   # Specify if start_end_method='manual'
    render_style='silhouette', # Or 'overlay'
    line_width=2,
    output_format='png',
    show_solution=False
)

# Or load/modify default config
# config = load_default_config()
# config.cell_size = 5
# config.maze_algorithm = 'prim'

# 2. Specify input image path
input_image_path = "data/examples/sample_cat.png"

# 3. Run the pipeline
# This function would orchestrate all steps: load, segment, grid, maze, render
# It might return the maze image as a PIL/OpenCV object, or save directly
maze_image_object = generate_maze_from_image(input_image_path, config)

# 4. Process the result (e.g., save it)
if maze_image_object:
    output_path = "output/cat_maze_api.png"
    # Assuming the function returns a PIL Image object
    maze_image_object.save(output_path)
    print(f"Maze saved to {output_path}")
else:
    print("Maze generation failed.")

# You might also have access to intermediate results if the API provides it:
# results = generate_maze_from_image(..., return_intermediates=True)
# mask = results['mask']
# grid = results['grid']
# maze_data = results['maze_data']
# final_image = results['image']
Use code with caution.
Python
Key Components/Functions (Example):
generate_maze_from_image(image_path_or_object, config): The main entry point. Takes image source and configuration, returns the final maze image.
MazeConfig: A class or dictionary to hold all configuration parameters.
Potentially separate functions for each stage if you want finer control:
load_and_preprocess(image_path)
segment_subject(image, method, params)
create_grid(mask, cell_size)
generate_maze(grid, algorithm)
find_start_end(maze_data, grid, method)
render_maze(maze_data, grid, style, linewidth, show_solution)
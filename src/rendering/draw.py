"""
Functions for rendering the maze into an image.
"""

from typing import Dict, Tuple, Set, Optional, Union, List
from PIL import Image, ImageDraw, ImageOps

# Attempt to import solve_maze for example usage, handle if not found
try:
    from src.maze.solve import solve_maze
except ImportError:
    solve_maze = None # type: ignore

from src.maze.types import MazeData, Point # Import the newly defined types

# Define default colors (can be made configurable later)
WALL_COLOR = "black"
BACKGROUND_COLOR = "white" # Or (0, 0, 0, 0) for transparent
START_COLOR = "green"
END_COLOR = "red"
SOLUTION_COLOR = "blue"
SOLUTION_LINEWIDTH_FACTOR = 0.25 # Fraction of cell_size for solution line thickness
MARKER_RADIUS_FACTOR = 0.3 # Fraction of cell_size for marker radius
SHAPE_COLOR = (200, 200, 200, 255) # Default light grey for the shape background

def render_maze_silhouette(
    maze_data: MazeData,
    cell_size: int,
    linewidth: int = 1,
    start_marker_color: str = START_COLOR,
    end_marker_color: str = END_COLOR,
    wall_color: str = WALL_COLOR,
    bg_color: str = BACKGROUND_COLOR,
    solution_path: Optional[List[Point]] = None,
    solution_color: str = SOLUTION_COLOR,
) -> Image.Image:
    """
    Renders the maze walls onto a blank background (silhouette style).
    Optionally draws the solution path.

    Draws the walls based on the grid structure. Does not require the original
    shape mask, as it only visualizes the maze structure itself.

    Args:
        maze_data: A dictionary or object containing maze details:
            - width (int): Number of columns in the grid.
            - height (int): Number of rows in the grid.
            - walls (Set[Tuple[Point, Point]]): A set of pairs of adjacent
              cells ((r1, c1), (r2, c2)) that have a wall BETWEEN them.
            - start (Point): The (row, col) of the start cell.
            - end (Point): The (row, col) of the end cell.
            - grid_mask (Optional[Set[Point]]): Optional set of valid cells.
        cell_size: The size of each cell in pixels.
        linewidth: The thickness of the maze walls in pixels.
        start_marker_color: Color for the start marker.
        end_marker_color: Color for the end marker.
        wall_color: Color for the maze walls.
        bg_color: Background color of the image.
        solution_path: Optional list of (row, col) points for the solution.
        solution_color: Color for the solution path.

    Returns:
        A PIL Image object representing the rendered maze silhouette.
    """
    grid_height = maze_data['height']
    grid_width = maze_data['width']
    walls = maze_data['walls']
    start_point = maze_data['start']
    end_point = maze_data['end']
    # grid_mask = maze_data.get('grid_mask') # Use if needed for complex shapes

    img_width = grid_width * cell_size
    img_height = grid_height * cell_size

    # Create image with appropriate background
    # Use "RGBA" and (0,0,0,0) for transparency if needed
    image = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # --- Draw Walls ---
    # Iterate through all possible wall locations
    for r in range(grid_height):
        for c in range(grid_width):
            # Cell coordinates (top-left corner)
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = (c + 1) * cell_size
            y2 = (r + 1) * cell_size

            # Check for horizontal wall below (between (r, c) and (r+1, c))
            if r < grid_height - 1:
                p1 = (r, c)
                p2 = (r + 1, c)
                # Draw wall if the pair exists in the walls set (order doesn't matter)
                if (p1, p2) in walls or (p2, p1) in walls:
                    draw.line([(x1, y2), (x2, y2)], fill=wall_color, width=linewidth)

            # Check for vertical wall to the right (between (r, c) and (r, c+1))
            if c < grid_width - 1:
                p1 = (r, c)
                p2 = (r, c + 1)
                # Draw wall if the pair exists in the walls set
                if (p1, p2) in walls or (p2, p1) in walls:
                     draw.line([(x2, y1), (x2, y2)], fill=wall_color, width=linewidth)

    # --- Draw Outer Boundary Walls ---
    # Top boundary
    for c in range(grid_width):
        # Simplified: Draw top wall for all cells. Refine if using grid_mask
        draw.line([(c * cell_size, 0), ((c + 1) * cell_size, 0)], fill=wall_color, width=linewidth)
    # Bottom boundary
    for c in range(grid_width):
        # Simplified: Draw bottom wall for all cells. Refine if using grid_mask
        y_bottom = grid_height * cell_size
        draw.line([(c * cell_size, y_bottom), ((c + 1) * cell_size, y_bottom)], fill=wall_color, width=linewidth)
    # Left boundary
    for r in range(grid_height):
        # Simplified: Draw left wall for all cells. Refine if using grid_mask
        draw.line([(0, r * cell_size), (0, (r + 1) * cell_size)], fill=wall_color, width=linewidth)
    # Right boundary
    for r in range(grid_height):
        # Simplified: Draw right wall for all cells. Refine if using grid_mask
        x_right = grid_width * cell_size
        draw.line([(x_right, r * cell_size), (x_right, (r + 1) * cell_size)], fill=wall_color, width=linewidth)


    # --- Draw Start/End Markers ---
    marker_radius = int(cell_size * MARKER_RADIUS_FACTOR)

    # Start marker (centered in the cell)
    if start_point:
        sr, sc = start_point
        sx_center = int((sc + 0.5) * cell_size)
        sy_center = int((sr + 0.5) * cell_size)
        s_bbox = [
            (sx_center - marker_radius, sy_center - marker_radius),
            (sx_center + marker_radius, sy_center + marker_radius)
        ]
        draw.ellipse(s_bbox, fill=start_marker_color, outline=start_marker_color)

    # End marker (centered in the cell)
    if end_point:
        er, ec = end_point
        ex_center = int((ec + 0.5) * cell_size)
        ey_center = int((er + 0.5) * cell_size)
        e_bbox = [
            (ex_center - marker_radius, ey_center - marker_radius),
            (ex_center + marker_radius, ey_center + marker_radius)
        ]
        draw.ellipse(e_bbox, fill=end_marker_color, outline=end_marker_color)

    # Optional: Add anti-aliasing effect by rendering at 2x and resizing
    # upscale_factor = 2
    # hires_image = image.resize(
    #     (img_width * upscale_factor, img_height * upscale_factor),
    #     resample=Image.Resampling.NEAREST # Or BICUBIC for smoother lines
    # )
    # final_image = hires_image.resize(
    #     (img_width, img_height),
    #     resample=Image.Resampling.LANCZOS # High-quality downsampling
    # )
    # return final_image

    # --- Draw Solution Path ---
    if solution_path and len(solution_path) > 1:
        solution_linewidth = max(1, int(cell_size * SOLUTION_LINEWIDTH_FACTOR))
        # Draw lines between centers of consecutive cells in the path
        for i in range(len(solution_path) - 1):
            r1, c1 = solution_path[i]
            r2, c2 = solution_path[i+1]

            x1_center = int((c1 + 0.5) * cell_size)
            y1_center = int((r1 + 0.5) * cell_size)
            x2_center = int((c2 + 0.5) * cell_size)
            y2_center = int((r2 + 0.5) * cell_size)

            draw.line(
                [(x1_center, y1_center), (x2_center, y2_center)],
                fill=solution_color,
                width=solution_linewidth
            )

    return image


def render_maze_overlay(
    maze_data: MazeData,
    shape_mask: Image.Image,
    cell_size: int,
    linewidth: int = 1,
    start_marker_color: str = START_COLOR,
    end_marker_color: str = END_COLOR,
    wall_color: str = WALL_COLOR,
    shape_color: Union[Tuple[int, int, int, int], str] = SHAPE_COLOR,
    bg_color: Optional[Union[Tuple[int, int, int, int], str]] = None, # None for transparent BG
    solution_path: Optional[List[Point]] = None,
    solution_color: str = SOLUTION_COLOR,
) -> Image.Image:
    """
    Renders the maze walls overlaid onto the original shape mask.
    Optionally draws the solution path.

    Draws the maze walls only within the white areas of the provided mask.

    Args:
        maze_data: Maze details (width, height, walls, start, end).
        shape_mask: A PIL Image ('L' mode preferred) where white (255)
                    indicates the area where the maze should be drawn.
                    Must have dimensions compatible with grid_width * cell_size
                    and grid_height * cell_size.
        cell_size: The size of each cell in pixels.
        linewidth: The thickness of the maze walls in pixels.
        start_marker_color: Color for the start marker.
        end_marker_color: Color for the end marker.
        wall_color: Color for the maze walls.
        shape_color: Color to fill the shape defined by the mask. Can be RGBA.
        bg_color: Background color for the image outside the shape.
                  Use None or (0,0,0,0) for a transparent background.
        solution_path: Optional list of (row, col) points for the solution.
        solution_color: Color for the solution path.

    Returns:
        A PIL Image object representing the rendered maze overlaid on the shape.
    """
    grid_height = maze_data['height']
    grid_width = maze_data['width']
    walls = maze_data['walls']
    start_point = maze_data['start']
    end_point = maze_data['end']

    img_width = grid_width * cell_size
    img_height = grid_height * cell_size

    # Ensure mask matches expected dimensions
    if shape_mask.size != (img_width, img_height):
        # Option 1: Resize mask (might distort shape)
        # shape_mask = shape_mask.resize((img_width, img_height), Image.Resampling.NEAREST)
        # Option 2: Raise error
        raise ValueError(
            f"Shape mask size {shape_mask.size} does not match "
            f"calculated image size ({img_width}, {img_height})"
        )

    # Ensure mask is 'L' mode for easy processing
    if shape_mask.mode != 'L':
        shape_mask = shape_mask.convert('L')

    # Create the base image (RGBA for transparency)
    image = Image.new("RGBA", (img_width, img_height), color=bg_color or (0, 0, 0, 0))

    # Create the shape layer: fill 'shape_color' where mask is white
    shape_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
    # Invert mask because paste uses mask where 0 is transparent, 255 is opaque
    inverted_mask = ImageOps.invert(shape_mask)
    shape_layer.paste(shape_color, (0, 0), mask=shape_mask) # Paste color where mask is white

    # Composite shape layer onto the base image
    image.paste(shape_layer, (0, 0), shape_layer) # Use shape_layer's alpha

    # --- Draw Walls ---
    draw = ImageDraw.Draw(image) # Draw directly onto the composited image

    # Iterate through all possible wall locations
    for r in range(grid_height):
        for c in range(grid_width):
            # Cell center coordinates (for mask checking)
            center_x = int((c + 0.5) * cell_size)
            center_y = int((r + 0.5) * cell_size)

            # Check if the *center* of the current cell is inside the mask
            # Use a tolerance or check corners if center isn't sufficient
            is_cell_inside = False
            if 0 <= center_x < img_width and 0 <= center_y < img_height:
                 is_cell_inside = shape_mask.getpixel((center_x, center_y)) > 128 # Check if mask is white enough

            if not is_cell_inside:
                continue # Don't draw walls originating from outside the shape

            # Pixel coordinates for drawing lines
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = (c + 1) * cell_size
            y2 = (r + 1) * cell_size

            # Check for horizontal wall below (between (r, c) and (r+1, c))
            if r < grid_height - 1:
                p1 = (r, c)
                p2 = (r + 1, c)
                # Draw wall ONLY if it exists AND the neighbor cell is also inside
                neighbor_center_y = int((r + 1.5) * cell_size)
                is_neighbor_inside = False
                if 0 <= center_x < img_width and 0 <= neighbor_center_y < img_height:
                     is_neighbor_inside = shape_mask.getpixel((center_x, neighbor_center_y)) > 128

                if is_neighbor_inside and ((p1, p2) in walls or (p2, p1) in walls):
                    draw.line([(x1, y2), (x2, y2)], fill=wall_color, width=linewidth)

            # Check for vertical wall to the right (between (r, c) and (r, c+1))
            if c < grid_width - 1:
                p1 = (r, c)
                p2 = (r, c + 1)
                 # Draw wall ONLY if it exists AND the neighbor cell is also inside
                neighbor_center_x = int((c + 1.5) * cell_size)
                is_neighbor_inside = False
                if 0 <= neighbor_center_x < img_width and 0 <= center_y < img_height:
                     is_neighbor_inside = shape_mask.getpixel((neighbor_center_x, center_y)) > 128

                if is_neighbor_inside and ((p1, p2) in walls or (p2, p1) in walls):
                     draw.line([(x2, y1), (x2, y2)], fill=wall_color, width=linewidth)

    # --- Draw Outer Boundary Walls (Only along the mask edge) ---
    # This is more complex. A simpler approach is to rely on the internal walls
    # and the shape background providing the boundary visually.
    # For a hard boundary line, one could trace the contour of the mask
    # or draw walls for cells at the edge of the mask.
    # Let's omit explicit outer boundary drawing for now, assuming the shape provides it.


    # --- Draw Start/End Markers (only if they fall within the mask) ---
    marker_radius = int(cell_size * MARKER_RADIUS_FACTOR)

    # Start marker
    if start_point:
        sr, sc = start_point
        sx_center = int((sc + 0.5) * cell_size)
        sy_center = int((sr + 0.5) * cell_size)
        # Check if start point is inside the mask
        if 0 <= sx_center < img_width and 0 <= sy_center < img_height and \
           shape_mask.getpixel((sx_center, sy_center)) > 128:
            s_bbox = [
                (sx_center - marker_radius, sy_center - marker_radius),
                (sx_center + marker_radius, sy_center + marker_radius)
            ]
            draw.ellipse(s_bbox, fill=start_marker_color, outline=start_marker_color)

    # End marker
    if end_point:
        er, ec = end_point
        ex_center = int((ec + 0.5) * cell_size)
        ey_center = int((er + 0.5) * cell_size)
         # Check if end point is inside the mask
        if 0 <= ex_center < img_width and 0 <= ey_center < img_height and \
           shape_mask.getpixel((ex_center, ey_center)) > 128:
            e_bbox = [
                (ex_center - marker_radius, ey_center - marker_radius),
                (ex_center + marker_radius, ey_center + marker_radius)
            ]
            draw.ellipse(e_bbox, fill=end_marker_color, outline=end_marker_color)

    # --- Draw Solution Path (only if it falls within the mask) ---
    if solution_path and len(solution_path) > 1:
        solution_linewidth = max(1, int(cell_size * SOLUTION_LINEWIDTH_FACTOR))
        for i in range(len(solution_path) - 1):
            r1, c1 = solution_path[i]
            r2, c2 = solution_path[i+1]

            x1_center = int((c1 + 0.5) * cell_size)
            y1_center = int((r1 + 0.5) * cell_size)
            x2_center = int((c2 + 0.5) * cell_size)
            y2_center = int((r2 + 0.5) * cell_size)

            # Check if both points of the segment are roughly inside the mask
            p1_inside = 0 <= x1_center < img_width and 0 <= y1_center < img_height and \
                        shape_mask.getpixel((x1_center, y1_center)) > 128
            p2_inside = 0 <= x2_center < img_width and 0 <= y2_center < img_height and \
                        shape_mask.getpixel((x2_center, y2_center)) > 128

            if p1_inside and p2_inside:
                draw.line(
                    [(x1_center, y1_center), (x2_center, y2_center)],
                    fill=solution_color,
                    width=solution_linewidth
                )

    return image


# Example Usage (requires a MazeData structure)
if __name__ == '__main__':
    # Create a simple dummy MazeData for testing
    # Note: The wall set contains pairs of cells *with* a wall between them.
    dummy_maze: MazeData = {
        'width': 5,
        'height': 4,
        'start': (0, 0),
        'end': (3, 4), # Initially out of bounds (width is 5, max index is 4)
        'walls': {
            # Horizontal walls (between rows)
            ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 3), (1, 3)), ((0, 4), (1, 4)),
            ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((1, 3), (2, 3)),
            ((2, 0), (3, 0)), ((2, 2), (3, 2)), ((2, 4), (3, 4)),
            # Vertical walls (between columns)
            ((0, 0), (0, 1)), ((0, 2), (0, 3)), ((0, 3), (0, 4)),
            ((1, 0), (1, 1)), ((1, 1), (1, 2)), # ((1, 4), (1, 5)) removed (out of bounds)
            ((2, 1), (2, 2)), ((2, 3), (2, 4)),
            ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (3, 3)), ((3, 3), (3, 4)),
        },
        'grid_mask': None # Assume full rectangle
    }

    # Adjust end point if it's out of bounds
    if dummy_maze['end'][1] >= dummy_maze['width']:
        print(f"Adjusting end point {dummy_maze['end']} as it's outside grid width {dummy_maze['width']}")
        dummy_maze['end'] = (dummy_maze['end'][0], dummy_maze['width'] - 1)

    # Define rendering parameters
    cell_s = 30
    line_w = 2

    # --- Solve the dummy maze (if solver is available) ---
    solution = None
    if solve_maze:
        print("Solving dummy maze for rendering test...")
        solution = solve_maze(dummy_maze)
        if solution:
            print(f"Found solution with {len(solution)} steps.")
        else:
            print("Solver did not find a solution for the dummy maze.")
    else:
        print("Skipping solution solving/rendering (solve_maze not imported).")


    # --- Test Silhouette ---
    print("\nRendering dummy silhouette maze (with solution if found)...")
    rendered_silhouette = render_maze_silhouette(
        dummy_maze, cell_s, line_w, solution_path=solution
    )
    try:
        output_path_sil = "silhouette_maze_solution_test.png"
        rendered_silhouette.save(output_path_sil)
        print(f"Saved test silhouette maze to {output_path_sil}")
    except Exception as e:
        print(f"Error saving silhouette image: {e}")

    # --- Test Overlay ---
    print("\nRendering dummy overlay maze (with solution if found)...")
    # Create a dummy mask (e.g., a circle) matching the expected size
    mask_width = dummy_maze['width'] * cell_s
    mask_height = dummy_maze['height'] * cell_s
    dummy_mask = Image.new('L', (mask_width, mask_height), 0) # Black background
    mask_draw = ImageDraw.Draw(dummy_mask)
    # Draw a white filled circle as the mask shape
    center_x, center_y = mask_width // 2, mask_height // 2
    radius = min(mask_width, mask_height) // 2 - 10 # Make it slightly smaller than bounds
    mask_draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=255 # White
    )

    try:
        rendered_overlay = render_maze_overlay(
            dummy_maze, dummy_mask, cell_s, line_w,
            bg_color=(50, 50, 50, 255), # Dark grey BG
            solution_path=solution
        )
        output_path_over = "overlay_maze_solution_test.png"
        rendered_overlay.save(output_path_over)
        print(f"Saved test overlay maze to {output_path_over}")
        # rendered_overlay.show() # Optionally display the image
    except ValueError as ve:
         print(f"ValueError during overlay rendering: {ve}")
    except Exception as e:
        print(f"Error saving overlay image: {e}")

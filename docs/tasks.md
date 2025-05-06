# Project Tasks: Maze-in-Shape Generator

This document outlines the tasks required to build the Maze-in-Shape Generator application. Each task includes context explaining its purpose, importance, and relevant considerations drawn from the project documentation (`pipeline.md`, `segmentation.md`, etc.). Tasks are organized by phase and module. Check off tasks as they are completed.

## Phase 0: Project Setup & Configuration

**Goal:** Establish a clean, organized, and configurable foundation for the project. This ensures consistency, maintainability, and ease of dependency management from the start.

*   [ ] **Task 0.1: Create Base Directory Structure**
    *   **Context:** A standard structure (`src/`, `tests/`, etc.) makes the codebase predictable. As outlined in the project structure doc, this separates source code, tests, documentation, configuration, and data.
    *   **Action:** Create folders: `maze-in-shape/`, `src/`, `tests/`, `docs/`, `config/`, `data/examples/`, `models/`.
*   [ ] **Task 0.2: Initialize Git & `.gitignore`**
    *   **Context:** Version control (Git) is essential. `.gitignore` prevents committing unnecessary files like virtual environments (`venv/`), Python cache (`__pycache__/`), large model files (`models/*` unless small), and OS/editor artifacts (`.DS_Store`, `.vscode/`).
    *   **Action:** Run `git init`. Create `.gitignore` listing these common ignores.
*   [ ] **Task 0.3: Add `LICENSE` File**
    *   **Context:** Defines usage rights. Choice (e.g., MIT, Apache 2.0) must be compatible with dependencies. Notably, if using `rembg` directly and distributing, its GPL/LGPL license has implications – consider this when choosing the project license or how `rembg` is used (e.g., optional dependency, subprocess).
    *   **Action:** Choose license, add full text to `LICENSE`. Verify dependency compatibility.
*   [ ] **Task 0.4: Create Initial `README.md`**
    *   **Context:** The project's front page. Needs title, description, core features (automatic shape extraction, configurable maze generation), installation steps, and quick start example as outlined in the documentation structure.
    *   **Action:** Create `README.md` with title, description, and placeholders for Features, Installation, Usage, Contributing, License. Add a placeholder for a compelling visual example later.
*   [ ] **Task 0.5: Create Initial `requirements.txt`**
    *   **Context:** Lists dependencies for reproducible builds (`pip install -r`). Core libraries identified in `pipeline.md` include `numpy`, `opencv-python`, `Pillow`, `matplotlib`. Add others (`rembg`, `torch`, `torchvision`, `scikit-image`, `segmentation-models-pytorch`, `detectron2` etc.) as features requiring them are implemented. Specify versions for stability.
    *   **Action:** Create `requirements.txt` listing initial core libraries with version specifiers (e.g., `numpy>=1.20`).
*   [ ] **Task 0.6: Define Basic Configuration Structure (`src/config.py`)**
    *   **Context:** Centralizes tunable parameters identified throughout `pipeline.md` (like `cell_size`, segmentation method choice, maze algorithm choice, `linewidth`, rendering style) to avoid hardcoding and allow easy modification via CLI/UI/files later.
    *   **Action:** Define a structure (dataclass, Pydantic model recommended for validation, or dict) in `src/config.py` holding default values for key pipeline parameters.
*   [ ] **Task 0.7: Implement Basic Configuration Loading**
    *   **Context:** Makes the default settings accessible to the application logic. This forms the basis for overriding defaults later.
    *   **Action:** Implement logic for the application to load and use the default configuration object/structure.

## Phase 1: Image Input & Pre-processing (`src/image_utils/`)

**Goal:** Reliably load user images (common formats like JPEG, PNG) and prepare them (standardize format like RGB/RGBA, optionally resize for performance) for consistent processing by downstream modules.

*   [ ] **Task 1.1: Implement Image Loading (`io.py::load_image`)**
    *   **Context:** Handles image input (`cv2.imread` or `Image.open`). Must manage potential file errors and ensure images are converted to a consistent format (e.g., RGB or RGBA NumPy array) as expected by segmentation modules.
    *   **Action:** Implement loading, error handling (FileNotFoundError), and color space standardization (e.g., to RGB).
*   [ ] **Task 1.2: Implement Image Saving (`io.py::save_image`)**
    *   **Context:** Needed to output the final maze image in user-specified formats (PNG, JPG).
    *   **Action:** Implement saving using Pillow/OpenCV, supporting common formats.
*   [ ] **Task 1.3: Implement Image Resizing (`preprocess.py::resize_image`)**
    *   **Context:** As noted in `pipeline.md`, large images significantly impact performance, especially for DL segmentation and dense maze generation. Resizing (e.g., based on max dimension) is a crucial pre-processing step. Interpolation choice (`cv2.INTER_AREA`, `cv2.INTER_LINEAR`, etc.) affects quality.
    *   **Action:** Implement resizing function, allowing configurable target size or scale factor and interpolation method.
*   [ ] **Task 1.4: Implement Color Space Conversion (`preprocess.py::convert_color_space`)**
    *   **Context:** Needed because different segmentation algorithms perform best in different spaces (e.g., `hsv` method requires HSV, `threshold` often uses Grayscale).
    *   **Action:** Implement conversions (e.g., `cv2.cvtColor`) between RGB, Grayscale, HSV.
*   [ ] **Task 1.5: Write Unit Tests (`tests/test_image_utils.py`)**
    *   **Context:** Verifies robustness of loading different formats, handling errors, correct resizing output dimensions, and accurate color space conversions.
    *   **Action:** Write tests covering these aspects.

## Phase 2: Subject Segmentation (`src/segmentation/`)

**Goal:** Isolate the main subject(s), producing a binary mask defining the maze shape. This is highly challenging for arbitrary images. Offering multiple methods allows users to choose the best trade-off (speed vs. accuracy vs. ease of use) for their input, as detailed in `segmentation.md`.

*   [ ] **Task 2.1: Define Base Class `BaseSegmenter` (`base.py`)**
    *   **Context:** Creates a uniform interface (`segment(self, image, **params) -> np.ndarray`) for all methods. This polymorphism is key to the factory pattern (Task 2.9) and allows the main pipeline to easily switch between segmentation strategies based on configuration.
    *   **Action:** Define the abstract base class with the abstract `segment` method ensuring it returns a binary mask (uint8 NumPy array, 0/255).
*   [ ] **Task 2.2: Implement `ThresholdSegmenter` (`thresholding.py`)**
    *   **Context:** Simple, fast method for high-contrast cases. As per `segmentation.md`, suitable for silhouettes but fails on complex scenes. Needs parameters like `threshold_value` (or use `cv2.THRESH_OTSU`) for global, or `adaptive_method`, `block_size`, `C` for adaptive.
    *   **Action:** Implement using `cv2.threshold`/`cv2.adaptiveThreshold`. Expose relevant parameters. Inherit from `BaseSegmenter`.
*   [ ] **Task 2.3: Implement `HSVSegmenter` (`hsv_slicing.py`)**
    *   **Context:** Isolates based on color range (using `cv2.inRange` in HSV space). Good for specific known colors but sensitive to lighting, as noted in `segmentation.md`. Requires `lower_hsv` and `upper_hsv` bounds as parameters.
    *   **Action:** Implement HSV conversion and range slicing. Parameterize bounds. Inherit from `BaseSegmenter`.
*   [ ] **Task 2.4: Implement `CannyContourSegmenter` (`contours.py`)**
    *   **Context:** Uses edge detection (`cv2.Canny`) and contour finding (`cv2.findContours`). Works for clear boundaries but breaks on texture (`segmentation.md`). Requires logic to select the main contour (e.g., `cv2.contourArea`) and fill it (`cv2.drawContours`). Needs `threshold1`, `threshold2` for Canny as parameters.
    *   **Action:** Implement the Canny -> findContours -> filter -> fill workflow. Parameterize thresholds. Inherit from `BaseSegmenter`.
*   [ ] **Task 2.5: Implement `KMeansSegmenter` (`kmeans.py`)**
    *   **Context:** Unsupervised color clustering (`cv2.kmeans`). Can separate distinct color groups but `segmentation.md` notes it's computationally heavier and needs heuristics/user input to map clusters to foreground/background. `num_clusters` (K) is the key parameter.
    *   **Action:** Implement k-means on pixel colors. Parameterize K. Add logic to select foreground cluster(s). Inherit from `BaseSegmenter`.
*   [ ] **Task 2.6: Implement `GrabCutSegmenter` (`grabcut.py`)**
    *   **Context:** Powerful interactive/semi-interactive method (`cv2.grabCut`) noted in `segmentation.md` for good quality on complex textures, but critically requires an initial bounding box (`roi_rect`) parameter for good results. Slower due to iterative nature (`iterations` parameter).
    *   **Action:** Implement using `cv2.grabCut`. Parameterize `iterations`. Define clear strategy for providing `roi_rect` (config, auto-detection heuristic?, requires thought). Inherit from `BaseSegmenter`.
*   [ ] **Task 2.7: Implement `RembgSegmenter` (`rembg_wrapper.py`)**
    *   **Context:** Uses U²-Net via the `rembg` library. Highlighted in `segmentation.md` as excellent for single salient object removal and easy to use (`pip install rembg`). Models (`u2net`, `u2netp`, etc.) are downloaded automatically. *Reiterate license consideration (GPL/LGPL).*
    *   **Action:** Create wrapper for `rembg.remove()`. Parameterize `model_name`. Handle import errors. Inherit from `BaseSegmenter`.
*   [ ] **Task 2.8: Implement `DeepLearningSegmenter` (`deep_learning.py`)**
    *   **Context:** Wrapper for advanced models (U-Net, DeepLab, Mask R-CNN from `torchvision`, `segmentation_models.pytorch`, `detectron2` etc.). Offers highest accuracy but requires significant setup (`segmentation.md`). Must handle model loading, pre/post-processing specific to each architecture. GPU highly recommended for speed.
    *   **Action:** Create flexible wrapper inheriting from `BaseSegmenter`.
        *   **Task 2.8.1 (Load Models):** Load specified model (`model_path`, `model_type`) using PyTorch/TF. Handle device placement (CPU/GPU).
        *   **Task 2.8.2 (Pre-process):** Implement model-specific normalization/resizing.
        *   **Task 2.8.3 (Inference):** Run prediction.
        *   **Task 2.8.4 (Post-process):** Convert output (logits, probabilities, instance masks) to a single binary mask. For Mask R-CNN, handle merging multiple instance masks if needed. Address potential "noisy masks" mentioned in `segmentation.md`.
*   [ ] **Task 2.9: Implement Segmentation Factory**
    *   **Context:** Decouples the main pipeline from specific segmentation implementations. Takes a string identifier (e.g., "rembg", "threshold") from config and returns the corresponding initialized `BaseSegmenter` object.
    *   **Action:** Implement factory function or logic (e.g., dictionary mapping names to classes) in `segmentation/__init__.py` or `main_pipeline.py`.
*   [ ] **Task 2.10: Write Unit Tests (`tests/test_segmentation.py`)**
    *   **Context:** Verifies each method works, handles parameters, returns correct mask format. Mocks essential for testing DL/rembg wrappers without heavy dependencies or long runtimes.
    *   **Action:** Test each segmenter class, mocking external libs/models. Test factory logic.

## Phase 3: Grid Creation (`src/grid/`)

**Goal:** Translate the continuous binary mask into a discrete grid for maze algorithms. The `cell_size` parameter directly controls the maze resolution vs. shape fidelity trade-off mentioned in `pipeline.md`.

*   [ ] **Task 3.1: Implement `create_grid_from_mask` (`creation.py`)**
    *   **Context:** Maps the mask onto a grid. Key decision: how to classify cells on the boundary? Checking the cell's center pixel is simplest; checking the majority of pixels within the cell's area in the mask is more robust but slower. `pipeline.md` also mentions polygon conversion (`Shapely`, `rasterio`) as an alternative for geometric control.
    *   **Action:** Implement the function taking mask and `cell_size`. Return 2D boolean/int NumPy array. Choose and implement cell classification logic (start with center-point check).
*   [ ] **Task 3.2: (Optional) Define `MazeGrid` Class (`creation.py`)**
    *   **Context:** Encapsulation can simplify maze generation logic by providing grid-aware helpers (checking bounds, passability, finding neighbors).
    *   **Action:** Create class holding grid array and methods like `is_passable(r, c)`, `get_neighbors(r, c)`, `height`, `width`.
*   [ ] **Task 3.3: Write Unit Tests (`tests/test_grid.py`)**
    *   **Context:** Ensures correct grid dimensions and cell classification for various inputs.
    *   **Action:** Test with simple geometric masks (square, circle), different `cell_size` values, verify output shape and cell states. Test edge cases (empty/full masks).

## Phase 4: Maze Generation (`src/maze/`)

**Goal:** Implement algorithms to carve paths within the passable grid cells, creating a connected maze respecting the shape. Offer variety as different algorithms produce visually distinct maze styles (long corridors vs. many short branches), as noted in `pipeline.md`.

*   [ ] **Task 4.1: Define Maze Data Structure**
    *   **Context:** How to represent the "carved" maze? A set of tuples `((r1, c1), (r2, c2))` representing removed walls between adjacent cells is common and flexible for rendering.
    *   **Action:** Define the structure (e.g., `Set[Tuple[Tuple[int, int], Tuple[int, int]]]`).
*   [ ] **Task 4.2: Define Base Class `BaseMazeGenerator` (`base_generator.py`)**
    *   **Context:** Common interface (`generate(self, grid) -> MazeData`) for all maze algorithms (DFS, Prim, etc.) enabling polymorphic use via the factory pattern (Task 4.9).
    *   **Action:** Define abstract base class with `generate` method returning the chosen maze data structure.
*   [ ] **Task 4.3: Implement `DFSMazeGenerator` (`dfs.py`)**
    *   **Context:** Recursive backtracker. `pipeline.md` notes it produces long, winding corridors and is easy to implement with a stack. Must be adapted to only move between *passable* grid cells. Generates a "perfect" maze (no loops).
    *   **Action:** Implement DFS backtracker logic. Ensure only valid moves on the grid. Inherit from `BaseMazeGenerator`.
*   [ ] **Task 4.4: Implement `PrimMazeGenerator` (`prim.py`)**
    *   **Context:** Grows the maze from a start cell. `pipeline.md` notes it tends towards more uniform trees with many short branches compared to DFS. Also produces a perfect maze. Requires managing a frontier set.
    *   **Action:** Implement Prim's algorithm adapted for grids. Manage frontier, add cells/remove walls respecting passable areas. Inherit from `BaseMazeGenerator`.
*   [ ] **Task 4.5: Implement `KruskalMazeGenerator` (`kruskal.py`)**
    *   **Context:** Builds a random spanning tree using Union-Find. `pipeline.md` notes it has uniform randomness but can be slower on grids. Produces a perfect maze. Requires a Union-Find data structure implementation.
    *   **Action:** Implement Kruskal's using randomized edges between passable cells and Union-Find. Inherit from `BaseMazeGenerator`.
*   [ ] **Task 4.6: Implement `WilsonMazeGenerator` (`wilson.py`)**
    *   **Context:** Generates uniformly random spanning trees using loop-erased random walks. `pipeline.md` notes it's unbiased but often slower and complex. Produces a perfect maze.
    *   **Action:** Implement Wilson's algorithm tracking visited cells and performing random walks on passable cells, erasing loops. Inherit from `BaseMazeGenerator`.
*   [ ] **Task 4.7: Implement Maze Utility Functions (`utils.py`)**
    *   **Context:** Shared logic like finding valid *passable* neighbors for a given cell `(r, c)` within grid bounds, needed by most generators.
    *   **Action:** Implement `get_valid_neighbors(grid, r, c)` and potentially others.
*   [ ] **Task 4.8: (Optional) Implement Loop Creation (`add_loops`)**
    *   **Context:** To create non-perfect mazes with multiple solutions/shortcuts, as discussed in `pipeline.md`. This involves removing extra walls after a perfect maze is generated.
    *   **Action:** Implement logic to randomly select and remove a percentage of existing internal walls, ensuring connectivity isn't broken.
*   [ ] **Task 4.9: Implement Maze Factory**
    *   **Context:** Selects and returns an initialized maze generator instance (`DFS`, `Prim`, etc.) based on the `maze_algorithm` string specified in the configuration.
    *   **Action:** Implement factory function or logic.
*   [ ] **Task 4.10: Write Unit Tests (`tests/test_maze_generation.py`)**
    *   **Context:** Verifies each algorithm generates a connected maze covering all passable cells within simple test grids. Checks properties like perfectness (if applicable).
    *   **Action:** Test each generator. Verify connectivity (e.g., BFS/DFS traversal reaches all passable cells). Test factory logic.

## Phase 5: Start/End Point Selection (`src/maze/start_end.py`)

**Goal:** Define logical entry/exit points for the maze, typically on the boundary. Method choice affects perceived difficulty.

*   [ ] **Task 5.1: Implement `find_boundary_cells`**
    *   **Context:** Needed to identify candidate locations for start/end points if they must lie on the shape's edge.
    *   **Action:** Implement function taking the grid, return list of `(r, c)` coordinates of passable cells bordering impassable ones or grid edges.
*   [ ] **Task 5.2: Implement `find_farthest_points`**
    *   **Context:** Automatic method described in `pipeline.md` using two BFS passes on the *generated maze graph* (not just the grid) to find the maze diameter. Often yields the longest possible solution path, maximizing difficulty.
    *   **Action:** Implement the BFS-twice algorithm using the `maze_data` structure to respect carved passages. Consider using boundary cells as start/end candidates.
*   [ ] **Task 5.3: Implement Manual Start/End Logic**
    *   **Context:** Allows user override via config. Requires validating that specified `(r, c)` points are within grid bounds and are *passable* cells.
    *   **Action:** Add logic to read start/end from config and validate against the grid.
*   [ ] **Task 5.4: Integrate Start/End Selection**
    *   **Context:** Pipeline needs to invoke either the automatic ('farthest') or 'manual' method based on configuration.
    *   **Action:** Implement logic to select and run the appropriate method.
*   [ ] **Task 5.5: Write Unit Tests (`tests/test_maze_start_end.py`)**
    *   **Context:** Verifies boundary finding logic and the farthest points algorithm on known simple maze structures.
    *   **Action:** Test `find_boundary_cells`. Test `find_farthest_points` with predefined simple mazes where the farthest points are known. Test validation of manual points.

## Phase 6: Rendering & Output (`src/rendering/`)

**Goal:** Create the final visual output image, drawing the maze walls, start/end points, and optional solution path according to configured style and parameters.

*   [ ] **Task 6.1: Implement `render_maze_silhouette` (`draw.py`)**
    *   **Context:** Renders maze walls on a blank background, clipped by the shape, as described in `pipeline.md`. Needs `linewidth` parameter. Anti-aliasing or drawing at higher resolution and downsampling improves quality.
    *   **Action:** Use Pillow Draw or OpenCV lines. Iterate potential walls; draw if wall exists in `maze_data`. Scale positions by `cell_size`. Draw start/end markers (e.g., arrows, colored cells).
*   [ ] **Task 6.2: Implement `render_maze_overlay` (`draw.py`)**
    *   **Context:** Shows maze within the context of the original shape (drawn from mask or using original image), as described in `pipeline.md`. Requires careful compositing.
    *   **Action:** Draw base shape (filled mask or processed original image). Overlay maze walls. Handle transparency/colors appropriately.
*   [ ] **Task 6.3: (Optional) Implement Maze Solving (`solve.py::solve_maze`)**
    *   **Context:** Finds the solution path using graph search (BFS is suitable for shortest path) on the maze graph (passable cells connected by removed walls). Needed if `show_solution` is enabled.
    *   **Action:** Implement BFS/DFS from start to end using `maze_data`. Return path as list of `(r, c)`.
*   [ ] **Task 6.4: (Optional) Implement Solution Path Rendering**
    *   **Context:** Draws the path found by the solver onto the maze image, typically in a contrasting color.
    *   **Action:** Modify rendering functions to take the solution path and draw it (e.g., line connecting cell centers).
*   [ ] **Task 6.5: Integrate Rendering Style Selection**
    *   **Context:** Allows user choice between `silhouette` and `overlay` via configuration.
    *   **Action:** Add logic to call the correct rendering function based on `render_style` config.
*   [ ] **Task 6.6: Ensure Final Image Output**
    *   **Context:** Rendering must produce an image object (e.g., PIL Image, NumPy array) ready for saving via `io.py::save_image`.
    *   **Action:** Ensure render functions return the final image object.
*   [ ] **Task 6.7: Write Unit Tests (`tests/test_rendering.py`)**
    *   **Context:** Verifies rendering functions produce images without crashing and include expected elements (walls, start/end). Pixel-perfect tests are brittle; focus on successful execution and output type/size.
    *   **Action:** Test both styles. Test solution path drawing if implemented. Check output format.

## Phase 7: Integration & Pipeline (`src/main_pipeline.py`)

**Goal:** Connect all independent modules into a single, configurable workflow orchestrated by a main function. This function handles data flow between stages and manages errors.

*   [ ] **Task 7.1: Implement Main Pipeline Function/Class**
    *   **Context:** The central coordinator (`generate_maze_from_image`). Takes image source and config, returns final maze image.
    *   **Action:** Define the main function/class signature and basic structure in `main_pipeline.py`.
*   [ ] **Task 7.2: Orchestrate Pipeline Calls**
    *   **Context:** Implements the sequence described in `pipeline.md`: Load -> Preprocess -> Segment (using factory) -> Create Grid -> Generate Maze (using factory) -> Find Start/End -> Render -> Return/Save. Manages passing data (image, mask, grid, maze_data) between stages.
    *   **Action:** Write the core sequential logic, using configuration to select methods and parameters.
*   [ ] **Task 7.3: Implement Robust Error Handling**
    *   **Context:** Real-world use involves potential failures (bad input, segmentation failure, empty mask, etc.). The pipeline needs `try...except` blocks and checks on intermediate results to fail gracefully and provide informative feedback.
    *   **Action:** Add error handling and checks (e.g., `if mask is None: raise SegmentationError`). Log errors. Return `None` or raise specific exceptions on failure.
*   [ ] **Task 7.4: Write Integration Tests (`tests/test_pipeline_integration.py`)**
    *   **Context:** Crucial for verifying the entire system works end-to-end with different configurations, catching issues that unit tests might miss.
    *   **Action:** Write tests calling `generate_maze_from_image` with various sample images and config combinations (different segmenters, maze algos). Assert that an image is successfully produced.

## Phase 8: CLI / UI (Optional)

**Goal:** Provide user-friendly interfaces (command-line or graphical) to access the pipeline functionality.

*   [ ] **Task 8.1: Implement Command Line Interface (`src/cli.py`)**
    *   **Context:** Uses `argparse` or `click` (recommended for richer features) to expose pipeline configuration options as CLI arguments (`--input`, `--output`, `--cell-size`, `--segmentation-method`, etc.) as described in `usage.md`.
    *   **Action:** Build CLI parser, map arguments to config object, call main pipeline function, print feedback/errors.
*   [ ] **Task 8.2: Write CLI Tests (`tests/test_cli.py`)**
    *   **Context:** Verifies argument parsing and basic CLI execution flow.
    *   **Action:** Use `subprocess` or test runners' capabilities to simulate CLI calls and check outputs/exit codes.
*   [ ] **Task 8.3: (Optional) Develop Graphical User Interface (`src/ui/`)**
    *   **Context:** Increases accessibility. Libraries like `Streamlit` or `Gradio` allow rapid development for simple UIs; `Flask`/`Django` or `PyQt`/`Tkinter` offer more customization, as noted in `pipeline.md`.
    *   **Action:** Build UI with image upload, configuration widgets (sliders, dropdowns), execution button, and result display area.

## Phase 9: Documentation & Finalization

**Goal:** Ensure the project is well-documented, polished, and ready for users or further development.

*   [ ] **Task 9.1: Write/Review All Docstrings**
    *   **Context:** Essential for code maintainability and understanding internal APIs.
    *   **Action:** Ensure comprehensive docstrings (Google/NumPy style) for all public elements.
*   [ ] **Task 9.2: Complete/Update `docs/` Files**
    *   **Context:** Update user/developer guides (`index.md`, `usage.md`, `segmentation.md`, `development.md`) and the core technical spec (`pipeline.md`) to reflect the final implementation.
    *   **Action:** Review and finalize all markdown documentation.
*   [ ] **Task 9.3: (Optional) Set up Auto-API Documentation**
    *   **Context:** Tools like Sphinx with `napoleon` and `autodoc` extensions can generate professional API references directly from docstrings, keeping docs synchronized with code.
    *   **Action:** Configure Sphinx to build HTML docs for `src/`. Link from `index.md`.
*   [ ] **Task 9.4: Finalize `README.md`**
    *   **Context:** Needs complete installation steps, clear usage examples (CLI/API), feature list, and ideally the compelling visual example.
    *   **Action:** Update `README.md` with final details and examples. Add the overview image.
*   [ ] **Task 9.5: Ensure `LICENSE` Compliance**
    *   **Context:** Final check that project license choice is valid given all included dependencies.
    *   **Action:** Review dependency licenses again.
*   [ ] **Task 9.6: Final Code Review & Cleanup**
    *   **Context:** Catch remaining bugs, inconsistencies, or areas for improvement. Ensure adherence to code style.
    *   **Action:** Perform code review. Run `black`, `flake8`. Ensure all tests pass (`pytest`). Remove dead code.
*   [ ] **Task 9.7: Tag Release Version**
    *   **Context:** Creates an official, identifiable version of the software.
    *   **Action:** Use `git tag v1.0.0` (or appropriate version) and push the tag.
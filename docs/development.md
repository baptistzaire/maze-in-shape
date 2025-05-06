# Developer Guide

This guide provides instructions for setting up the development environment, running tests, understanding the project structure, and contributing to the Maze-in-Shape Generator.

## Development Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd maze-in-shape
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:** Install core requirements and development tools (like `pytest`, `flake8`, `black`). You might have a separate `requirements-dev.txt`.
    ```bash
    pip install -r requirements.txt
    # If you have a dev requirements file:
    # pip install -r requirements-dev.txt
    ```

4.  **Install ML Models (If applicable):** Some deep learning segmentation methods require downloading pre-trained model weights.
    *   For `rembg`: Models are typically downloaded automatically on first use.
    *   For PyTorch/TensorFlow models: Follow instructions specific to the chosen models (e.g., from TorchVision or TensorFlow Hub documentation). You might store them in the `models/` directory (ensure this is in `.gitignore` if large).

5.  **Verify Setup:** Try running the CLI or a basic API example to ensure dependencies are correctly installed.

## Running Tests

We use `pytest` for testing. Tests are located in the `tests/` directory.

*   **Run all tests:**
    ```bash
    pytest
    ```
*   **Run specific test file:**
    ```bash
    pytest tests/test_maze_generation.py
    ```
*   **Run tests with coverage:**
    ```bash
    pytest --cov=src tests/
    ```

Please ensure all tests pass before submitting contributions. Write new tests for any new features or bug fixes.

## Code Style

*   Follow **PEP 8** guidelines for Python code.
*   Use **Black** for code formatting. Run `black src/ tests/` before committing.
*   Use **Flake8** for linting. Run `flake8 src/ tests/` to check for style issues.
*   Use clear variable and function names.
*   Add docstrings to modules, classes, and functions.

## Project Architecture Overview

The project is organized into several main components within the `src/` directory:

*   **`main_pipeline.py`:** Orchestrates the entire maze generation process, calling functions/classes from other modules in sequence.
*   **`image_utils/`:** Handles image loading, saving, and basic pre-processing (resizing, color conversion).
*   **`segmentation/`:** Contains modules for different subject segmentation algorithms. A base class (`base.py`) likely defines a common interface (`segment(image, **params) -> mask`).
*   **`grid/`:** Responsible for converting the binary mask from segmentation into a 2D grid suitable for maze generation.
*   **`maze/`:** Implements maze generation algorithms (DFS, Prim, etc.), potentially using a base class. Also includes logic for start/end point selection (`start_end.py`).
*   **`rendering/`:** Takes the generated maze data and grid, and draws the final output image according to specified styles.
*   **`cli.py`:** Provides the command-line interface using `argparse` or `click`.
*   **`ui/`:** (Optional) Contains code for a graphical user interface (e.g., Flask web app, Streamlit dashboard).
*   **`config.py` / `config/`:** Defines or loads configuration settings.

The design emphasizes modularity, allowing different implementations (e.g., segmentation methods, maze algorithms) to be swapped or added more easily by adhering to common interfaces.

## Contributing

We welcome contributions! Please follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/issue-description`.
3.  **Make your changes.** Ensure you add relevant tests and documentation.
4.  **Run tests and linters** (`pytest`, `black`, `flake8`) to ensure code quality.
5.  **Commit your changes** with clear commit messages.
6.  **Push your branch** to your fork: `git push origin feature/your-feature-name`.
7.  **Open a Pull Request** against the main repository's `main` (or `develop`) branch. Describe your changes clearly in the PR.

Please refer to the `CONTRIBUTING.md` file *(if one exists)* for more detailed guidelines.

## Building Documentation

*(Add instructions here if using a documentation generator like Sphinx)*

Example using Sphinx:
```bash
cd docs
make html
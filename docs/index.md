# Maze-in-Shape Generator

Welcome to the documentation for the Maze-in-Shape Generator!

This project allows you to take an image, automatically identify the main subject, and generate a solvable maze that fits precisely within the shape of that subject.

![Placeholder Image: Example Input Image -> Output Maze](_static/example_overview.png)
*(Replace with an actual example image)*

## Features

*   **Automatic Shape Extraction:** Uses various image segmentation techniques (from simple thresholding to advanced Deep Learning) to find the subject's outline.
*   **Custom Shape Mazes:** Generates standard maze structures (like perfect mazes using DFS, Prim's, etc.) constrained within the extracted shape.
*   **Configurable:** Control segmentation method, maze complexity (cell size), maze generation algorithm, and output style.
*   **Multiple Output Formats:** Save the generated maze as an image file (PNG, JPG).
*   **Extensible:** Designed with modular components for easier addition of new algorithms or techniques.

## Quick Links

*   **[Installation Guide](README.md#installation)** *(Link to main README section)*
*   **[Usage Guide](usage.md)** (CLI and API)
*   **[Pipeline Details](pipeline.md)** (Technical breakdown of the generation process)
*   **[Segmentation Options](segmentation.md)** (Choosing the right subject extraction method)
*   **[Developer Guide](development.md)** (Contributing and project architecture)
*   **[API Reference](api/index.html)** *(Link to auto-generated API docs if using Sphinx/etc.)*


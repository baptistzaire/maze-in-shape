# Segmentation Options

Subject segmentation is a critical step in the Maze-in-Shape pipeline. It determines the exact boundary within which the maze will be generated. Choosing the right method depends heavily on the input image characteristics, desired accuracy, and computational resources.

This page provides a deeper dive into the available methods and guidance on selecting one.

## Comparison Summary

*(Insert the Segmentation Method table from pipeline.md again here for quick reference)*

| Segmentation Method         | Type             | Output             | Pros                                                                            | Cons                                                                                    | Tools/Models                                       |
| :-------------------------- | :--------------- | :----------------- | :------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------- | :------------------------------------------------- |
| ... (table content) ...     | ...              | ...                | ...                                                                             | ...                                                                                     | ...                                                |

## Detailed Method Descriptions

Here's more information on each method, including typical parameters and use cases:

1.  **Global/Adaptive Thresholding (`threshold`)**
    *   **Best For:** High-contrast images where the subject is clearly distinct from the background (e.g., black object on white background, silhouettes).
    *   **Parameters:**
        *   `threshold_value` (Global): The pixel intensity cutoff (0-255). Requires manual tuning or automatic methods like Otsu's thresholding.
        *   `adaptive_method` (Adaptive): `MEAN_C` or `GAUSSIAN_C`. Better for varying lighting but needs `block_size` and `C` (constant subtracted) parameters.
    *   **Pros:** Extremely fast, simple.
    *   **Cons:** Fails completely with complex backgrounds or subjects with internal color variation.

2.  **HSV/Color Range Slicing (`hsv`)**
    *   **Best For:** Isolating objects of a specific, consistent color range that differs from the background.
    *   **Parameters:** Lower and upper bounds for Hue, Saturation, and Value (`lower_hsv`, `upper_hsv`). Requires tuning these ranges.
    *   **Pros:** Fast, good for specific color targeting.
    *   **Cons:** Highly sensitive to lighting changes, shadows, and color variations within the object. Doesn't work well for multi-colored subjects.

3.  **Edge Detection + Contours (`canny`)**
    *   **Best For:** Objects with well-defined, continuous edges against a relatively clean background.
    *   **Parameters:** Canny edge detector thresholds (`threshold1`, `threshold2`). May require morphological operations (dilation, erosion) to close gaps in contours before filling. Selection of the "correct" contour might be needed (e.g., largest area).
    *   **Pros:** Good at capturing sharp geometric boundaries.
    *   **Cons:** Fails on textured objects/backgrounds, broken edges lead to incomplete masks.

4.  **K-Means Clustering (`kmeans`)**
    *   **Best For:** Images where the subject and background can be reasonably separated into distinct color groups. Unsupervised approach.
    *   **Parameters:** `num_clusters` (K): The number of color clusters to create. Choosing K and identifying which cluster(s) represent the foreground requires heuristics or user input.
    *   **Pros:** Can sometimes separate objects thresholding fails on.
    *   **Cons:** Computationally more expensive than thresholding, sensitive to the choice of K, doesn't understand "objects" only color similarity.

5.  **GrabCut (`grabcut`)**
    *   **Best For:** Extracting a single, primary foreground object when texture and color are complex but somewhat distinct from the background. Works best with user guidance.
    *   **Parameters:**
        *   `roi_rect`: A bounding box `(x, y, width, height)` drawn *around* the subject. Crucial for good results. (May need UI or heuristic for automatic use).
        *   `iterations`: Number of refinement iterations (e.g., 5).
    *   **Pros:** High-quality segmentation for single objects, handles texture well.
    *   **Cons:** Requires initialization (bounding box), slower due to iterative nature, typically isolates only one connected component well.

6.  **Deep Learning - Semantic (e.g., `unet`, `deeplab`)**
    *   **Best For:** General-purpose segmentation when objects belong to known classes (person, car, dog, etc., depending on the pre-trained model) or for foreground/background separation using models trained for that purpose.
    *   **Parameters:** `model_name` or `model_path`. May need configuration for input normalization specific to the model.
    *   **Pros:** Very accurate pixel-level masks, robust to variations.
    *   **Cons:** Requires downloading large pre-trained models, significantly slower on CPU (GPU highly recommended), limited by the classes the model was trained on. Might merge all instances of a class (e.g., all people) into one mask.
    *   **Setup:** Requires installing PyTorch/TensorFlow and relevant model libraries (`segmentation_models.pytorch`, `torchvision`, etc.). Download model weights.

7.  **Deep Learning - Instance (e.g., `maskrcnn`)**
    *   **Best For:** Scenes with multiple objects, potentially overlapping. Identifies and masks each instance separately.
    *   **Parameters:** `model_name` or `model_path`, `confidence_threshold`.
    *   **Pros:** Handles multiple objects, provides individual masks.
    *   **Cons:** Heavy models (CPU slow, GPU needed), masks can sometimes be less precise at boundaries ("background bleed"), limited by training classes. May need post-processing to combine masks if treating all subjects as one shape.
    *   **Setup:** Requires ML framework, libraries (Detectron2, TorchVision), model download.

8.  **Deep Learning - Salient Object (`rembg`)**
    *   **Best For:** Extracting the single most prominent foreground object from potentially complex backgrounds. Designed specifically for background removal.
    *   **Parameters:** `model_name` (e.g., `u2net`, `u2netp`, `silueta`).
    *   **Pros:** Excellent quality for single-object extraction, easy to use via `rembg` library, relatively fast (especially on GPU).
    *   **Cons:** Primarily designed for one main object; might ignore smaller secondary subjects.
    *   **Setup:** `pip install rembg`. Models are downloaded automatically on first use. *Note `rembg` license (GPL/LGPL) implications if distributing.*

## Choosing a Method

*   **Simple Contrast:** Start with `threshold` or `hsv`.
*   **Clear Edges:** Try `canny`.
*   **Distinct Colors (Unsupervised):** Consider `kmeans`.
*   **One Complex Object (Interactive/Semi-Auto):** `grabcut` is strong if you can provide a bounding box.
*   **Best General Automatic (Single Subject):** `rembg` is often the easiest and most effective starting point for high-quality single-subject extraction.
*   **Best General Automatic (Known Classes/Multiple Objects):** `unet`/`deeplab` (semantic) or `maskrcnn` (instance) if you need class awareness or multiple object handling, provided you have the computational resources and necessary models.

Experimentation is often necessary. Consider providing a preview of the generated mask in your application to help users choose or fine-tune parameters.
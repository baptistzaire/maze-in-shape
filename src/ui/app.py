# src/ui/app.py
"""
Streamlit Graphical User Interface for the Maze-in-Shape Generator.

Allows users to upload an image, configure generation parameters via widgets,
and view the resulting maze image.
"""

import streamlit as st
from PIL import Image
import numpy as np
import io
import sys
import os
from pathlib import Path

# Ensure the src directory is in the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from src.main_pipeline import generate_maze_from_image
    from src.config import MazeConfig # To access defaults and type hints if needed
except ImportError as e:
    st.error(f"Error importing project modules: {e}. Ensure the app is run from the project root or 'src' is in PYTHONPATH.")
    st.stop() # Stop execution if imports fail

# --- Page Configuration ---
st.set_page_config(page_title="Maze-in-Shape Generator", layout="wide")
st.title("🖼️ Maze-in-Shape Generator")
st.write("Upload an image and configure the parameters below to generate a maze within the main subject's shape.")

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")

# --- Image Upload ---
uploaded_file = st.sidebar.file_uploader("1. Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"])

# --- Configuration Widgets ---
# Get defaults from MazeConfig
cfg_defaults = MazeConfig()

st.sidebar.subheader("2. Segmentation")
# TODO: Dynamically get choices from config/factory later if possible
segmentation_method = st.sidebar.selectbox(
    "Method",
    options=["threshold", "rembg"], # Based on current MazeConfig literals
    index=["threshold", "rembg"].index(cfg_defaults.segmentation_method),
    help="Algorithm to separate the main subject from the background."
)

# Conditional parameters
thresholding_method_param = None
threshold_value_param = None
if segmentation_method == 'threshold':
    thresholding_method_param = st.sidebar.radio(
        "Thresholding Type",
        options=['global', 'adaptive'],
        index=0, # Default to 'global'
        help="'global': Single threshold value. 'adaptive': Threshold varies across image."
    )
    threshold_value_param = st.sidebar.slider(
        "Threshold Value (for 'global')",
        min_value=0, max_value=255,
        value=cfg_defaults.threshold_value, # Use default from config
        help="Pixel intensity value (0-255) used for 'global' thresholding."
    )
    # TODO: Add widgets for adaptive thresholding parameters if needed (block size, C)

# Add more conditional params for other methods (e.g., rembg model) if needed

st.sidebar.subheader("3. Maze Generation")
cell_size = st.sidebar.slider(
    "Cell Size",
    min_value=2, max_value=50,
    value=cfg_defaults.cell_size,
    step=1,
    help="Approximate size (pixels) of each maze cell. Smaller = higher detail, slower generation."
)
maze_algorithm = st.sidebar.selectbox(
    "Algorithm",
    options=["dfs", "prim"], # Based on current MazeConfig literals
    index=["dfs", "prim"].index(cfg_defaults.maze_algorithm),
    help="Algorithm used to carve the maze paths."
)

st.sidebar.subheader("4. Rendering")
rendering_style = st.sidebar.radio(
    "Style",
    options=["silhouette", "overlay"], # Based on current MazeConfig literals
    index=["silhouette", "overlay"].index(cfg_defaults.rendering_style),
    help="'Silhouette': Maze walls only. 'Overlay': Maze overlaid on the shape."
)
linewidth = st.sidebar.slider(
    "Line Width",
    min_value=1, max_value=10,
    value=cfg_defaults.linewidth,
    step=1,
    help="Thickness of the maze walls in pixels."
)
show_solution = st.sidebar.checkbox(
    "Show Solution Path",
    value=False,
    help="Draw the path from the start to the end of the maze."
)

# --- Execution Button ---
st.sidebar.divider()
generate_button = st.sidebar.button("🚀 Generate Maze", type="primary", use_container_width=True, disabled=(uploaded_file is None))

# --- Main Area for Input/Output Display ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    if uploaded_file is not None:
        try:
            # Read image using PIL
            input_image_pil = Image.open(uploaded_file).convert('RGB')
            st.image(input_image_pil, caption="Uploaded Image", use_column_width=True)
            # Convert PIL Image to NumPy array (RGB) for the pipeline
            input_image_np = np.array(input_image_pil)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            input_image_np = None # Ensure it's None if loading fails
            generate_button = False # Disable button if image load fails
    else:
        st.info("Upload an image using the sidebar to begin.")
        input_image_np = None # Ensure it's None if no image uploaded

with col2:
    st.subheader("Generated Maze")
    output_placeholder = st.empty() # Placeholder for the output image/message

# --- Pipeline Execution Logic ---
if generate_button and input_image_np is not None:
    output_placeholder.info("⏳ Generating maze, please wait...")

    # Construct config_dict from widget values
    config_dict = {
        'segmentation': {
            'method': segmentation_method,
            'params': {}, # Add specific params below
            # 'init_params': {}
        },
        'grid': {
            'cell_size': cell_size,
        },
        'maze': {
            'algorithm': maze_algorithm,
            # Start/End points are auto-selected by default in pipeline
            'start_point': None,
            'end_point': None,
        },
        'solve': {
            'enabled': show_solution,
        },
        'rendering': {
            'style': rendering_style,
            'linewidth': linewidth,
        },
        # 'preprocessing': {} # No preprocessing options in UI yet
    }

    # Add conditional segmentation parameters
    if segmentation_method == 'threshold':
        if thresholding_method_param:
             config_dict['segmentation']['params']['method'] = thresholding_method_param
        if threshold_value_param is not None:
             config_dict['segmentation']['params']['threshold_value'] = threshold_value_param
        # Add adaptive params if implemented:
        # config_dict['segmentation']['params']['adaptive_method'] = ...
        # config_dict['segmentation']['params']['block_size'] = ...
        # config_dict['segmentation']['params']['C'] = ...

    # Add params for other methods here if UI elements are added

    st.sidebar.write("Running with configuration:")
    st.sidebar.json(config_dict) # Show the config being used

    try:
        with st.spinner('Processing...'):
            # Call the main pipeline function
            # Pass the NumPy array directly
            # output_path is None, so the function returns the PIL image
            result_image_pil = generate_maze_from_image(
                image_source=input_image_np,
                config_dict=config_dict,
                output_path=None # Don't save, just return the image object
            )

        if result_image_pil:
            output_placeholder.image(result_image_pil, caption="Generated Maze", use_column_width=True)
            st.balloons()

            # Add download button
            buf = io.BytesIO()
            result_image_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Maze Image",
                data=byte_im,
                file_name="generated_maze.png",
                mime="image/png"
            )
        else:
            # This case might not be reachable if pipeline raises errors instead of returning None
            output_placeholder.error("Maze generation failed to produce an image.")

    except (FileNotFoundError, ValueError, TypeError, ImportError, RuntimeError) as e:
        output_placeholder.error(f"An error occurred during maze generation:\n\n{e}")
        st.exception(e) # Show traceback in UI for debugging
    except Exception as e:
        output_placeholder.error(f"An unexpected error occurred:\n\n{e}")
        st.exception(e) # Show traceback in UI for debugging

elif not uploaded_file:
     output_placeholder.info("Upload an image and click 'Generate Maze'.")
elif not generate_button:
    # If image is loaded but button not clicked yet
     output_placeholder.info("Configure parameters and click 'Generate Maze'.")

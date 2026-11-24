# Monocular Endoscopy Depth Estimation for AR Overlay

A deep-learning mini project that estimates a depth map from a single monocular endoscopy image and creates a colorized depth visualization and augmented-reality (AR) overlay.

## Features

- Upload an endoscopy image (`.png`, `.jpg`, or `.jpeg`) through a Streamlit interface.
- Predict a per-pixel depth map using a custom PyTorch encoder-decoder model.
- Visualize the predicted depth map using OpenCV's JET color map.
- Blend the colorized depth map with the input image to create an AR-style overlay.
- Display minimum, maximum, and average predicted depth values.
- Download the generated depth map and AR overlay.

## Model

The model is implemented in `model.py` and is named `EndoscopyDepthModel`.

- ResNet18 encoder.
- Convolutional decoder blocks with skip connections.
- Batch normalization and ReLU activations in decoder blocks.
- Sigmoid output layer for normalized depth predictions.

The training notebook uses paired RGB images and depth maps. It applies ImageNet normalization to RGB inputs and scales depth targets by 100. Invalid depth pixels are excluded using a validity mask during training/evaluation. The notebook records evaluation metrics including Absolute Relative Error (Abs Rel), RMSE, and Delta thresholds.

## Repository structure

```text
.
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── depth_estimation.ipynb
├── outputs/
│   ├── original_image.png
│   ├── depth_map.png
│   └── ar_overlay.png
└── checkpoints/
    └── README.md
```

The filenames under `outputs/` are examples; use the actual names of your demonstration images.

## Dataset

The training notebook reads two dataset archives named `trans_t1_a.zip` and `trans_t2_b.zip`. It pairs RGB files (`*_color.png`) with corresponding depth files (`*_depth.tiff`).

The notebook uses the `trans_t2_b` sequence for training/validation (72 training pairs and 31 validation pairs) and the `trans_t1_a` sequence for testing (61 pairs).

**Dataset source:** (https://drive.google.com/file/d/1AaMbvQRaKRxyz3jmjda5HtwGQSZi4QOB/view?usp=sharing) trans_t2_b
(https://drive.google.com/file/d/1CE5KEvw5gq5Qj1u4XYfZLVwtivvfmabs/view?usp=sharing) trans_t1_a

## Setup

1. Clone or download this repository.
2. Create and activate a Python virtual environment (recommended).
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Obtain the trained checkpoint `best_depth_model.pth` from the project's model-hosting link and place it at:

   ```text
   checkpoints/best_depth_model.pth
   ```
   https://drive.google.com/drive/folders/1lh57xrUrL4XLgPKLjPLh6kCGqsARdnvE?usp=sharing

## Run the application

From the repository root, run:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a supported endoscopy image, and view/download the predicted depth map and AR overlay.

## Trained model download

**Checkpoint filename:** `best_depth_model.pth`

After downloading, make sure the file is located at `checkpoints/best_depth_model.pth`. The application expects the checkpoint to contain a `model_state_dict` entry.

## Notebook

The training and experimentation notebook is available in `notebooks/depth_estimation.ipynb`. It documents dataset inspection, RGB/depth pairing, preprocessing, model training, evaluation, visualizations, and a Gradio demonstration.

Some notebook cells use Google Colab and Google Drive paths. Update those paths if running outside the original Colab environment.

## Limitations and intended use

This is an academic image-processing/deep-learning project and a visualization prototype. The generated depth map and AR overlay are model predictions, not verified clinical measurements. This project is not intended for diagnosis, surgical guidance, or other clinical use.

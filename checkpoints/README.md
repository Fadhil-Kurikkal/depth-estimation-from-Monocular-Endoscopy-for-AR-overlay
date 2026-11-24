
# Trained Model Checkpoint

This folder contains instructions for downloading the trained depth estimation model.

## Model Details

- **Model:** EndoscopyDepthModel
- **Architecture:** ResNet18 Encoder + Custom CNN Decoder
- **Framework:** PyTorch
- **Checkpoint:** best_depth_model.pth
- **Format:** .pth

## Download Model

The trained model checkpoint is hosted externally because its file size exceeds GitHub's regular file upload limit.

**Download Link:** https://drive.google.com/file/d/1JaNySNpAO4mD8RF7q5OQU8NA32sEOnrK/view?usp=sharing
## Installation

1. Download `best_depth_model.pth`.
2. Place the downloaded file inside this folder.
3. Ensure the folder structure is:

```text
checkpoints/
└── best_depth_model.pth
```

The application uses this checkpoint to perform depth estimation.

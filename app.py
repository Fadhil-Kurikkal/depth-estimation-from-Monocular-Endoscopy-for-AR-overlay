
import streamlit as st
import torch
import numpy as np
import cv2

from PIL import Image
from model import EndoscopyDepthModel


# Application configuration
st.set_page_config(
    page_title="Endoscopy Depth Estimation",
    page_icon="🔬",
    layout="wide"
)

st.title("Endoscopy Depth Estimation for AR Overlay")

st.write(
    "Upload a monocular endoscopy image to estimate its depth "
    "and generate an augmented-reality depth overlay."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "checkpoints/best_depth_model.pth"


# Load trained model
@st.cache_resource
def load_model():

    model = EndoscopyDepthModel().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


try:
    model = load_model()
    st.success("Best trained model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


st.write("Device:", DEVICE)


# Image upload
uploaded_file = st.file_uploader(
    "Upload Endoscopy Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image = np.array(image)

    # Resize image
    resized = cv2.resize(
        image,
        (320, 256),
        interpolation=cv2.INTER_AREA
    )

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std

    # Convert to tensor
    input_tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0).to(DEVICE)

    # Predict depth
    with torch.no_grad():
        prediction = model(input_tensor)

    # Convert depth to millimeters
    depth_mm = prediction[0, 0].cpu().numpy() * 100.0
    # Calculate depth statistics
    min_depth = float(np.min(depth_mm))
    max_depth = float(np.max(depth_mm))
    avg_depth = float(np.mean(depth_mm))

    # Generate colorized depth map
    depth_uint8 = np.clip(
        depth_mm / 100.0 * 255,
        0, 255
    ).astype(np.uint8)

    depth_colored = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_JET
    )

    depth_colored = cv2.cvtColor(
        depth_colored,
        cv2.COLOR_BGR2RGB
    )

    # Generate AR overlay
    overlay = cv2.addWeighted(
        resized,
        0.6,
        depth_colored,
        0.4,
        0
    )

    # Display results
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(resized, caption="Original Image",
                 use_container_width=True)

    with col2:
        st.image(depth_colored, caption="Predicted Depth Map",
                 use_container_width=True)

    with col3:
        st.image(overlay, caption="AR Depth Overlay",
                 use_container_width=True)

    # Display depth statistics
    st.subheader("Depth Statistics")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric("Minimum Depth", f"{min_depth:.2f} mm")

    with stat2:
        st.metric("Maximum Depth", f"{max_depth:.2f} mm")

    with stat3:
        st.metric("Average Depth", f"{avg_depth:.2f} mm")
    st.success("Depth estimation completed!")

    # Download predicted depth
    st.download_button(
        "Download Depth Map",
        data=cv2.imencode(
            ".png",
            cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
        )[1].tobytes(),
        file_name="predicted_depth.png",
        mime="image/png"
    )

    # Download overlay
    st.download_button(
        "Download AR Overlay",
        data=cv2.imencode(
            ".png",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )[1].tobytes(),
        file_name="ar_depth_overlay.png",
        mime="image/png"
    )

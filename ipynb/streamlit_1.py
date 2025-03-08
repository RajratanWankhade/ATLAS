import streamlit as st
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Spacing,
    ScaleIntensity,
    ResizeWithPadOrCrop,
    ToTensor
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import Activations
import matplotlib.pyplot as plt
import io

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # Load the trained model weights
    model_path = r"C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\results\results_again\best_metric_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Preprocessing function
def preprocess_volume(volume):
    transforms = Compose([
        EnsureChannelFirst(),
        Spacing(pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
        ScaleIntensity(minv=0.0, maxv=1.0),
        ResizeWithPadOrCrop(spatial_size=(448, 352, 64)),
        ToTensor(),
    ])
    return transforms(volume)

# Inference function
def infer_volume(model, device, volume_tensor):
    with torch.no_grad():
        roi_size = (448, 352, 64)
        sw_batch_size = 4
        output = sliding_window_inference(volume_tensor.unsqueeze(0), roi_size, sw_batch_size, model)

        # Apply sigmoid activation and thresholding
        sigmoid_activation = Activations(sigmoid=True)
        output = sigmoid_activation(output)
        output = output > 0.53  # Adjust threshold if necessary
        return output.squeeze(0).cpu().numpy()

# Streamlit interface
st.title("3D Liver Segmentation Tool")
st.write("Upload a 3D medical image file (e.g., `.nii.gz`), and the model will segment the liver.")

# File uploader for 3D volumes
uploaded_file = st.file_uploader("Choose a 3D medical image file", type=["nii", "nii.gz"], key="3d_file_uploader")

import tempfile

#uploaded_file = st.file_uploader("Choose a 3D medical image file", type=["nii", "nii.gz"], key="3d_file_uploader")
uploaded_file = st.file_uploader("Choose a 3D medical image file", type=["nii", "nii.gz"], key="unique_3d_uploader_key")

if uploaded_file is not None:
    st.subheader("Uploaded Volume")
    
    """
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        # Load the NIfTI file using nibabel
        volume = nib.load(temp_file_path).get_fdata()
        st.write(f"Volume shape: {volume.shape}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
        """

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        # Load the NIfTI file using nibabel
        volume = nib.load(temp_file_path).get_fdata()
        st.write(f"Volume shape: {volume.shape}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()    

    
    # Preview a middle slice
    st.subheader("Preview: Middle Slice")
    middle_slice = volume[..., volume.shape[-1] // 2]
    st.image(middle_slice, caption="Middle Slice", use_column_width=True, clamp=True)

    # Load the model
    model, device = load_model()

    # Preprocess the volume
    st.subheader("Preprocessing the volume...")
    preprocessed_volume = preprocess_volume(volume).to(device)

    # Perform inference
    st.subheader("Running 3D Segmentation...")
    segmentation_result = infer_volume(model, device, preprocessed_volume)

    # Visualize slices
    st.subheader("Visualize Slices")
    slice_index = st.slider("Choose slice index", 0, volume.shape[-1] - 1, volume.shape[-1] // 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(volume[..., slice_index], cmap="gray")
    ax[0].set_title("Input Slice")
    ax[1].imshow(segmentation_result[1, ..., slice_index], cmap="viridis")  # Liver channel
    ax[1].set_title("Segmented Slice")
    st.pyplot(fig)

    # Optionally save the segmentation
    save_option = st.checkbox("Save segmented volume as NIfTI?")
    if save_option:
        output_file = "segmented_volume.nii.gz"
        nib.save(nib.Nifti1Image(segmentation_result[1], np.eye(4)), output_file)
        st.success(f"Segmented volume saved as: {output_file}")

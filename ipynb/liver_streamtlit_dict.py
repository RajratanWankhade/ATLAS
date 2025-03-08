import streamlit as st
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstD,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import Activations
import matplotlib.pyplot as plt
import tempfile

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # Assuming 2 output channels (liver and background)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    model_path = r"C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\results\results_again\best_metric_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Define preprocessing transforms
def get_inference_transforms():
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-50, a_max=1250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(448, 352, 64)),
        ToTensord(keys=["image"]),
    ])
    return transforms

# Inference function
def infer_volume(model, device, volume_tensor):
    with torch.no_grad():
        roi_size = (448, 352, 64)
        sw_batch_size = 4
        output = sliding_window_inference(volume_tensor.unsqueeze(0), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        output = sigmoid_activation(output)
        output = output > 0.53  # Adjust threshold if needed
        return output.squeeze(0).cpu().numpy()

# Streamlit UI
st.title("3D Liver Segmentation Tool")
st.write("Upload a 3D medical image file (`.nii` or `.nii.gz`), and the model will segment the liver.")

uploaded_file = st.file_uploader("Choose a 3D medical image file", type=["nii", "nii.gz"], key="file_uploader_key")

if uploaded_file is not None:
    st.subheader("Uploaded Volume")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        # Prepare dictionary for preprocessing
        data_dict = {"image": temp_file_path}

        # Load preprocessing transforms
        transforms = get_inference_transforms()
        preprocessed_dict = transforms(data_dict)

        # Extract preprocessed volume
        preprocessed_volume = preprocessed_dict["image"]
        st.write(f"Preprocessed volume shape: {preprocessed_volume.shape}")

        # Display slice 30
        slice_30 = preprocessed_volume[0, :, :, 30].numpy()  # Slice 30 along the depth dimension
        st.subheader("Preview: Slice 30")
        st.image(slice_30, caption="Slice 30 (Preprocessed)", use_container_width=True, clamp=True)

        # Load model
        model, device = load_model()

        st.subheader("Running 3D Segmentation...")
        segmentation_result = infer_volume(model, device, preprocessed_volume.to(device))

        st.subheader("Visualize Results for Slice 30")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(slice_30, cmap="gray")
        ax[0].set_title("Input Slice 30")
        ax[1].imshow(segmentation_result[1, :, :, 30], cmap="viridis")  # Segmented slice
        ax[1].set_title("Segmented Slice 30")
        st.pyplot(fig)

        # Option to save the segmented volume
        save_option = st.checkbox("Save segmented volume as NIfTI?")
        if save_option:
            output_file = "segmented_volume.nii.gz"
            nib.save(nib.Nifti1Image(segmentation_result[1], np.eye(4)), output_file)
            st.success(f"Segmented volume saved as: {output_file}")

    except Exception as e:
        st.error(f"Error during preprocessing or inference: {e}")

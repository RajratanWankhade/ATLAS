import os
import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.transforms import Activations
from monai.data import DataLoader, Dataset
from preporcess import prepare
from monai.networks.nets import UNet
from monai.networks.layers import Norm

# Paths
data_dir = r"C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train"
model_dir = r'C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\results\results_again'   
output_dir = r"C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\results\output"     # Directory to save prediction results
os.makedirs(output_dir, exist_ok=True)

# Model Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Load trained model weights
model_path = os.path.join(model_dir, "best_metric_model.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# Preprocess and prepare data
_, test_loader = prepare(data_dir, cache=True)  

# Parameters for inference
sw_batch_size = 4
roi_size = (448, 352, 64)
slice_idx = 30  # The slice number to visualize

# Inference and Visualization
print("Starting inference on test dataset...")
for idx, test_patient in enumerate(test_loader):
    with torch.no_grad():
        t_volume = test_patient['vol'].to(device)
        t_segmentation = test_patient['seg']  # Ground truth segmentation (optional)

        # Perform sliding window inference
        test_outputs = sliding_window_inference(t_volume, roi_size, sw_batch_size, model)

        # Apply sigmoid activation and threshold
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53  # Binary thresholding at 0.53

        # Visualize and save slice number 30
        print(f"Processing Test Patient {idx + 1}/{len(test_loader)}")
        plt.figure(f"Patient {idx + 1}", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"Patient {idx + 1}: Input Image - Slice {slice_idx}")
        plt.imshow(t_volume[0, 0, :, :, slice_idx].cpu(), cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"Patient {idx + 1}: Ground Truth - Slice {slice_idx}")
        plt.imshow(t_segmentation[0, 0, :, :, slice_idx] != 0, cmap="viridis")
        plt.subplot(1, 3, 3)
        plt.title(f"Patient {idx + 1}: Predicted Output - Slice {slice_idx}")
        plt.imshow(test_outputs[0, 1, :, :, slice_idx].cpu(), cmap="viridis")
        plt.tight_layout()

        # Save the visualization
        output_file = os.path.join(output_dir, f"patient_{idx + 1}_slice_{slice_idx}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved prediction for Patient {idx + 1} to {output_file}")

print("Inference completed and results saved.")

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preporcess import prepare
from utilities import train, calculate_weights, calculate_pixels
import os
import numpy as np

# Define data and model directories
data_dir = r"C:\\Users\\rajwn\\Videos\\monai_ML_project\\Atlas\\atlas-train-dataset-1.0.1\\atlas-train-dataset-1.0.1\\train"
model_dir = r"C:\\Users\\rajwn\\Videos\\monai_ML_project\\Atlas\\atlas-train-dataset-1.0.1\\results\\results_again"

# Prepare dataset
data_in = prepare(data_dir, cache=True)

# Compute pixel counts dynamically
pixel_counts = calculate_pixels(data_in[0])  # Train dataset
background_pixels, foreground_pixels = pixel_counts[0][0], pixel_counts[0][1]

# Compute class weights for loss balancing
weights = calculate_weights(background_pixels, foreground_pixels).to(torch.device("cuda:0"))

device = torch.device("cuda:0")

# Define model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,  # Background, Liver, Tumor
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Define loss function
loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=weights)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

# Train model
if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, max_epochs=150, model_dir=model_dir)

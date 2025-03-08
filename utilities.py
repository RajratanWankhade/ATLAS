from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm

def dice_metric(predicted, target, num_classes=3):
    """Compute Dice coefficient for each class separately."""
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    pred_onehot = torch.nn.functional.one_hot(torch.argmax(predicted, dim=1), num_classes=num_classes).permute(0, 4, 1, 2, 3)
    target_onehot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)
    return [1 - dice_value(pred_onehot[:, i], target_onehot[:, i]).item() for i in range(1, num_classes)]

def calculate_weights(val1, val2):
    """Compute class weights for handling imbalance in segmentation tasks."""
    count = np.array([val1, val2])
    if count.sum() == 0:  # Prevent division by zero
        return torch.tensor([0.5, 0.5], dtype=torch.float32)  # Assign equal weights
    
    weights = 1 / (count / count.sum())
    return torch.tensor(weights / weights.sum(), dtype=torch.float32)

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1):
    """Train the segmentation model and evaluate on the validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_metric, best_metric_epoch = -1, -1
    save_loss_train, save_loss_test = [], []
    save_metric_train, save_metric_test = [], []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss, epoch_metric_train = 0, np.zeros(2)
        for batch_data in train_loader:
            volume, label = batch_data["vol"].to(device), batch_data["seg"].to(device)
            optim.zero_grad()
            outputs = model(volume)
            train_loss = loss(outputs, label)
            train_loss.backward()
            optim.step()
            train_epoch_loss += train_loss.item()
            train_dice_scores = dice_metric(outputs, label, num_classes=3)
            epoch_metric_train += np.array(train_dice_scores)
            print(f'Train Dice - Liver: {train_dice_scores[0]:.4f}, Tumor: {train_dice_scores[1]:.4f}')
        
        train_epoch_loss /= len(train_loader)
        epoch_metric_train /= len(train_loader)
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Dice - Liver: {epoch_metric_train[0]:.4f}, Tumor: {epoch_metric_train[1]:.4f}")
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)
        
        if (epoch + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss, epoch_metric_test = 0, np.zeros(2)
                for test_data in test_loader:
                    test_volume, test_label = test_data["vol"].to(device), test_data["seg"].to(device)
                    test_outputs = model(test_volume)
                    test_epoch_loss += loss(test_outputs, test_label).item()
                    test_dice_scores = dice_metric(test_outputs, test_label, num_classes=3)
                    epoch_metric_test += np.array(test_dice_scores)
                
                test_epoch_loss /= len(test_loader)
                epoch_metric_test /= len(test_loader)
                print(f"Test Loss: {test_epoch_loss:.4f}, Test Dice - Liver: {epoch_metric_test[0]:.4f}, Tumor: {epoch_metric_test[1]:.4f}")
                save_loss_test.append(test_epoch_loss)
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                if epoch_metric_test.mean() > best_metric:
                    best_metric, best_metric_epoch = epoch_metric_test.mean(), epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
    
    print(f"Training completed, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}")

def calculate_pixels(data):
    """Count the number of foreground and background pixels in the dataset."""
    val = np.zeros((1, 2))
    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)
        if len(count) == 1:
            count = np.append(count, 0)
        val += count
    print('Pixel count:', val)
    return val

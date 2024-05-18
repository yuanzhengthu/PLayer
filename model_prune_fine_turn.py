import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PLayer import PLayer  # Assuming your custom model is defined in PLayer.py
from data_loader import MultiPageDataset_Continuous_Select, MultiPageDataset_Random_Select
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import datetime
from utils import configure_logging, CustomRandomTransform
from torchvision.utils import save_image
import logging
import tifffile
# Choose the GPU device(s) you want to use (e.g., GPU 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Create a subdirectory in the "experiments" directory with the current time as the folder name
experiment_dir = os.path.join("experiments", datetime.datetime.now().strftime("fine_turn_%Y%m%d_%H%M%S"))
os.makedirs(experiment_dir, exist_ok=True)
results_dir = experiment_dir + '/temp_results'
os.makedirs(results_dir, exist_ok=True)
# Configure logging
log_file_path = os.path.join(experiment_dir, 'fine_turn.log')
configure_logging(log_file_path)


# ########################################Define your model, criterion, and optimizer ######################################## #
input_channels = 3  # Replace with the actual number of input channels
output_channels = 3  # Replace with the actual number of output channels
n_feat = 40  # Replace with the desired number of features
stage = 1  # Replace with the desired number of stages
num_blocks = [1, 2, 2]  # Replace with the desired number of blocks for each stage
ifcuda = False
data_dir_for_train = 'D:/YuanzhengMA/2023-12-15 Nature it is/train_afterdenoise_multipage/train_new'  # 'dataset/train/input & output
data_dir_for_val = 'D:/YuanzhengMA/2023-12-15 Nature it is/train_afterdenoise_multipage/val_new'
batch_size = 32
crop_size = (256, 256)
# Training loop
num_epochs = 10  # Adjust based on your dataset
save_interval = 1  # Save the model every 5 epochs, adjust as needed
best_val_loss = float('inf')  # Initialize to positive infinity
# Define paths
pruned_model_path = 'deployment\pruned_quantized_model.pth' # 'experiments/20231230_223752/PLayer_epoch_30.pth' #
fine_tuned_model_path = 'fine_tuned_model.pth'
start_epoch = 0
# ########################################                                            ######################################## #
if ifcuda:
    device = 'cuda'
else:
    device = 'cpu'
# Data augmentation transforms
data_transform = CustomRandomTransform(rotation_range=15, color_jitter=(0.2, 0.2, 0.2, 0.2), crop_size=crop_size, device=device)

# Instantiate your custom model
pruned_model = PLayer(in_channels=input_channels, out_channels=output_channels, n_feat=n_feat, num_blocks=num_blocks).to('cuda')

# Load the pruned model
# checkpoint = torch.load(pruned_model_path)
#pruned_model.load_state_dict(checkpoint['model_state_dict'])
pruned_model = torch.load(pruned_model_path)
# pruned_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
#
# # pruned_model.qconfig = torch.quantization.QConfig(
# #     activation=torch.quantization.default_observer,
# #     weight=torch.quantization.default_per_tensor_weight_observer
# # )
# qconfig = torch.quantization.QConfig(
#     activation=torch.quantization.default_qconfig.activation,
#     weight=torch.quantization.default_observer.with_args(dtype=torch.qint8)
# )
# pruned_model.qconfig = qconfig
# # Disable per-channel quantization for all ConvTranspose2d layers
# for name, module in pruned_model.named_modules():
#     if isinstance(module, nn.ConvTranspose2d):
#         # Set the observer for the ConvTranspose2d layer to None
#         module.qconfig = None
#         module.activation_post_process = None
# pruned_model_prepared = torch.quantization.prepare(pruned_model)
# # Disable CuDNN during quantization
# torch.backends.cudnn.enabled = False
# pruned_model_prepared = torch.quantization.convert(pruned_model_prepared)
# pruned_model.load_state_dict(checkpoint)
# pruned_model = pruned_model_prepared
# Define a new loss function and optimizer for fine-tuning
criterion = nn.L1Loss()
optimizer = optim.Adam(pruned_model.parameters(), lr=2e-5, betas=[0.9, 0.999])

# Create an instance of your dataset and a DataLoader for training
train_dataset = MultiPageDataset_Random_Select(data_dir=data_dir_for_train, transform=data_transform, ifcuda=ifcuda, crop_size=crop_size, phase='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create an instance of your dataset and a DataLoader for validation
val_dataset = MultiPageDataset_Continuous_Select(data_dir=data_dir_for_val, transform=transforms.ToTensor(), ifcuda=ifcuda, crop_size=(512, 512), phase='test')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001, last_epoch=-1)

val_loss = 0.0
with torch.no_grad():
    for idx in range(0, val_dataset.hr_images.__len__()):
        til_page_num = len(tifffile.imread(os.path.join(val_dataset.hr_dir, val_dataset.hr_images[idx])))
        for idy in range(0, til_page_num):
            for idz, (val_lr_images, val_hr_images, original_size) in enumerate(val_dataset.__getitem__(idx=idx, target_page_index=idy)):
                val_lr_images = val_lr_images
                val_hr_images = val_hr_images

                val_sr_images = pruned_model(val_lr_images)
                val_loss += criterion(val_sr_images, val_hr_images).item()

                save_image(val_lr_images, os.path.join(experiment_dir, f'temp_results/val_lr_images_{idx}_{idy}.png'))
                save_image(val_sr_images, os.path.join(experiment_dir, f'temp_results/val_sr_images_{idx}_{idy}.png'))
                save_image(val_hr_images, os.path.join(experiment_dir, f'temp_results/val_hr_images_{idx}_{idy}.png'))
average_val_loss = val_loss / len(val_loader)
logging.info(f'Validation - Epoch {0}/{num_epochs}, Average Loss: {average_val_loss}')

# Fine-tuning loop
for epoch in range(start_epoch, num_epochs):
    pruned_model.train()

    for lr_images, hr_images, original_size in train_loader:
        lr_images = lr_images
        hr_images = hr_images

        # Forward pass
        outputs = pruned_model(lr_images)

        # Compute loss
        loss = criterion(outputs, hr_images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pruned_model.parameters(), 0.01)
        optimizer.step()

        # Update the learning rate
        scheduler.step()

    logging.info(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

    # Save the model checkpoint every n epochs in the experiment directory
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join(experiment_dir, f'PLayer_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model': pruned_model, #.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add other necessary information to save if needed
        }, model_save_path)
        print(f'Model saved at: {model_save_path}')

        # Validation loop
        pruned_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for idx, (val_lr_images, val_hr_images, original_size) in enumerate(val_loader):
                val_lr_images = val_lr_images
                val_hr_images = val_hr_images

                val_sr_images = pruned_model(val_lr_images)
                val_loss += criterion(val_sr_images, val_hr_images).item()

                save_image(val_lr_images, os.path.join(experiment_dir, f'temp_results/val_lr_images_{idx}.png'))
                save_image(val_sr_images, os.path.join(experiment_dir, f'temp_results/val_sr_images_{idx}.png'))
                save_image(val_hr_images, os.path.join(experiment_dir, f'temp_results/val_hr_images_{idx}.png'))
        average_val_loss = val_loss / len(val_loader)
        logging.info(f'Validation - Epoch {epoch+1}/{num_epochs}, Average Loss: {average_val_loss}')

        # Save the model checkpoint if it has the best validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model': pruned_model, #.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                # Add other necessary information to save if needed
            }, best_model_path)
            logging.info(f'Best model saved at: {best_model_path}')

# Save the log of the experiment
log_file_path = os.path.join(experiment_dir, 'experiment_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write(f'Experiment completed at: {datetime.datetime.now()}')
    log_file.write(f'Final validation loss: {average_val_loss}')

# Save the fine-tuned model
torch.save(pruned_model.state_dict(), fine_tuned_model_path)

# Quantize the fine-tuned model (optional)
quantized_model = torch.quantization.quantize_dynamic(
    pruned_model, {nn.Conv2d, nn.Linear, nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU}, dtype=torch.qint8
)

# Save the quantized model
quantized_model_path = 'quantized_model.pth'
torch.save(quantized_model.state_dict(), quantized_model_path)

print(f'Fine-tuned and quantized model saved at: {quantized_model_path}')

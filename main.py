import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from PLayer import PLayer  # Assuming your custom model is defined in custom_model.py
from torchvision import transforms
from data_loader import MultiPageDataset_Random_Select, MultiPageDataset_Continuous_Select
import datetime
from torchvision.utils import save_image
import logging
from utils import load_checkpoint, configure_logging, print_model_info, seed, CustomRandomTransform, CosineAnnealingRestartCyclicLR
import tifffile
import time


# Choose the GPU device(s) you want to use (e.g., GPU 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Set random seed for CPU and GPU
seed_num = None
seed_num = seed(seed_num)

# Create a subdirectory in the "experiments" directory with the current time as the folder name
experiment_dir = os.path.join("experiments", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(experiment_dir, exist_ok=True)
results_dir = experiment_dir + '/temp_results'
os.makedirs(results_dir, exist_ok=True)

# Configure logging
log_file_path = os.path.join(experiment_dir, 'training.log')
configure_logging(log_file_path)

# ########################################Define your model, criterion, and optimizer ######################################## #
input_channels = 3  # Replace with the actual number of input channels
output_channels = 3  # Replace with the actual number of output channels
n_feat = 40  # Replace with the desired number of features
stage = 1  # Replace with the desired number of stages
num_blocks = [1, 2, 2]  # Replace with the desired number of blocks for each stage
ifcuda = True
data_dir_for_train = 'D:\\YuanzhengMA\\2023-12-15 Nature it is\\train_afterdenoise_multipage\\train256'  # 'dataset/train/input & output
data_dir_for_val = 'D:\\YuanzhengMA\\2023-12-15 Nature it is\\train_afterdenoise_multipage\\val256'
batch_size = 2
crop_size = (256, 256)
crop_size_test = (1024, 1024)
# Training loop
num_epochs = 2000  # Adjust based on your dataset
save_interval = 20  # Save the model every 5 epochs, adjust as needed
best_val_loss = float('inf')  # Initialize to positive infinity
resume_path = None # 'experiments/20231230_134247/PLayer_epoch_20.pth'  # Change to the path of the checkpoint if you want to resume training
# ########################################                                            ######################################## #
if ifcuda:
    device = 'cuda'
else:
    device = 'cpu'

# Initialize the model
player_model = PLayer(in_channels=input_channels, out_channels=output_channels, n_feat=n_feat, num_blocks=num_blocks).to('cuda')

# Print model structure and hyperparameters to log
print_model_info(player_model,
                 input_channels=input_channels,
                 output_channels=output_channels,
                 n_feat=n_feat,
                 stage=stage,
                 num_blocks=num_blocks,
                 ifcuda=ifcuda,
                 data_dir_for_train=data_dir_for_train,
                 data_dir_for_val=data_dir_for_val,
                 batch_size=batch_size,
                 crop_size=crop_size,
                 num_epochs=num_epochs,
                 save_interval=save_interval,
                 resume_path=resume_path)

# Define criterion and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(player_model.parameters(), lr=2e-4, betas=[0.9, 0.999])

# Data augmentation transforms
data_transform = CustomRandomTransform(rotation_range=90, color_jitter=(0.2, 0.2, 0.2, 0.2), crop_size=crop_size, device=device)

# Create an instance of your dataset and a DataLoader for training
train_dataset = MultiPageDataset_Continuous_Select(data_dir=data_dir_for_train, transform=data_transform, ifcuda=ifcuda, crop_size=crop_size, phase='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create an instance of your dataset and a DataLoader for validation
val_dataset = MultiPageDataset_Continuous_Select(data_dir=data_dir_for_val, transform=transforms.ToTensor(), ifcuda=ifcuda, crop_size=crop_size_test, phase='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Learning rate scheduler
# scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001, last_epoch=-1)
scheduler = CosineAnnealingRestartCyclicLR(optimizer)
# Check if a checkpoint exists in the experiment directory
start_epoch, player_model, optimizer = load_checkpoint(experiment_dir, player_model, optimizer, resume_path)

for epoch in range(start_epoch, num_epochs):
    player_model.train()
    train_loss = 0.0
    # # 3D volume number
    # for idx in range(0, train_dataset.hr_images.__len__()):
    # page number of the 3D volume
    # til_page_num = len(tifffile.imread(os.path.join(train_dataset.hr_dir, train_dataset.hr_images[idx])))
    # for idy in range(0, til_page_num):
    #     train_dataset.set_target_page_index(idy)
          # 3D volume number
    for idz, (train_lq_images, train_gt_images, original_size) in enumerate(train_loader):
        # Forward pass
        outputs = player_model(train_lq_images)

        # Compute loss
        loss = criterion(outputs, train_gt_images)
        train_loss += loss
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(player_model.parameters(), 0.01)
        optimizer.step()

        # Update the learning rate
        scheduler.step()
    average_train_loss = train_loss / train_loader.batch_size
    logging.info(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {average_train_loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

    # Save the model checkpoint every n epochs in the experiment directory
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join(experiment_dir, f'PLayer_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model': player_model,
            'optimizer_state_dict': optimizer.state_dict(),
            # Add other necessary information to save if needed
        }, model_save_path)
        print(f'Model saved at: {model_save_path}')

        # Validation loop
        player_model.eval()
        val_loss = 0.0
        #total_reconstruction_time = 0
        with torch.no_grad():
            # 3D volume number
            for idz, (val_lq_images, val_gt_images, page_num) in enumerate(val_loader):
                # page number of the 3D volume
                til_page_num = page_num.item()
                total_reconstruction_time_volume = 0
                for idy in range(0, til_page_num):
                    val_dataset.set_target_page_index(idy)
                    # Measure the start time
                    start_time = time.time()

                    val_sr_images = player_model(val_lq_images)
                    val_loss += criterion(val_sr_images, val_gt_images).item()

                    # Measure the end time
                    end_time = time.time()

                    # Calculate the reconstruction time for the current image
                    reconstruction_time = end_time - start_time
                    # logging.info("Reconstruction time for image %d-%d: %f seconds", idz, idy, reconstruction_time)

                    # Accumulate the total reconstruction time
                    total_reconstruction_time_volume += reconstruction_time

                    save_image(val_lq_images, os.path.join(results_dir, f'val_lr_images_{idz}_{idy}.png'))
                    save_image(val_sr_images, os.path.join(results_dir, f'val_sr_images_{idz}_{idy}.png'))
                    save_image(val_gt_images, os.path.join(results_dir, f'val_hr_images_{idz}_{idy}.png'))
                logging.info("Reconstruction time for volume %d: %f seconds", idz, total_reconstruction_time_volume)
        average_val_loss = val_loss / til_page_num / val_loader.batch_size
        logging.info(f'Validation - Epoch {epoch+1}/{num_epochs}, Average Loss: {average_val_loss}')

        # Save the model checkpoint if it has the best validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': player_model.state_dict(),
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

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
# from piq import SSIMLoss, psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from data_loader import MultiPageDataset_Continuous_Select  # Assuming your dataset is defined in data_loader.py
from PLayer import PLayer  # Assuming your model is defined in PLayer.py
import datetime
from utils import configure_logging
import logging
import time
from torchvision.utils import save_image
import tifffile
import onnx
import torch.onnx
# Results directory
experiments_dir = os.path.join("results", datetime.datetime.now().strftime("%Y%m%d_%H%M%S_1024_40_02"))
os.makedirs(experiments_dir, exist_ok=True)
results_dir = experiments_dir + '/temp_results'
os.makedirs(results_dir, exist_ok=True)

# Configure logging
log_dir = experiments_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test.log")
configure_logging(log_file)

# Define your model, criterion, and load the trained weights
input_channels = 3  # Replace with the actual number of input channels
output_channels = 3  # Replace with the actual number of output channels
n_feat = 40  # Replace with the desired number of features
stage = 1  # Replace with the desired number of stages
num_blocks = [1, 2, 2]  # Replace with the desired number of blocks for each stage
ifcuda = False
crop_size = (1024, 1024)
if ifcuda:
    device = 'cuda'
else:
    device = 'cpu'
# Log model information
player_model = PLayer(in_channels=input_channels, out_channels=output_channels, n_feat=n_feat, num_blocks=num_blocks).to(device)
logging.info("Model parameters:\n%s", player_model)

# # Load ONNX model into PyTorch
# onnx_model = onnx.load('deployment/pruned_quantized_model_organoid.onnx')
# # Import the Tensorflow model into PyTorch
# pruned_model = torch.onnx(onnx_model)

# Load the trained weights
model_checkpoint_path = 'net_g_16000_1024_40.pth'  # Replace with the actual path
checkpoint = torch.load(model_checkpoint_path)
player_model.load_state_dict(checkpoint['params'])
pruned_model = player_model.to(device)

# pruned_model_path = "deployment/pruned_quantized_model_organoid_1024_40_02.pth"  # 'experiments/20231230_171303/PLayer_epoch_25.pth'  # Replace with the actual path
# pruned_model = torch.load(pruned_model_path).to(device)

# Data transformation for testing (similar to validation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create an instance of your test dataset and a DataLoader
test_dataset = MultiPageDataset_Continuous_Select(data_dir='D:/YuanzhengMA/2023-12-15 Nature it is/train_afterdenoise_multipage/val11', transform=test_transform, ifcuda=ifcuda, crop_size=crop_size, phase='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Criterion for evaluation (can be different from the training criterion if needed)
criterion = nn.L1Loss()
logging.info("Criterion: %s", criterion)

# Metrics
# ssim_loss = SSIMLoss(data_range=1, reduction='mean')
ssim_total = []
psnr_total = []
# niqe_total = []

# Testing loop
player_model.eval()
test_loss = 0.0
test_loss_total = []

val_loss = 0.0
total_reconstruction_time = 0.0
with torch.no_grad():
    for idx in range(0, test_dataset.hr_images.__len__()):
        til_page_num = len(tifffile.imread(os.path.join(test_dataset.hr_dir, test_dataset.hr_images[idx])))
        for idy in range(0, til_page_num):
            test_dataset.set_target_page_index(idy)
            for idz, (test_lq_images, test_gt_images, original_size) in enumerate(test_loader):

                # Measure the start time
                start_time = time.time()

                test_sr_images = pruned_model(test_lq_images)
                val_loss += criterion(test_sr_images, test_gt_images).item()

                # Measure the end time
                end_time = time.time()

                # Calculate the reconstruction time for the current image
                reconstruction_time = end_time - start_time
                logging.info("Reconstruction time for image %d-%d: %f seconds", idx, idy, reconstruction_time)

                # Accumulate the total reconstruction time
                total_reconstruction_time += reconstruction_time

                save_image(test_lq_images, os.path.join(results_dir, f'val_lr_images_{idx}_{idy}.png'))
                save_image(test_sr_images, os.path.join(results_dir, f'val_sr_images_{idx}_{idy}.png'))
                save_image(test_gt_images, os.path.join(results_dir, f'val_hr_images_{idx}_{idy}.png'))


                # Compute loss
                test_loss = criterion(test_sr_images, test_gt_images).item()
                test_loss_total.append(test_loss)

                # Clamp the pixel values to be within the valid range [0, 1]
                test_sr_images = torch.clamp(test_sr_images, 0, 1)
                test_gt_images = torch.clamp(test_gt_images, 0, 1)

                # Compute SSIM
                ssim_temp = ssim(test_sr_images.squeeze().permute(2,1,0).cpu().numpy(), test_gt_images.squeeze().permute(2,1,0).cpu().numpy(), win_size=3)
                ssim_total.append(ssim_temp)
                logging.info("SSIM for image %d: %f", idx, ssim_temp)

                # Compute PSNR
                psnr_temp = psnr(test_sr_images.squeeze().permute(2, 1, 0).cpu().numpy(),
                                 test_gt_images.squeeze().permute(2, 1, 0).cpu().numpy())
                psnr_total.append(psnr_temp)
                logging.info("PSNR for image %d: %f", idx, psnr_temp)

                # Compute NIQE
                # niqe_total.append(niqe(test_outputs, data_range=1.0).item())

# Calculate average reconstruction time per image
average_reconstruction_time = total_reconstruction_time / (len(test_dataset.hr_images) * til_page_num)
logging.info("Average Reconstruction Time per Image: %f seconds", average_reconstruction_time)

# Calculate average test loss
average_test_loss = sum(test_loss_total) / len(test_loss_total)
logging.info("Average Test Loss: %f", average_test_loss)

# Calculate average SSIM, PSNR, and NIQE
average_ssim = sum(ssim_total) / len(ssim_total)
average_psnr = sum(psnr_total) / len(psnr_total)
# average_niqe = niqe_total / len(niqe_total)

logging.info("Average SSIM: %f", average_ssim)
logging.info("Average PSNR: %f", average_psnr)

# Save metrics in the results directory
metrics_file_path = os.path.join(experiments_dir, 'metrics.txt')
with open(metrics_file_path, 'w') as metrics_file:
    metrics_file.write(f'Test Loss: {test_loss_total}\n')
    metrics_file.write(f'Average Test Loss: {average_test_loss}\n')
    metrics_file.write(f'SSIM: {ssim_total}\n')
    metrics_file.write(f'Average SSIM: {average_ssim}\n')
    metrics_file.write(f'PSNR: {psnr_total}\n')
    metrics_file.write(f'Average PSNR: {average_psnr}\n')
    # metrics_file.write(f'NIQE: {niqe_total}\n')
    # metrics_file.write(f'Average NIQE: {average_niqe}\n')

logging.info(f'Results saved in: {results_dir}')

# 3D displaying

import os
import numpy as np
import imageio
from PIL import Image
from mayavi import mlab
from niqe import calculate_niqe
import cv2

# Set the path to your image directory
# image_directory = 'NatureComm\\3ddemo\\raw'
image_directory = results_dir
# Get a list of image filenames in the directory
# image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')])
image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png') and 'sr' in f])
# Read the images, resize them, and stack them into a 3D numpy array
image_stack = []
for filename in image_files:
    image_path = os.path.join(image_directory, filename)

    # Read the image using imageio
    img = imageio.imread(image_path)

    # Resize the image to (512, 512) using Pillow
    img_resized = Image.fromarray(img).resize((512, 512))

    # Convert the resized image back to a NumPy array
    img_resized = np.array(img_resized)

    # Append the resized image to the stack
    image_stack.append(img_resized)

# Convert the list of images into a 3D numpy array
image_stack = np.stack(image_stack)

# # Assuming 'image_stack' is your 3D volume
# niqe_values = []
#
# for slice_2d in image_stack:
#     # Call the NIQE function from the MATLAB code
#     slice_2d = cv2.cvtColor(slice_2d, cv2.COLOR_BGR2GRAY)
#     niqe_value = calculate_niqe(slice_2d)
#     niqe_values.append(niqe_value)
#     print(niqe_value)
#
# average_niqe = np.mean(niqe_values)
# print(f"Average NIQE: {average_niqe}")

# Ensure the array is 3D
if image_stack.ndim == 4:
    # Assuming you want to use the first channel if it's an RGB image
    image_stack = image_stack[:, :, :, 1]

print("Min Value:", np.min(image_stack))
print("Max Value:", np.max(image_stack))

# Specify the resolution of the scalar field
resolution = (512, 512, len(image_files)*10)

# Create a Mayavi volume visualization with the specified colormap and resolution
mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(image_stack, figure=mlab.gcf(), extent=(0, 512, 0, 512, 0, len(image_files)), shape=resolution))
# mlab.colorbar(orientation='vertical')

#mlab.view(azimuth=0, elevation=90)
# mlab.view(azimuth=60, elevation=90)
# mlab.roll(90)
# mlab.show()

mlab.view(azimuth=00, elevation=90)
mlab.roll(90)
mlab.show()

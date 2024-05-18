import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import MultiPageDataset_Continuous_Select  # Assuming your dataset is defined in data_loader.py
from PLayer import RetinexFormer as PLayer # Assuming your model is defined in PLayer.py
import datetime
from utils import configure_logging
import logging
import time
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# import torch.onnx
# # Results directory
# experiments_dir = os.path.join("results", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
# os.makedirs(experiments_dir, exist_ok=True)
# results_dir = experiments_dir + '/temp_results'
# os.makedirs(results_dir, exist_ok=True)

# # Configure logging
# log_dir = experiments_dir
# os.makedirs(log_dir, exist_ok=True)
# log_file = os.path.join(log_dir, "test.log")
# configure_logging(log_file)

# # Define your model, criterion, and load the trained weights
# input_channels = 3  # Replace with the actual number of input channels
# output_channels = 3  # Replace with the actual number of output channels
# n_feat = 20  # Replace with the desired number of features
# stage = 3  # Replace with the desired number of stages
# num_blocks = [1, 2, 2]  # Replace with the desired number of blocks for each stage
# ifcuda = False
# if ifcuda:
    # device = 'cuda'
# else:
    # device = 'cpu'

# # Load the pre-trained model
# # SME = torch.load('SME_model_2x.pth')
# # SME = SME['model'].to(device)
# # SME.eval()  # Set the model to evaluation mode

# # Log model information
# # player_model = PLayer(in_channels=input_channels, out_channels=output_channels, n_feat=n_feat, num_blocks=num_blocks).to(device)
# #  logging.info("Model parameters:\n%s", player_model)

# # # Load ONNX model into PyTorch
# # onnx_model = onnx.load('deployment/pruned_quantized_model_organoid.onnx')
# # # Import the Tensorflow model into PyTorch
# # pruned_model = torch.onnx(onnx_model)

# # Load the trained weights
# # model_checkpoint_path = 'net_g_39000.pth' #'PLayer_8x_downup4.pth'  # Replace with the actual path
# # checkpoint = torch.load(model_checkpoint_path)
# # player_model.load_state_dict(checkpoint['params'])
# # player_model = checkpoint['model'].to(device)

# pruned_model_path = "deployment/pruned_model_organoid_0.99.pth"  # 'experiments/20231230_171303/PLayer_epoch_25.pth'  # Replace with the actual path
# player_model = torch.load(pruned_model_path).to(device)

# # Data transformation for testing (similar to validation)
# test_transform = transforms.Compose([
    # transforms.ToTensor(),
# ])

# # Create an instance of your test dataset and a DataLoader
# test_dataset = MultiPageDataset_Continuous_Select(data_dir='dataset/val256_11', transform=test_transform, ifcuda=ifcuda, crop_size=(512, 512), phase='test')
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Criterion for evaluation (can be different from the training criterion if needed)
# criterion = nn.L1Loss()
# logging.info("Criterion: %s", criterion)

# # Metrics
# ssim_total = []
# psnr_total = []
# # niqe_total = []

# # Testing loop
# player_model.eval()
# test_loss = 0.0
# test_loss_total = []

# val_loss = 0.0
# total_reconstruction_time = 0.0
# with torch.no_grad():
    # # # 3D volume number
    # # for idz, (test_lq_images, test_gt_images, page_num) in enumerate(test_loader):
    # # page number of the 3D volume
    # til_page_num = 21 # page_num.item()
    # total_reconstruction_time_volume = 0
    # for idy in range(0, til_page_num):
        # test_dataset.set_target_page_index(idy)
        # for idz, (test_lq_images, test_gt_images, page_num) in enumerate(test_loader):

            # # Measure the start time
            # start_time = time.time()

            # test_sr_images = player_model(test_lq_images)
            # val_loss += criterion(test_sr_images, test_gt_images).item()

            # # Measure the end time
            # end_time = time.time()

            # # Calculate the reconstruction time for the current image
            # reconstruction_time = end_time - start_time# - 5
            # logging.info("Reconstruction time for image %d-%d: %f seconds", idz, idy, reconstruction_time)

            # # Accumulate the total reconstruction time
            # total_reconstruction_time += reconstruction_time
            # save_image(test_lq_images, os.path.join(results_dir, f'val_lr_images_{idz}_{idy}.png'))
            # save_image(test_sr_images, os.path.join(results_dir, f'val_sr_images_{idz}_{idy}.png'))
            # save_image(test_gt_images, os.path.join(results_dir, f'val_hr_images_{idz}_{idy}.png'))

            # # Compute loss
            # # test_loss = criterion(test_sr_images, test_gt_images).item()
            # # test_loss_total.append(test_loss)

            # # Clamp the pixel values to be within the valid range [0, 1]
            # test_sr_images = torch.clamp(test_sr_images, 0, 1)
            # test_gt_images = torch.clamp(test_gt_images, 0, 1)

            # # Compute SSIM
            # # ssim_temp = ssim(test_sr_images.squeeze().permute(2,1,0).cpu().numpy(), test_gt_images.squeeze().permute(2,1,0).cpu().numpy(), win_size=3)
            # ssim_temp = ssim(test_sr_images.squeeze().permute(2,1,0).cpu().numpy(), test_gt_images.squeeze().permute(2,1,0).cpu().numpy(), win_size=3, data_range=test_gt_images.max().numpy() - test_gt_images.min().numpy())
            # ssim_total.append(ssim_temp)
            # logging.info("SSIM for image %d: %f", idz, ssim_temp)

            # # Compute PSNR
            # # Convert test_sr_images to the data type of ground_truth_images
            # test_sr_images = test_sr_images.to(test_gt_images.dtype)
            # psnr_temp = psnr(test_sr_images.squeeze().permute(2, 1, 0).cpu().numpy(),
                             # test_gt_images.squeeze().permute(2, 1, 0).cpu().numpy(), data_range=test_gt_images.max().numpy() - test_gt_images.min().numpy())
            # psnr_total.append(psnr_temp)
            # logging.info("PSNR for image %d: %f", idz, psnr_temp)


            # # Compute loss
            # # test_loss = criterion(test_sr_images, test_gt_images).item()c
            # # test_loss_total.append(test_loss)
            # #
            # # # Clamp the pixel values to be within the valid range [0, 1]
            # # test_sr_images = torch.clamp(test_sr_images, 0, 1)
            # # test_gt_images = torch.clamp(test_gt_images, 0, 1)
            # #
            # # # Compute SSIM
            # # ssim_total.append(ssim_loss(test_sr_images, test_gt_images).item())
            # # logging.info("SSIM for image %d: %f", idx, ssim_loss(test_sr_images, test_gt_images).item())
            # #
            # # # Compute PSNR
            # # psnr_total.append(psnr(test_sr_images, test_gt_images).item())
            # # logging.info("PSNR for image %d: %f", idx, psnr(test_sr_images, test_gt_images).item())
            # #
            # # # Compute NIQE
            # # # niqe_total.append(niqe(test_outputs, data_range=1.0).item())

# # Calculate average reconstruction time per image
# average_reconstruction_time = total_reconstruction_time / (len(test_dataset.hr_images) * til_page_num)
# logging.info("Average Reconstruction Time per Image: %f seconds", average_reconstruction_time)

# Calculate average test loss
# average_test_loss = sum(test_loss_total) / len(test_loss_total)
# logging.info("Average Test Loss: %f", average_test_loss)

# # Calculate average SSIM, PSNR, and NIQE
# average_ssim = sum(ssim_total) / len(ssim_total)
# average_psnr = sum(psnr_total) / len(psnr_total)
# # average_niqe = niqe_total / len(niqe_total)

# logging.info("Average SSIM: %f", average_ssim)
# logging.info("Average PSNR: %f", average_psnr)

# 3D displaying

import os
import numpy as np
import vtk
import imageio.v2 as imageio  # Importing imageio v2
from PIL import Image
from mayavi import mlab
from niqe import calculate_niqe
import cv2
class MultiTouchInteractor(vtk.vtkRenderWindowInteractor):
    def __init__(self):
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.touch_points = []

    def left_button_press_event(self, obj, event):
        event_pos = self.GetEventPosition()
        self.touch_points.append(event_pos)

    def mouse_move_event(self, obj, event):
        if len(self.touch_points) == 2:
            event_pos = self.GetEventPosition()
            self.handle_pinch_gesture(event_pos)

    def left_button_release_event(self, obj, event):
        self.touch_points = []

    def handle_pinch_gesture(self, event_pos):
        touch_point_1 = self.touch_points[0]
        touch_point_2 = self.touch_points[1]
        prev_distance = self.get_distance(touch_point_1, touch_point_2)

        touch_point_1 = (touch_point_1[0], event_pos[1])
        touch_point_2 = (event_pos[0], touch_point_2[1])
        new_distance = self.get_distance(touch_point_1, touch_point_2)

        zoom_factor = new_distance / prev_distance
        self.zoom_camera(zoom_factor)

        self.touch_points = [touch_point_1, touch_point_2]

    def get_distance(self, point_1, point_2):
        x1, y1 = point_1
        x2, y2 = point_2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def zoom_camera(self, zoom_factor):
        camera = mlab.gcf().scene.camera
        camera.zoom(zoom_factor)
        mlab.draw()
# Set the path to your image directory
# image_directory = 'D:\\pycharm_projs\\PLayer_proj_multipage2024\\results\\20240220_193706_2024_2x1\\temp_results'#'NatureComm\\3ddemo\\raw'
image_directory = 'results/20240516_004000/temp_results'#results_dir
# Get a list of image filenames in the directory
# image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')])
image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png') and 'hr' in f])
image_files = sorted(image_files, key=lambda x: int(x.split('_0_')[-1].split('.')[0]))

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
image_stack = np.stack(image_stack)#/255.0

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
# image_stack = image_stack.transpose(1,2,0)
print("Min Value:", np.min(image_stack))
print("Max Value:", np.max(image_stack))
# Get the minimum and maximum values along each axis
x_min, x_max = 0, image_stack.shape[2]
y_min, y_max = 0, image_stack.shape[1]
z_min, z_max = 0, image_stack.shape[0]

# Specify the resolution of the scalar field
resolution = image_stack.shape[::-1]

# Create a Mayavi volume visualization with the specified colormap and resolution
mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
src = mlab.pipeline.scalar_field(image_stack, figure=mlab.gcf(), extent=(x_min, x_max, y_min, y_max, z_min, z_max))
volume = mlab.pipeline.volume(src)

#mlab.view(azimuth=0, elevation=90)
# mlab.view(azimuth=60, elevation=90)
mlab.view(azimuth=00, elevation=90)
mlab.roll(90)


def quit_callback(vtk_obj, event):
    key = vtk_obj.GetKeySym()
    print(f"Key pressed: {key}")  # Debug print to check the key pressed
    if key == 'shift_L' or 'shift_R':
        mlab.close(all=True)
# Get the current figure and scene interactor
fig = mlab.gcf()
interactor = fig.scene.interactor

# Add the callback function to the key press event
interactor.add_observer('KeyPressEvent', quit_callback)

mlab.show()


# # Define variables for multi-touch handling
# import os
# import numpy as np
# from PIL import Image
# from mayavi import mlab
# import imageio
# import cv2
# from evdev import InputDevice, ecodes
#
# # Set the path to your image directory
# image_directory = 'results/20240516_004000/temp_results'
#
# # Get a list of image filenames in the directory
# image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png') and 'hr' in f])
# image_files = sorted(image_files, key=lambda x: int(x.split('_0_')[-1].split('.')[0]))
#
# # Read the images, resize them, and stack them into a 3D numpy array
# image_stack = []
# for filename in image_files:
#     image_path = os.path.join(image_directory, filename)
#
#     # Read the image using imageio
#     img = imageio.imread(image_path)
#
#     # Resize the image to (512, 512) using Pillow
#     img_resized = Image.fromarray(img).resize((512, 512))
#
#     # Convert the resized image back to a NumPy array
#     img_resized = np.array(img_resized)
#
#     # Append the resized image to the stack
#     image_stack.append(img_resized)
#
# # Convert the list of images into a 3D numpy array
# image_stack = np.stack(image_stack)
#
# # Ensure the array is 3D
# if image_stack.ndim == 4:
#     # Assuming you want to use the first channel if it's an RGB image
#     image_stack = image_stack[:, :, :, 1]
#
# print("Min Value:", np.min(image_stack))
# print("Max Value:", np.max(image_stack))
#
# # Specify the resolution of the scalar field
# resolution = (512, 512, len(image_files)*10)
#
# # Create a Mayavi volume visualization with the specified colormap and resolution
# mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
# volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(image_stack, figure=mlab.gcf(), extent=(0, 512, 0, 512, 0, len(image_files)), shape=resolution))
# mlab.colorbar(orientation='vertical')
#
# # Define variables for multi-touch handling
# touch1_x, touch1_y, touch2_x, touch2_y = None, None, None, None
# previous_zoom_distance = None
# touch1_release_timestamp, touch2_release_timestamp = None, None  # Add these lines
# touch1_timestamp, touch2_timestamp = None, None
#
# def update_zoom():
#     global previous_zoom_distance
#     if touch1_x is not None and touch2_x is not None:
#         current_zoom_distance = np.sqrt((touch2_x - touch1_x)**2 + (touch2_y - touch1_y)**2)
#         if previous_zoom_distance is not None:
#             zoom_factor = current_zoom_distance / previous_zoom_distance
#             mlab.move(forward=True, amount=zoom_factor)
#         previous_zoom_distance = current_zoom_distance
#
# # Touchscreen event handling
# def handle_touch_event(event):
#     global touch1_x, touch1_y, touch2_x, touch2_y
#     if event.code == ecodes.ABS_MT_TRACKING_ID:
#         if event.value == -1:  # Touch released
#             if event.timestamp() == touch1_release_timestamp:
#                 touch1_x, touch1_y = None, None
#             elif event.timestamp() == touch2_release_timestamp:
#                 touch2_x, touch2_y = None, None
#         else:
#             if touch1_x is None:
#                 touch1_x, touch1_y = event.timestamp()
#             elif touch2_x is None:
#                 touch2_x, touch2_y = event.timestamp()
#             else:
#                 # If already two touches are detected, ignore additional touches
#                 pass
#     elif event.code == ecodes.ABS_MT_POSITION_X:
#         if event.timestamp() == touch1_timestamp:
#             touch1_x = event.value
#         elif event.timestamp() == touch2_timestamp:
#             touch2_x = event.value
#         update_zoom()
#     elif event.code == ecodes.ABS_MT_POSITION_Y:
#         if event.timestamp() == touch1_timestamp:
#             touch1_y = event.value
#         elif event.timestamp() == touch2_timestamp:
#             touch2_y = event.value
#         update_zoom()
#
# # Open the touchscreen device
# touchscreen = InputDevice('/dev/input/event4')  # Replace X with your touchscreen device number
#
# # Read and handle touchscreen events
# try:
#     for event in touchscreen.read_loop():
#         handle_touch_event(event)
# finally:
#     touchscreen.close()

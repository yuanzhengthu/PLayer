# data_loader.py

import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
import numpy as np
import cv2
import tifffile
from utils import normalize_image, calculate_multiPIL_weights
from torchvision import transforms
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        #x = self.conv3(x)
        return x

class MultiPageDataset_Random_Select(Dataset):
    def __init__(self, data_dir, transform=None, ifcuda=True, crop_size=(256, 256), phase='train'):

        self.hr_dir = data_dir
        self.lr_dir = data_dir
        self.lr_images = os.listdir(self.lr_dir)
        self.hr_images = os.listdir(self.hr_dir)

        self.ifcuda = ifcuda
        self.crop_size = crop_size

        # Set the device based on ifcuda
        self.device = 'cuda' if ifcuda else 'cpu'

        self.transform = transform
        self.phase = phase
        # Create an instance of the MLP model
        self.model = CNN()
    def __len__(self):
        return len(self.lr_images)
    def set_target_page_index(self, new_index):
        self.target_page_index = new_index

    def __getitem__(self, idx):
        gt_path = os.path.join(self.hr_dir, self.hr_images[idx])

        # Read the multipage TIFF image
        multipage_tif = tifffile.imread(gt_path)

        # Random choose one page on the multipage PIL image
        random_target_index = random.randint(0, len(multipage_tif) - 1)
        target_page_index = random_target_index

        #multipage_tif = resize_and_crop_multiPIL(multipage_tif)
        # Calculate weights for neighboring pages
        weights = calculate_multiPIL_weights(target_page_index, len(multipage_tif))

        # Extract neighboring pages for weighted average
        left_indexs = [target_page_index - 2, target_page_index - 4, target_page_index - 6, 0]
        filtered_candidates = [idx for idx in left_indexs if idx >= 0]
        left_index = min(filtered_candidates)
        right_indexs = [target_page_index + 2, target_page_index + 4, target_page_index + 6, len(weights)]
        filtered_candidates = [idx for idx in right_indexs if idx <= len(weights)]
        right_index = max(filtered_candidates)
        neighbor_pages = multipage_tif[left_index:right_index + 1:2]
        weights = weights[left_index:right_index + 1:2]
        # Compute the weighted average
        weights = np.array(weights)
        blurred_images = []
        for img in neighbor_pages:
            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img, (7, 7), 5)  # Adjust kernel size as needed
            blurred_images.append(blurred_img)
        neighbor_pages = blurred_images
        img_lq = np.mean(neighbor_pages * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0) / 255.0
        img_lq = normalize_image(img_lq)
        img_gt = multipage_tif[target_page_index] / 255.0
        img_gt = normalize_image(img_gt)
        img_lq = Image.fromarray((img_lq*255.0).astype(np.uint8))
        img_gt = Image.fromarray((img_gt*255.0).astype(np.uint8))
        # Store the original sizes before any transformation
        original_gt_size = img_gt.size

        # Check if crop_size is None
        if self.crop_size is not None:
            # Random crop images to the specified size (option 1)
            # if self.crop_size[0] > original_lr_size[0] or self.crop_size[1] > original_lr_size[1]:
            #     lr_img = lr_img.resize(self.crop_size, resample=Image.BICUBIC))
            #     hr_img = hr_img.resize(self.crop_size, resample=Image.BICUBIC)
            # Add images into canvas with the specified size (option 2)
            if self.crop_size[0] > original_gt_size[0] or self.crop_size[1] > original_gt_size[1]:
                # Create a new image with zero pixels
                new_lr_img = Image.new("RGB", self.crop_size, (0, 0, 0))
                new_hr_img = Image.new("RGB", self.crop_size, (0, 0, 0))
                # Paste the original image onto the new image
                position = ((self.crop_size[0] - original_gt_size[0]) // 2, (self.crop_size[1] - original_gt_size[1]) // 2)
                new_lr_img.paste(img_lq, position)
                new_hr_img.paste(img_gt, position)
                # Update lr_img and hr_img with the resized images
                img_lq = new_lr_img
                img_gt = new_hr_img
        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(img_lq, output_size=(self.crop_size[0], self.crop_size[1]))
        img_lq = transforms.functional.crop(img_lq, i, j, h, w)
        img_gt = transforms.functional.crop(img_gt, i, j, h, w)

        if self.phase == 'train':
            img_gt, img_lq = self.transform(img_gt, img_lq)
        else:
            img_lq = self.transform(img_lq).to(self.device)
            img_gt = self.transform(img_gt).to(self.device)
        # img_lq = self.transform(img_lq).to(self.device)
        # img_gt = self.transform(img_gt).to(self.device)
        # Store the original sizes as tensor
        # original_hr_size_tensor = torch.tensor(original_hr_size)

        # Return original sizes along with images
        return img_lq, img_gt, original_gt_size


class MultiPageDataset_Continuous_Select(Dataset):
    def __init__(self, data_dir, transform=None, ifcuda=True, crop_size=(256, 256), phase='train'):
        self.hr_dir = data_dir
        self.lr_dir = data_dir
        self.lr_images = os.listdir(self.lr_dir)
        self.hr_images = os.listdir(self.hr_dir)
        self.ifcuda = ifcuda
        self.crop_size = crop_size

        # Set the device based on ifcuda
        self.device = 'cuda' if ifcuda else 'cpu'

        self.transform = transform
        self.phase = phase

        self.target_page_index = 0  # Initialize target_page_index

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.hr_dir, self.hr_images[idx])

        # Read the multipage TIFF image
        multipage_tif = tifffile.imread(gt_path)
        page_num = len(multipage_tif)
        # Choose page on the multipage PIL image one-by-one
        #for target_page_index in range(0, len(multipage_tif) - 1):

        # multipage_tif = resize_and_crop_multiPIL(multipage_tif)
        # Calculate weights for neighboring pages
        weights = calculate_multiPIL_weights(self.target_page_index, len(multipage_tif))

        # Extract neighboring pages for weighted average
        left_indexs = [self.target_page_index - 2, self.target_page_index - 4, self.target_page_index - 6, 0]
        filtered_candidates = [idx for idx in left_indexs if idx >= 0]
        left_index = min(filtered_candidates)
        right_indexs = [self.target_page_index + 2, self.target_page_index + 4, self.target_page_index + 6, len(weights)]
        filtered_candidates = [idx for idx in right_indexs if idx <= len(weights)]
        right_index = max(filtered_candidates)
        neighbor_pages = multipage_tif[left_index:right_index + 1:2]
        weights = weights[left_index:right_index + 1:2]
        # Compute the weighted average
        weights = np.array(weights)
        blurred_images = []
        for img in neighbor_pages:
            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img, (7, 7), 5)  # Adjust kernel size as needed
            blurred_images.append(blurred_img)
        neighbor_pages = blurred_images
        img_lq = np.mean(neighbor_pages * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0) / 255.0
        img_lq = normalize_image(img_lq)
        img_gt = multipage_tif[self.target_page_index] / 255.0
        img_gt = normalize_image(img_gt)
        img_lq = Image.fromarray((img_lq*255.0).astype(np.uint8))
        img_gt = Image.fromarray((img_gt*255.0).astype(np.uint8))
        # Store the original sizes before any transformation
        original_gt_size = img_gt.size

        # Check if crop_size is None
        if self.crop_size is not None:
            # Random crop images to the specified size (option 1)
            # if self.crop_size[0] > original_lr_size[0] or self.crop_size[1] > original_lr_size[1]:
            #     lr_img = Image.fromarray(np.array(lr_img.resize(self.crop_size, resample=Image.BICUBIC)))
            #     hr_img = Image.fromarray(np.array(hr_img.resize(self.crop_size, resample=Image.BICUBIC)))
            # Add images into canvas with the specified size (option 2)
            if self.crop_size[0] > original_gt_size[0] or self.crop_size[1] > original_gt_size[1]:
                # Create a new image with zero pixels
                new_lr_img = Image.new("RGB", self.crop_size, (0, 0, 0))
                new_hr_img = Image.new("RGB", self.crop_size, (0, 0, 0))
                # Paste the original image onto the new image
                position = (
                (self.crop_size[0] - original_gt_size[0]) // 2, (self.crop_size[1] - original_gt_size[1]) // 2)
                new_lr_img.paste(img_lq, position)
                new_hr_img.paste(img_gt, position)
                # Update lr_img and hr_img with the resized images
                img_lq = new_lr_img
                img_gt = new_hr_img
            # else:
            #     img_lq = img_lq.resize(self.crop_size, resample=Image.BICUBIC).convert('RGB')
            #     img_gt = img_gt.resize(self.crop_size, resample=Image.BICUBIC).convert('RGB')

        # Convert to BGR format (swap R and B channels)
        # img_lq = cv2.cvtColor(np.array(img_lq), cv2.COLOR_RGB2BGR)  # Convert PIL Image to BGR using OpenCV
        # img_gt = cv2.cvtColor(np.array(img_gt), cv2.COLOR_RGB2BGR)
        # Convert back to PIL Image
        # img_lq = Image.fromarray(cv2.cvtColor(img_lq_cv2, cv2.COLOR_BGR2RGB))
        # img_gt = Image.fromarray(cv2.cvtColor(img_gt_cv2, cv2.COLOR_BGR2RGB))
        if self.phase == 'train':
            img_gt, img_lq = self.transform(img_gt, img_lq)
        else:
            img_lq = self.transform(img_lq).to(self.device)
            img_gt = self.transform(img_gt).to(self.device)
        # Store the original sizes as tensor
        # original_hr_size_tensor = torch.tensor(original_hr_size)

        # Return original sizes along with images
        return img_lq, img_gt, page_num

    def set_target_page_index(self, new_index):
        self.target_page_index = new_index
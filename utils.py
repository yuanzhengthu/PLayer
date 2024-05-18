import os
import torch
import logging
import random
import numpy as np
import torch
# from torch.optim.lr_scheduler import LRScheduler

import torchvision.transforms.functional as F
import random
from PIL import Image
from torchvision import transforms
def load_checkpoint(experiment_dir, player_model, optimizer, resume_path):
    # Check if a checkpoint exists in the experiment directory
    if os.path.exists(experiment_dir):
        checkpoint_path = os.path.join(experiment_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            resume_path = checkpoint_path

    # Resume training or start from scratch
    if resume_path is not None:
        checkpoint = torch.load(resume_path)
        # player_model.load_state_dict(checkpoint['model_state_dict'])
        player_model = checkpoint['model']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        # Initialize weights only if starting from scratch
        start_epoch = 0  # Initialize start_epoch before using it
        player_model.apply(player_model._init_weights)
        logging.info(f"Training from scratch")

    return start_epoch, player_model, optimizer

def seed(seed_num=None):
    if seed_num is not None:
        # Set random seed for CPU
        torch.manual_seed(seed_num)
        # Set random seed for GPU (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True  # For deterministic results
            torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking
    else:
        seed_num = np.random.randint(1e6)  # Generate a random seed if not provided
        random.seed(seed_num)
        np.random.seed(seed_num)
        # Set random seed for GPU (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed_num

def configure_logging(log_file_path):
    # Configure logging
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Add a stream handler to write log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def print_model_info(player_model, input_channels, output_channels, n_feat, stage, num_blocks, ifcuda,
                     data_dir_for_train, data_dir_for_val, batch_size, crop_size, num_epochs, save_interval,
                     resume_path):
    # Print model structure to log
    logging.info("Model Structure:\n%s", str(player_model))

    # Print hyperparameters to log
    logging.info("Hyperparameters:")
    logging.info(f"Input Channels: {input_channels}")
    logging.info(f"Output Channels: {output_channels}")
    logging.info(f"Number of Features: {n_feat}")
    logging.info(f"Stage: {stage}")
    logging.info(f"Number of Blocks: {num_blocks}")
    logging.info(f"CUDA: {ifcuda}")
    logging.info(f"Data Directory for Training: {data_dir_for_train}")
    logging.info(f"Data Directory for Validation: {data_dir_for_val}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Crop Size: {crop_size}")
    logging.info(f"Number of Epochs: {num_epochs}")
    logging.info(f"Save Interval: {save_interval}")
    logging.info(f"Resume Path: {resume_path}")


import math

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


# class CosineAnnealingRestartCyclicLR(LRScheduler):
    # """ Cosine annealing with restarts learning rate scheme.
    # An example of config:
    # periods = [10, 10, 10, 10]
    # restart_weights = [1, 0.5, 0.5, 0.5]
    # eta_min=1e-7
    # It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    # scheduler will restart with the weights in restart_weights.
    # Args:
        # optimizer (torch.nn.optimizer): Torch optimizer.
        # periods (list): Period for each cosine anneling cycle.
        # restart_weights (list): Restart weights at each restart iteration.
            # Default: [1].
        # eta_min (float): The mimimum lr. Default: 0.
        # last_epoch (int): Used in _LRScheduler. Default: -1.
    # """

    # def __init__(self,
                 # optimizer,
                 # periods=[46000, 104000],
                 # restart_weights=(1,1),
                 # eta_mins=[0.0003,0.000001],
                 # last_epoch=-1):
        # self.periods = periods
        # self.restart_weights = restart_weights
        # self.eta_mins = eta_mins
        # assert (len(self.periods) == len(self.restart_weights)
                # ), 'periods and restart_weights should have the same length.'
        # self.cumulative_period = [
            # sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        # ]
        # super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)



    # def get_lr(self):
        # idx = get_position_from_periods(self.last_epoch,
                                        # self.cumulative_period)
        # current_weight = self.restart_weights[idx]
        # nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        # current_period = self.periods[idx]
        # eta_min = self.eta_mins[idx]

        # return [
            # eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            # (1 + math.cos(math.pi * (
                    # (self.last_epoch - nearest_restart) / current_period)))
            # for base_lr in self.base_lrs
        # ]

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0))#.to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class CustomRandomTransform(torch.nn.Module):
    def __init__(self, rotation_range=15, color_jitter=(0.2, 0.2, 0.2, 0.2, 0.2), crop_size=(256, 256), device='cuda'):
        self.angle = rotation_range
        self.brightness = color_jitter[0]
        self.contrast = color_jitter[1]
        self.saturation = color_jitter[2]
        self.hue = color_jitter[3]
        self.device = device
        self.crop_size = crop_size
        self.mixing_augmentation = Mixing_Augment(mixup_beta=1.2, use_identity='true', device=self.device)

    def __call__(self, hr_img, lr_img):
        # Random Rotation
        angle = random.choice([-self.angle*3, -self.angle*2, -self.angle, self.angle, 2*self.angle, 3*self.angle]) #random.uniform(-self.angle, self.angle)
        hr_img = F.rotate(hr_img, angle, Image.BICUBIC)
        lr_img = F.rotate(lr_img, angle, Image.BICUBIC)

        # Random Horizontal Flip
        if random.random() < 0.5:
            hr_img = F.hflip(hr_img)
            lr_img = F.hflip(lr_img)

        # Random Vertical Flip
        if random.random() < 0.5:
            hr_img = F.vflip(hr_img)
            lr_img = F.vflip(lr_img)

        # Color Jitter
        # if random.random() < 0.5:
        #     lr_img = F.color_jitter(lr_img, self.brightness, self.contrast, self.saturation, self.hue)
            # hr_img = F.color_jitter(hr_img, self.brightness, self.contrast, self.saturation, self.hue)

        # Additional Augmentations
        # # Random Crop
        # i, j, h, w = transforms.RandomCrop.get_params(lr_img, output_size=self.crop_size)
        # lr_img = F.crop(lr_img, i, j, h, w)
        # hr_img = F.crop(hr_img, i, j, h, w)

        # Gaussian Blur
        # if random.random() < 0.5:
        #     lr_img = F.gaussian_blur(lr_img, kernel_size=3)
        #     hr_img = F.gaussian_blur(hr_img, kernel_size=3)

        # Random Resized Crop
        # lr_img = F.resized_crop(lr_img, i, j, h, w, size=(256, 256), interpolation=Image.BICUBIC)
        # hr_img = F.resized_crop(hr_img, i, j, h, w, size=(256, 256), interpolation=Image.BICUBIC)

        # Random Affine
        # if random.random() < 0.5:
        #     scale_factor = random.uniform(0.8, 1.2)
        #     lr_img = F.affine(lr_img, angle, translate=(0.1, 0.1), scale=scale_factor, shear=15, interpolation=Image.BICUBIC)
        #     hr_img = F.affine(hr_img, angle, translate=(0.1, 0.1), scale=scale_factor, shear=15, interpolation=Image.BICUBIC)

        # Random Erasing
        # if random.random() < 0.5:
        #     lr_img = F.random_erasing(lr_img, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            # hr_img = F.random_erasing(hr_img, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))

        # Convert to Tensor
        hr_img = F.to_tensor(hr_img)
        lr_img = F.to_tensor(lr_img)

        # Mixing_Augment
        if random.random() < 0.5:
            lr_img, hr_img = self.mixing_augmentation(lr_img, hr_img)

        # Normalize
        # lr_img = F.normalize(lr_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # hr_img = F.normalize(hr_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return hr_img.to(self.device), lr_img.to(self.device)

def calculate_multiPIL_weights(target_page_index, num_pages):
    # Calculate the distances from the target page to all other pages
    distances = np.abs(np.arange(num_pages) - target_page_index)

    # Calculate weights as the inverse of distances (adding a small epsilon to avoid division by zero)
    weights = 1.0 / (distances + 1)

    # Normalize weights to sum to 1
    # weights /= np.sum(weights)

    return weights

def resize_and_crop_multiPIL(image, target_size=(1024, 1024), crop_size=(1024, 1024)):
    transformed_images = []

    for page in image:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(page.squeeze())

        # Resize and random crop
        pil_image = pil_image.resize(target_size, Image.BICUBIC)
        left = random.randint(0, pil_image.width - crop_size[0])
        top = random.randint(0, pil_image.height - crop_size[1])
        pil_image = pil_image.crop((left, top, left + crop_size[0], top + crop_size[1]))

        # Convert PIL Image back to numpy array
        transformed_image = np.array(pil_image)

        transformed_images.append(transformed_image)

    return np.stack(transformed_images)

def normalize_image(image):
    # Assuming 'image' is a NumPy array
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image

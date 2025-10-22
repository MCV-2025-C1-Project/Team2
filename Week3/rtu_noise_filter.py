import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from noise_filter import *

# Path to the images
img_folder_noisy = "../Data/Week3/qsd1_w3/"  # Update this path as necessary
img_folder_gt = "../Data/Week3/qsd1_w3/non_augmented/"  # Update this path as necessary
list_noisy_img = [f for f in os.listdir(img_folder_noisy) if f.endswith('.jpg')]
list_gt_img = [f for f in os.listdir(img_folder_gt) if f.endswith('.jpg')]
list_noisy_img.sort()
list_gt_img.sort()

THRESHOLD = 40

# Loop over the noisy (or hue changed or unmodified) images of qsd1_w3, along with their ground truth.
for n_img_path, gt_img_path in zip(list_noisy_img, list_gt_img):
    noisy_img = cv2.imread(os.path.join(img_folder_noisy, n_img_path))
    gt_img = cv2.imread(os.path.join(img_folder_gt, gt_img_path))
    # Calculate the noise score for the given image
    noise_score = fourier_noise_score(noisy_img, radius_ratio=0.75)
    # If the noise score is greater than the threshold, the image has noise in it.
    if noise_score > THRESHOLD:
        # Erase the noise from the image with two non-linear filters
        denoised_img = remove_noise_median(noisy_img, ksize=3)
        denoised_img = remove_noise_nlmeans(denoised_img, h=5, templateWindowSize=3, searchWindowSize=21)
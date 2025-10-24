import os
import cv2
import numpy as np
import pickle
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from rtu_noise_filter import fourier_noise_score, remove_noise_median, remove_noise_nlmeans



# CONFIG

IMG_FOLDER_NOISY = "../Data/Week3/qsd1_w3/"
IMG_FOLDER_GT = "../Data/Week3/qsd1_w3/non_augmented/"
GT_CORRESPS_PATH = os.path.join(IMG_FOLDER_NOISY, "gt_corresps.pkl")
THRESHOLD = 40


# DESCRIPTOR FUNCTIONS

def compute_lbp_descriptor(img, num_points=8, radius=1, grid_x=4, grid_y=4):
    """Compute block-based LBP descriptor."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    h, w = lbp.shape
    block_h, block_w = h // grid_y, w // grid_x
    histograms = []
    for i in range(grid_y):
        for j in range(grid_x):
            block = lbp[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            hist, _ = np.histogram(block.ravel(),
                                   bins=np.arange(0, num_points + 3),
                                   range=(0, num_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-8)
            histograms.extend(hist)
    return np.array(histograms)


def compute_dct_descriptor(img, num_coeff=64):
    """Compute DCT-based descriptor."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    dct_vector = dct_coeffs.flatten()[:num_coeff]
    return dct_vector



# NOISE PREPROCESSING

def preprocess_image(img):
    """Detect noise and denoise if necessary using rtu_noise_filter logic."""
    noise_score = fourier_noise_score(img, radius_ratio=0.75)
    if noise_score > THRESHOLD:
        print(f"Detected noise (score={noise_score:.2f}) → applying denoising filters...")
        img = remove_noise_median(img, ksize=3)
        img = remove_noise_nlmeans(img, h=5, templateWindowSize=3, searchWindowSize=21)
    else:
        print(f"No significant noise detected (score={noise_score:.2f})")
    return img


# DESCRIPTOR EXTRACTION

def extract_descriptors(folder):
    descriptors = []
    img_names = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for name in img_names:
        print(f"\nProcessing {name} ...")
        img_path = os.path.join(folder, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping {name}: could not read image.")
            continue
        processed_img = preprocess_image(img)
        lbp_desc = compute_lbp_descriptor(processed_img)
        dct_desc = compute_dct_descriptor(processed_img)
        combined_desc = np.concatenate([lbp_desc, dct_desc])
        descriptors.append(combined_desc)
    return np.array(descriptors), img_names




if __name__ == "__main__":
    main()

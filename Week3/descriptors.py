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


# EVALUATION FUNCTIONS

def compute_map_at_k(descriptors_query, descriptors_gt, gt_corresps, k=5):
    """Compute mean Average Precision at K using GT correspondences (list or dict)."""
    sims = cosine_similarity(descriptors_query, descriptors_gt)
    map_scores = []

    # Handle both dict and list structures for gt_corresps
    if isinstance(gt_corresps, list):
        gt_corresps_dict = {i: gt_corresps[i] for i in range(len(gt_corresps))}
    else:
        gt_corresps_dict = gt_corresps

    for i in range(len(descriptors_query)):
        ranked_indices = np.argsort(-sims[i])[:k]
        gt_indices = gt_corresps_dict.get(i, [])
        correct = [1 if idx in gt_indices else 0 for idx in ranked_indices]
        avg_prec = np.mean(correct[:k]) if np.any(correct) else 0
        map_scores.append(avg_prec)

    return np.mean(map_scores)



# MAIN PIPELINE

def main():
    print("=== TASK 1 + TASK 2 ===")
    print("Extracting descriptors for query and GT images...\n")

    # --- Extract descriptors ---
    desc_query, query_names = extract_descriptors(IMG_FOLDER_NOISY)
    desc_gt, gt_names = extract_descriptors(IMG_FOLDER_GT)

    # --- Save descriptors ---
    os.makedirs("results", exist_ok=True)
    with open("results/descriptors_task1_2.pkl", "wb") as f:
        pickle.dump({
            "query_descriptors": desc_query,
            "gt_descriptors": desc_gt,
            "query_names": query_names,
            "gt_names": gt_names
        }, f)
    print("\n✅ Descriptors saved to results/descriptors_task1_2.pkl")

    # --- Use direct 1-to-1 mapping with non_augmented folder ---
    print("\n⚙️ Using 1-to-1 GT mapping (query i ↔ non_augmented i) for Tasks 1 & 2.")
    gt_corresps = {i: [i] for i in range(len(desc_query))}


    # --- Evaluate ---
    map1 = compute_map_at_k(desc_query, desc_gt, gt_corresps, k=1)
    map5 = compute_map_at_k(desc_query, desc_gt, gt_corresps, k=5)
    print(f"\n📊 Evaluation Results:")
    print(f"   mAP@1 = {map1:.4f}")
    print(f"   mAP@5 = {map5:.4f}")


if __name__ == "__main__":
    main()

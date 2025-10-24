import os
import cv2
import numpy as np
import pickle
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
from rtu_noise_filter import fourier_noise_score, remove_noise_median, remove_noise_nlmeans

# CONFIG 
IMG_FOLDER_NOISY = "../Data/Week3/qsd1_w3/"
IMG_FOLDER_GT = "../Data/Week3/qsd1_w3/non_augmented/"
THRESHOLD = 40

# DESCRIPTOR FUNCTIONS 
def compute_lbp_descriptor(img, num_points=8, radius=1, grid_x=4, grid_y=4):
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
            hist = hist.astype("float") / (hist.sum() + 1e-8)
            histograms.extend(hist)
    return np.array(histograms)

def compute_dct_descriptor(img, num_coeff=64):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    return dct_coeffs.flatten()[:num_coeff]

# NOISE PREPROCESSING
def preprocess_image(img):
    noise_score = fourier_noise_score(img, radius_ratio=0.75)
    if noise_score > THRESHOLD:
        img = remove_noise_median(img, ksize=3)
        img = remove_noise_nlmeans(img, h=5, templateWindowSize=3, searchWindowSize=21)
    return img

# DESCRIPTOR EXTRACTION
def extract_descriptors(folder):
    descriptors = []
    img_names = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for name in img_names:
        img_path = os.path.join(folder, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = preprocess_image(img)
        lbp_desc = compute_lbp_descriptor(img)
        dct_desc = compute_dct_descriptor(img)
        descriptors.append(np.concatenate([lbp_desc, dct_desc]))
    return np.array(descriptors), img_names

# COSINE SIMILARITY
def cosine_similarity_matrix(Q, M, eps=1e-10):
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + eps)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)
    return np.dot(Q_norm, M_norm.T)

# MAP@K EVALUATION
def compute_map_at_k(descriptors_query, descriptors_gt, gt_corresps, k=5, similarity_func=cosine_similarity_matrix):
    sims = similarity_func(descriptors_query, descriptors_gt)
    nq = descriptors_query.shape[0]
    if isinstance(gt_corresps, list):
        gt_corresps = {i: gt_corresps[i] for i in range(len(gt_corresps))}
    map_scores = []
    for i in range(nq):
        ranked_indices = np.argsort(-sims[i])[:k]  # descending
        gt_indices = set(gt_corresps.get(i, []))
        num_relevant = len(gt_indices)
        if num_relevant == 0:
            map_scores.append(0)
            continue
        num_correct = 0
        precision_at_i = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if idx in gt_indices:
                num_correct += 1
                precision_at_i.append(num_correct / rank)
        ap = np.sum(precision_at_i) / num_relevant
        map_scores.append(ap)
    return np.mean(map_scores)

# MAIN PIPELINE 
def main():
    # Extract descriptors
    desc_query, query_names = extract_descriptors(IMG_FOLDER_NOISY)
    desc_gt, gt_names = extract_descriptors(IMG_FOLDER_GT)

    # Save descriptors
    os.makedirs("results", exist_ok=True)
    with open("results/descriptors_task1_2.pkl", "wb") as f:
        pickle.dump({
            "query_descriptors": desc_query,
            "gt_descriptors": desc_gt,
            "query_names": query_names,
            "gt_names": gt_names
        }, f)

    # 1-to-1 GT mapping
    gt_corresps = {i: [i] for i in range(len(desc_query))}

    # Evaluate
    map1 = compute_map_at_k(desc_query, desc_gt, gt_corresps, k=1)
    map5 = compute_map_at_k(desc_query, desc_gt, gt_corresps, k=5)
    print(f"mAP@1 = {map1:.4f}, mAP@5 = {map5:.4f}")

if __name__ == "__main__":
    main()

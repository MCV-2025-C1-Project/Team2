"""
Task 5: Evaluate Best Method from Week 2 with Background Removal
Adapted for qst2_w2 (no GT, just save retrieval results and masks)
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

from week2_histograms import SpatialPyramidHistogram
from similarity_measures_optimized import (
    l1_distance_matrix,
    histogram_intersection_matrix,
    kl_divergence_matrix,
    normalize_hist
)
from helper_functions_main import pil_to_cv2
from background_remover import remove_background_morphological_gradient

# --- Configuration ---
DB_PATH = "../Data/BBDD/"
QUERY_PATH = "../Data/Week2/qst2_w2/"   # changed to test set
RESULTS_DIR = "results_task6"
MASKS_DIR = os.path.join(RESULTS_DIR, "masks")
CACHE_DIR = "best_method_cache"

# Best method configuration
PYRAMID_CONFIG = {
    "bins": (4, 4, 4),
    "levels": 4,
    "weights": "uniform"
}

def load_image_cv(path):
    """Load image in OpenCV BGR format"""
    pil = Image.open(path).convert("RGB")
    return pil_to_cv2(pil)

def filename_to_id(fname):
    """Convert filename to image ID"""
    import re
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"\d+", base)
    return int(m.group()) if m else None

def get_bounding_box_mask(mask):
    """Get minimum bounding rectangle from polygon mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    rect_mask = np.zeros_like(mask)
    rect_mask[y_min:y_max+1, x_min:x_max+1] = 1
    return rect_mask, (y_min, y_max+1, x_min, x_max+1)

def build_descriptors_cached(image_list, base_path, color_space, cache_name, save_masks=False):
    """Build pyramid descriptors with caching and optional mask saving"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_{color_space}_bg_removed.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("image_list") == image_list:
                print(f"Loaded {color_space} descriptors from cache")
                return cached["descriptors"]
        except Exception as e:
            print(f"Cache load failed: {e}")

    print(f"Computing {color_space} descriptors...")
    pyramid = SpatialPyramidHistogram(
        bins=PYRAMID_CONFIG["bins"],
        levels=PYRAMID_CONFIG["levels"],
        color_space=color_space,
        weights=PYRAMID_CONFIG["weights"]
    )

    descriptors = []
    for fname in tqdm(image_list):
        img = load_image_cv(os.path.join(base_path, fname))

        # Only for query images (test set)
        if base_path == QUERY_PATH:
            _, poly_mask, _, _ = remove_background_morphological_gradient(img)
            rect_mask, (y1, y2, x1, x2) = get_bounding_box_mask(poly_mask)

            # Save mask
            if save_masks:
                mask_filename = os.path.splitext(fname)[0] + ".png"
                cv2.imwrite(os.path.join(MASKS_DIR, mask_filename), (poly_mask * 255).astype(np.uint8))

            # Crop image to bounding box
            img = img[y1:y2, x1:x2]

        desc = pyramid.compute(img)
        desc = normalize_hist(desc)
        descriptors.append(desc)

    descriptors = np.array(descriptors)
    with open(cache_file, "wb") as f:
        pickle.dump({"image_list": image_list, "descriptors": descriptors}, f)

    return descriptors


def run_retrieval():
    """Run retrieval on qst2_w2 and save results + masks"""
    print("=== Running Retrieval for qst2_w2 ===")
    print("Method: Pyramid Histogram Fusion")
    print(f"Configuration: {PYRAMID_CONFIG}")

    # Prepare folders
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    # Load image lists
    db_images = sorted([f for f in os.listdir(DB_PATH) if f.endswith('.jpg')])
    query_images = sorted([f for f in os.listdir(QUERY_PATH) if f.endswith('.jpg')])

    print(f"\nFound {len(db_images)} database images")
    print(f"Found {len(query_images)} query images")

    # Build descriptors
    db_hsv = build_descriptors_cached(db_images, DB_PATH, "HSV", "db")
    db_hls = build_descriptors_cached(db_images, DB_PATH, "HLS", "db")
    query_hsv = build_descriptors_cached(query_images, QUERY_PATH, "HSV", "query", save_masks=True)
    query_hls = build_descriptors_cached(query_images, QUERY_PATH, "HLS", "query")

    # Similarity measures
    sim_funcs = [l1_distance_matrix, histogram_intersection_matrix, kl_divergence_matrix]

    print("\nComputing similarities...")
    sim_hsv = np.stack([f(query_hsv, db_hsv) for f in sim_funcs], axis=-1)
    sim_hls = np.stack([f(query_hls, db_hls) for f in sim_funcs], axis=-1)

    # Convert intersection to distance
    sim_hsv[..., 1] *= -1
    sim_hls[..., 1] *= -1

    # Apply weights
    w1 = np.array([0.0, 1.0, 0.5])  # HSV
    w2 = np.array([0.5, 0.5, 0.0])  # HLS
    weighted_hsv = np.tensordot(sim_hsv, w1, axes=([2], [0]))
    weighted_hls = np.tensordot(sim_hls, w2, axes=([2], [0]))

    combined = 0.5 * (weighted_hsv + weighted_hls)

    print("\nComputing top-10 retrieval results...")
    results = []
    for q_idx in range(combined.shape[0]):
        top_k = np.argsort(combined[q_idx])[:10]
        pred_ids = [filename_to_id(db_images[idx]) for idx in top_k]
        results.append(pred_ids)

    # Save results
    results_file = os.path.join(RESULTS_DIR, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved top-10 retrieval results to {results_file}")
    print(f"Saved all query masks to {MASKS_DIR}")
    print("\nDone ✅")


if __name__ == "__main__":
    run_retrieval()

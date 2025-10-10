"""
Week 2 Descriptor Fusion
Combine best histogram descriptors (2D, 3D, Block, Pyramid)
and evaluate similarity fusion with multiple metrics.
"""

import os
import itertools
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from mapk import mapk

from similarity_measures_optimized import (
    euclidean_distance_matrix, l1_distance_matrix, x2_distance_matrix,
    histogram_intersection_matrix, hellinger_kernel_matrix, cosine_similarity_matrix,
    bhattacharyya_distance_matrix, correlation_matrix, kl_divergence_matrix,
    normalize_hist
)
from helper_functions_main import pil_to_cv2
from image_retrieval import load_ground_truth
from week2_histograms import Histogram2D, Histogram3D, BlockHistogram, SpatialPyramidHistogram
from Hyperparameter_search import compute_db_descriptors, compute_query_descriptors, filename_to_id


# --- CONFIG ---
RESULTS_DIR = "hp_results/"
CACHE_DIR = "hp_cache/"
os.makedirs("fusion_results", exist_ok=True)

DB_PATH = "../Data/BBDD/"
QSD_PATH = "../Data/Week1/qsd1_w1/"
GT_PATH = "../Data/Week1/qsd1_w1/gt_corresps.pkl"

# --- Similarity functions ---
SIM_FUNCS = [
    euclidean_distance_matrix,
    l1_distance_matrix,
    x2_distance_matrix,
    histogram_intersection_matrix,
    hellinger_kernel_matrix,
    cosine_similarity_matrix,
    bhattacharyya_distance_matrix,
    correlation_matrix,
    kl_divergence_matrix
]

SIM_NAMES = [f.__name__ for f in SIM_FUNCS]


# --- Load best configs from CSVs ---

def load_best_configs(csv_path):
    """
    Load best configuration per descriptor type based on MAP@5.
    Detects types from the config string (2D, 3D, BLOCK, PYRAMID).
    """
    df = pd.read_csv(csv_path)
    df["config_type"] = df["config"].apply(lambda x: x.split("_")[0].upper() if isinstance(x, str) else "UNKNOWN")
    
    type_mapping = {
        "2D": "2D",
        "3D": "3D",
        "BLOCK": "BLOCK",
        "PYRAMID": "PYRAMID"
    }

    best_configs = {}
    for key, val in type_mapping.items():
        subset = df[df["config"].str.startswith(key)]
        if not subset.empty:
            best_row = subset.loc[subset["MAP@5"].idxmax()]
            best_configs[val] = {
                "config": best_row["config"],
                "sim": best_row["sim"],
                "MAP@5": best_row["MAP@5"]
            }

    print("=== Loaded best configs ===")
    for k, v in best_configs.items():
        print(f"{k}: {v['config']} (MAP@5={v['MAP@5']:.4f}, sim={v['sim']})")

    return best_configs


# --- Load or compute descriptors ---
def get_descriptor_fn(name):
    if name.startswith("2D"):
        cs = "HSV" if "HSV" in name else "LAB"
        bins = (32, 32)
        return lambda img: Histogram2D(bins=bins, channels=(0, 1), color_space=cs).compute(img)
    elif name.startswith("3D"):
        cs = "RGB"
        bins = (8, 8, 8)
        return lambda img: Histogram3D(bins=bins, color_space=cs).compute(img)
    elif name.startswith("BLOCK"):
        return lambda img: BlockHistogram(bins=(8, 8, 8), grid=(2, 2), color_space="RGB").compute(img)
    elif name.startswith("PYR"):
        return lambda img: SpatialPyramidHistogram(bins=(8, 8, 8), levels=3, color_space="RGB").compute(img)
    else:
        raise ValueError(f"Unknown descriptor type for {name}")


# --- Concatenate and normalize descriptors ---
def concatenate_descriptors(desc_dict, keys):
    """Concat histograms [2D, 3D, Block, Pyramid] per image and normalize"""
    n = len(next(iter(desc_dict[keys[0]])))
    combined = []
    for i in range(n):
        parts = [desc_dict[k][i].ravel() for k in keys]
        vec = np.concatenate(parts)
        combined.append(normalize_hist(vec))
    return np.stack(combined)


# --- Fusion evaluation ---
def evaluate_fusion(Q, DB, gt, sim_funcs, weight_grid=[0, 0.5, 1]):
    results = []
    nq = Q.shape[0]
    nd = DB.shape[0]

    for sim in sim_funcs:
        D = sim(Q, DB)
        # Determine if similarity or distance
        if sim.__name__ in ['histogram_intersection_matrix', 'hellinger_kernel_matrix',
                            'cosine_similarity_matrix', 'correlation_matrix']:
            D = -D  # invert similarities
        indices = np.argsort(D, axis=1)
        preds = []
        for q in range(nq):
            topk = indices[q, :10]
            preds.append([filename_to_id(db_images[i]) for i in topk])
        map1 = mapk(gt, preds, 1)
        map5 = mapk(gt, preds, 5)
        results.append((sim.__name__, map1, map5))
    return results


if __name__ == "__main__":
    print("=== Fusion Experiment ===")
    csv_path = os.path.join(RESULTS_DIR, "all_20251010_175801_results.csv")
    best_cfgs = load_best_configs(csv_path)
    print(f"Loaded best configs: {list(best_cfgs.keys())}")

    db_images = sorted([f for f in os.listdir(DB_PATH) if f.endswith(".jpg")])
    query_images = sorted([f for f in os.listdir(QSD_PATH) if f.endswith(".jpg")])
    gt = load_ground_truth(GT_PATH)

    desc_types = ["2d", "3d", "block", "pyr"]
    desc_data = {}

    for d in desc_types:
        if d in best_cfgs:
            cfg = best_cfgs[d]
            name = cfg["config"]
            fn = get_descriptor_fn(name)
            desc_data[d] = {}
            print(f"\nComputing/loading {d.upper()} descriptors...")
            desc_data[d]["DB"] = compute_db_descriptors(db_images, DB_PATH, fn, cache_name=f"fusion_{d}")
            desc_data[d]["Q"] = compute_query_descriptors(query_images, QSD_PATH, fn)

    # Build concatenated feature sets
    print("\nBuilding concatenated descriptor sets...")
    db_features = concatenate_descriptors({k: desc_data[k]["DB"] for k in desc_types if k in desc_data}, desc_types)
    q_features = concatenate_descriptors({k: desc_data[k]["Q"] for k in desc_types if k in desc_data}, desc_types)

    print(f"Combined descriptor dimension: {db_features.shape[1]}")

    # Evaluate fusion
    results = evaluate_fusion(q_features, db_features, gt, SIM_FUNCS)

    df = pd.DataFrame(results, columns=["similarity", "MAP@1", "MAP@5"])
    df.sort_values("MAP@5", ascending=False, inplace=True)
    print(df.head(10))

    out_path = os.path.join("fusion_results", "fusion_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFusion results saved to {out_path}")

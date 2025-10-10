"""

Grid-search hyperparameters for Week 2 histograms:
- 2D histograms: HSV/LAB/HLS channel pairs, bins
- 3D histograms: RGB/HSV/LAB/HLS, bins
- Block histograms: grid sizes, bins, color spaces
- Pyramid histograms: levels, weights, bins, color spaces

Outputs CSVs with MAP@1 and MAP@5 for each config using two similarity measures:
- L1 distance (lower better)
- Histogram intersection (we convert to distance by negation for ranking)

"""

import os
import argparse
import itertools
import pickle
import time
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm


from week2_histograms import Histogram2D, Histogram3D, BlockHistogram, SpatialPyramidHistogram
from similarity_measures_optimized import (
    l1_distance_matrix,
    histogram_intersection_matrix,
    normalize_hist
)
from image_retrieval import load_ground_truth  
from helper_functions_main import pil_to_cv2, create_histogram_with_bins
from mapk import mapk


DEFAULT_DB_PATH = "../Data/BBDD/"
DEFAULT_QSD_PATH = "../Data/Week1/qsd1_w1/"
DEFAULT_GT_PATH = "../Data/Week1/qsd1_w1/gt_corresps.pkl"
CACHE_DIR = "hp_cache/"
RESULTS_DIR = "hp_results/"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def filename_to_id(fname):
    """
    Convert a database filename to integer id using same rule as your retrieval code:
    int(os.path.splitext(s)[0].split("_")[-1])
    If that fails, try to parse digits in name.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    try:
        return int(parts[-1])
    except:
        # fallback: extract first integer in the string
        import re
        m = re.search(r"\d+", base)
        return int(m.group()) if m else None


def load_image_cv(path):
    # read as PIL then convert to cv2 BGR to keep same pipeline used elsewhere
    pil = Image.open(path).convert("RGB")
    cv_img = pil_to_cv2(pil)
    return cv_img


def compute_db_descriptors(db_images, db_path, descriptor_fn, cache_name=None, recompute=False):
    """
    Compute descriptor for each db image (or load from cache).
    descriptor_fn(cv_img) -> 1D numpy array
    Returns list of descriptors in the same order as db_images.
    """
    if cache_name:
        cache_file = os.path.join(CACHE_DIR, f"dbdesc_{cache_name}.pkl")
    else:
        cache_file = None

    if cache_file and (not recompute) and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("image_list") == db_images:
                print(f"Loaded DB descriptors from cache: {cache_file}")
                return cached["descriptors"]
            else:
                print("Cache exists but image list mismatch; recomputing.")
        except Exception as e:
            print(f"Cache load failed: {e}")

    descriptors = []
    for fname in tqdm(db_images, desc="Computing DB descriptors"):
        path = os.path.join(db_path, fname)
        img = load_image_cv(path)
        desc = descriptor_fn(img)
        # ensure numpy array
        desc = np.asarray(desc, dtype=np.float32)
        # normalize (L1)
        if desc.sum() > 0:
            desc = desc / desc.sum()
        descriptors.append(desc)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump({"image_list": db_images, "descriptors": descriptors}, f)
    return descriptors


def compute_query_descriptors(query_images, query_path, descriptor_fn):
    """Compute descriptor for each query image (no caching)"""
    descriptors = []
    for fname in tqdm(query_images, desc="Computing query descriptors"):
        path = os.path.join(query_path, fname)
        img = load_image_cv(path)
        desc = descriptor_fn(img)
        desc = np.asarray(desc, dtype=np.float32)
        if desc.sum() > 0:
            desc = desc / desc.sum()
        descriptors.append(desc)
    return descriptors


def rank_db_for_queries(Q, DB, sim_func):
    """
    Q: numpy array (nq x d)
    DB: numpy array (nd x d)
    sim_func: function returning matrix of shape (nq, nd) where lower=closer OR higher=similar
    We'll rely on the semantics:
      - l1_distance_matrix returns distance: lower better
      - histogram_intersection_matrix returns similarity: higher better
    We'll convert to a "score" array where lower is better by:
      - if function returns similarity (we detect by name), we use -similarity
      - else use returned distances directly
    """
    M = sim_func(Q, DB)  # shape (nq, nd)
    name = sim_func.__name__
    if name in ['histogram_intersection_matrix', 'hellinger_kernel_matrix', 'cosine_similarity_matrix', 'correlation_matrix']:
        # treat higher = better -> convert to a distance by negation
        M = -M
    # Now smaller is better: argsort increasing
    idxs = np.argsort(M, axis=1)
    return idxs  # indices into DB for each Q row


# -------------------------
# Experiment definitions
# -------------------------
def make_2d_grid():
    # color spaces to explore and their channel pairs
    configs = []
    # For HSV/HLS, channel indices: H=0, S=1, V/L=2 (consistent with OpenCV layouts)
    channel_pairs = [(0, 1), (0, 2), (1, 2)]  # (0,1) (H,S),(H,V),(S,V)
    for cs in ['HSV', 'LAB', 'HLS']:
        for ch in channel_pairs:
            for bins in [(16, 16), (32, 32), (64, 64)]:
                name = f"2D_{cs}_{ch}_bins{bins[0]}x{bins[1]}"
                # descriptor function builder:
                def make_fn(bins=bins, ch=ch, cs=cs):
                    return lambda img: Histogram2D(bins=bins, channels=ch, color_space=cs).compute(img)
                configs.append((name, make_fn()))
    return configs


def make_3d_grid():
    configs = []
    for cs in ['RGB', 'HSV', 'LAB', 'HLS']:
        for bins in [(4, 4, 4), (8, 8, 8), (16, 16, 16)]:
            name = f"3D_{cs}_bins{bins[0]}x{bins[1]}x{bins[2]}"
            def make_fn(bins=bins, cs=cs):
                return lambda img: Histogram3D(bins=bins, color_space=cs).compute(img)
            configs.append((name, make_fn()))
    return configs


def make_block_grid():
    configs = []
    grids = [(2, 2), (3, 3), (4, 4)]
    color_spaces = ['RGB', 'LAB', 'HSV']
    bins_choices = [(4, 4, 4), (8, 8, 8)]
    for cs in color_spaces:
        for bins in bins_choices:
            for grid in grids:
                name = f"BLOCK_{cs}_bins{bins[0]}_grid{grid[0]}x{grid[1]}"
                def make_fn(bins=bins, cs=cs, grid=grid):
                    return lambda img: BlockHistogram(bins=bins, grid=grid, color_space=cs).compute(img)
                configs.append((name, make_fn()))
    return configs


def make_pyramid_grid():
    configs = []
    levels_choices = [2, 3, 4]  # level 0..L-1
    weightings = ['uniform', 'geometric', [1.0, 0.5, 0.25]]
    color_spaces = ['RGB', 'LAB', 'HSV']
    bins_choices = [(4, 4, 4), (8, 8, 8)]
    for cs in color_spaces:
        for bins in bins_choices:
            for levels in levels_choices:
                for w in weightings:
                    wlabel = 'uniform' if w == 'uniform' else ('geometric' if w == 'geometric' else 'custom')
                    name = f"PYR_{cs}_bins{bins[0]}_L{levels}_{wlabel}"
                    def make_fn(bins=bins, cs=cs, levels=levels, w=w):
                        return lambda img: SpatialPyramidHistogram(bins=bins, levels=levels, color_space=cs, weights=w).compute(img)
                    configs.append((name, make_fn()))
    return configs


# -------------------------
# Main evaluation loop
# -------------------------
def evaluate_configs(configs, db_path, qsd_path, gt_path, out_csv_prefix, recompute_db=False):
    # Load DB image list (sorted)
    db_images = sorted([f for f in os.listdir(db_path) if f.lower().endswith('.jpg')])
    print(f"DB images found: {len(db_images)}")
    query_images = sorted([f for f in os.listdir(qsd_path) if f.lower().endswith('.jpg')])
    print(f"Query images found: {len(query_images)}")
    gt = load_ground_truth(gt_path)
    if gt is None:
        raise RuntimeError("Ground truth not found or invalid.")

    # We'll evaluate using two similarity measures for each config
    sim_funcs = [l1_distance_matrix, histogram_intersection_matrix]
    sim_names = [f.__name__ for f in sim_funcs]

    rows = []
    t_tot_start = time.time()
    for name, desc_fn in configs:
        t0 = time.time()
        print(f"\n=== Evaluating {name} ===")
        # compute DB descriptors (cached per config)
        db_cache_name = f"{out_csv_prefix}_{name}"
        DB_descs = compute_db_descriptors(db_images, db_path, desc_fn, cache_name=db_cache_name, recompute=recompute_db)
        DB = np.stack([normalize_hist(d) for d in DB_descs])  # nd x d
        # compute query descriptors
        Q_descs = compute_query_descriptors(query_images, qsd_path, desc_fn)
        Q = np.stack([normalize_hist(d) for d in Q_descs])  # nq x d

        # For each similarity, rank and compute MAP@1 and MAP@5
        for sim in sim_funcs:
            rank_idxs = rank_db_for_queries(Q[np.newaxis,0] if False else Q, DB, sim)  # returns (nq, nd) idxs
            # convert to list of id lists 
            preds_ids = []
            for q_idx in range(rank_idxs.shape[0]):
                # top 10
                topk = rank_idxs[q_idx, :10]
                top_fnames = [db_images[i] for i in topk]
                ids = [filename_to_id(s) for s in top_fnames]
                preds_ids.append(ids)
            # compute map@k for k=1 and k=5
            map1 = mapk(gt, preds_ids, 1)
            map5 = mapk(gt, preds_ids, 5)
            row = {
                "config": name,
                "sim": sim.__name__,
                "MAP@1": map1,
                "MAP@5": map5,
                "db_desc_dim": DB.shape[1],
                "time_s": time.time() - t0,
                "num_db": DB.shape[0],
                "num_q": Q.shape[0]
            }
            print(f"  sim={sim.__name__} MAP@1={map1:.4f} MAP@5={map5:.4f} (dim={DB.shape[1]})")
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"{out_csv_prefix}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}. Total time: {time.time()-t_tot_start:.1f}s")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['2d', '3d', 'block', 'pyramid', 'all'], default='all',
                        help="Which group to run.")
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH)
    parser.add_argument("--qsd_path", default=DEFAULT_QSD_PATH)
    parser.add_argument("--gt_path", default=DEFAULT_GT_PATH)
    parser.add_argument("--recompute_db", action='store_true', help="Force recompute DB descriptors (ignore cache).")
    parser.add_argument("--limit", type=int, default=0, help="(Optional) limit number of configs to run (for quick tests).")
    args = parser.parse_args()

    configs = []
    if args.mode in ['2d', 'all']:
        configs.extend(make_2d_grid())
    if args.mode in ['3d', 'all']:
        configs.extend(make_3d_grid())
    if args.mode in ['block', 'all']:
        configs.extend(make_block_grid())
    if args.mode in ['pyramid', 'all']:
        configs.extend(make_pyramid_grid())

    if args.limit > 0:
        configs = configs[:args.limit]
        print(f"Limiting to first {args.limit} configs for quick test.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.mode}_{timestamp}"
    df = evaluate_configs(configs, args.db_path, args.qsd_path, args.gt_path, prefix, recompute_db=args.recompute_db)
    # Save a short CSV with the top configs by MAP@5 for easy inspection
    best = df.sort_values(["MAP@5", "MAP@1"], ascending=False).head(20)
    best_csv = os.path.join(RESULTS_DIR, f"{prefix}_top20.csv")
    best.to_csv(best_csv, index=False)
    print(f"Top 20 saved to {best_csv}")


if __name__ == "__main__":
    main()

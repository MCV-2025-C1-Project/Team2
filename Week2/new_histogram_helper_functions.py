# Import Required Libraries
import os, json, hashlib, itertools, re, cv2, pickle, time, warnings
from itertools import product, combinations
from collections import OrderedDict
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Import custom modules
from week2_histograms import Histogram2D, Histogram3D, BlockHistogram, SpatialPyramidHistogram
from similarity_measures_optimized import (
    l1_distance_matrix,
    histogram_intersection_matrix,
    kl_divergence_matrix,
    normalize_hist
)
from mapk import mapk

CACHE_DIR = "hp_cache/"

# Define Utility Functions
def pil_to_cv2(img):
    """Convert PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def filename_to_id(fname):
    """Convert a database filename to integer id"""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    try:
        return int(parts[-1])
    except:
        m = re.search(r"\d+", base)
        return int(m.group()) if m else None

def load_image_cv(path):
    """Load image as OpenCV BGR format"""
    pil = Image.open(path).convert("RGB")
    cv_img = pil_to_cv2(pil)
    return cv_img

def compute_db_descriptors(db_images, db_path, descriptor_fn, cache_name=None, recompute=False):
    """Compute descriptor for each db image (with caching)"""
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
        except Exception as e:
            print(f"Cache load failed: {e}")

    descriptors = []
    for fname in tqdm(db_images, desc="Computing DB descriptors"):
        path = os.path.join(db_path, fname)
        img = load_image_cv(path)
        desc = descriptor_fn(img)
        desc = np.asarray(desc, dtype=np.float32)
        if desc.sum() > 0:
            desc = desc / desc.sum()
        descriptors.append(desc)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump({"image_list": db_images, "descriptors": descriptors}, f)
    return descriptors

def compute_query_descriptors(query_images, query_path, descriptor_fn):
    """Compute descriptor for each query image"""
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
    """Rank database images for queries"""
    M = sim_func(Q, DB)
    name = sim_func.__name__
    if name in ['histogram_intersection_matrix', 'hellinger_kernel_matrix', 'cosine_similarity_matrix', 'correlation_matrix']:
        M = -M
    idxs = np.argsort(M, axis=1)
    return idxs

def evaluate_config(name, desc_fn, db_images, query_images, db_path, qsd_path, gt):
    """Evaluate a single configuration"""
    print(f"Evaluating {name}")
    
    # Compute descriptors
    db_cache_name = f"notebook_{name}"
    DB_descs = compute_db_descriptors(db_images, db_path, desc_fn, cache_name=db_cache_name)
    DB = np.stack([normalize_hist(d) for d in DB_descs])
    
    Q_descs = compute_query_descriptors(query_images, qsd_path, desc_fn)
    Q = np.stack([normalize_hist(d) for d in Q_descs])
    
    results = []
    sim_funcs = [l1_distance_matrix, histogram_intersection_matrix, kl_divergence_matrix]
    
    for sim in sim_funcs:
        rank_idxs = rank_db_for_queries(Q, DB, sim)
        preds_ids = []
        for q_idx in range(rank_idxs.shape[0]):
            topk = rank_idxs[q_idx, :10]
            top_fnames = [db_images[i] for i in topk]
            ids = [filename_to_id(s) for s in top_fnames]
            preds_ids.append(ids)
        
        map1 = mapk(gt, preds_ids, 1)
        map5 = mapk(gt, preds_ids, 5)
        
        results.append({
            "config": name,
            "sim": sim.__name__,
            "MAP@1": map1,
            "MAP@5": map5,
            "db_desc_dim": DB.shape[1]
        })
    
    return results

print("Utility functions defined successfully!")
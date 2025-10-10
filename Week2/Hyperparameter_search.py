"""
hyper-parameter search

Searches only the best color spaces from Week 1 (LAB, HLS, HSV)
and similarity metrics (L1, Histogram Intersection, KL-Divergence).

Evaluates 2D, 3D, Block, and Pyramid histograms implemented in week2_histograms.py
and logs MAP@1 / MAP@5 results to CSV.
"""

import os, time, pickle, argparse
import numpy as np, pandas as pd, cv2
from tqdm import tqdm
from PIL import Image

from week2_histograms import Histogram2D, Histogram3D, BlockHistogram, SpatialPyramidHistogram
from similarity_measures_optimized import (
    l1_distance_matrix,
    histogram_intersection_matrix,
    kl_divergence_matrix,
    normalize_hist
)
from image_retrieval import load_ground_truth
from helper_functions_main import pil_to_cv2
from mapk import mapk

# --- paths ---
DB_PATH_DEFAULT   = "../Data/BBDD/"
QSD_PATH_DEFAULT  = "../Data/Week1/qsd1_w1/"
GT_PATH_DEFAULT   = "../Data/Week1/qsd1_w1/gt_corresps.pkl"
CACHE_DIR, RESULTS_DIR = "hp_cache", "hp_results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- helper I/O ---
def filename_to_id(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    import re
    m = re.search(r"\d+", base)
    return int(m.group()) if m else None

def load_cv2_img(path):
    pil = Image.open(path).convert("RGB")
    return pil_to_cv2(pil)

# --- descriptor computation ---
def compute_db_desc(db_imgs, db_path, fn, cache_tag, recompute=False):
    cache_file = os.path.join(CACHE_DIR, f"{cache_tag}.pkl")
    if not recompute and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if data.get("image_list") == db_imgs:
            print(f"Loaded DB descriptors from cache: {cache_tag}")
            return data["desc"]
    descs = []
    for fimg in tqdm(db_imgs, desc=f"DB:{cache_tag}"):
        img = load_cv2_img(os.path.join(db_path, fimg))
        d = fn(img).astype("float32")
        d = normalize_hist(d)
        descs.append(d)
    with open(cache_file, "wb") as f:
        pickle.dump({"image_list": db_imgs, "desc": descs}, f)
    return descs

def compute_query_desc(q_imgs, q_path, fn):
    descs = []
    for fimg in tqdm(q_imgs, desc="Queries"):
        img = load_cv2_img(os.path.join(q_path, fimg))
        d = fn(img).astype("float32")
        d = normalize_hist(d)
        descs.append(d)
    return descs

# --- ranking + eval ---
def rank_queries(Q, DB, sim_func):
    M = sim_func(Q, DB)
    if sim_func.__name__ in ['histogram_intersection_matrix','hellinger_kernel_matrix',
                             'cosine_similarity_matrix','correlation_matrix']:
        M = -M   # convert similarity to distance
    return np.argsort(M, axis=1)

def evaluate_configs(configs, db_path, qsd_path, gt_path, prefix, recompute=False):
    db_imgs = sorted([f for f in os.listdir(db_path) if f.lower().endswith(".jpg")])
    q_imgs  = sorted([f for f in os.listdir(qsd_path) if f.lower().endswith(".jpg")])
    gt = load_ground_truth(gt_path)
    sim_funcs = [l1_distance_matrix, histogram_intersection_matrix, kl_divergence_matrix]
    rows=[]
    for name, fn in configs:
        print(f"\n=== {name} ===")
        DB = np.stack(compute_db_desc(db_imgs, db_path, fn, name, recompute))
        Q  = np.stack(compute_query_desc(q_imgs, qsd_path, fn))
        for sim in sim_funcs:
            idxs = rank_queries(Q, DB, sim)
            preds=[[filename_to_id(db_imgs[i]) for i in idxs[q,:10]] for q in range(len(q_imgs))]
            m1,m5=mapk(gt,preds,1),mapk(gt,preds,5)
            rows.append({"config":name,"sim":sim.__name__,"MAP@1":m1,"MAP@5":m5,
                         "dim":DB.shape[1],"num_q":len(q_imgs)})
            print(f"  {sim.__name__:<30} MAP@1={m1:.4f}  MAP@5={m5:.4f}")
    df=pd.DataFrame(rows)
    csv=os.path.join(RESULTS_DIR,f"{prefix}_results.csv")
    df.to_csv(csv,index=False)
    print(f"Saved results to {csv}")
    return df

# --- config grids (focused) ---
def make_2d_grid():
    configs=[]
    for cs in ['LAB','HLS','HSV']:
        for ch in [(0,1),(1,2)]:              
            for bins in [(16,16),(32,32)]:
                name=f"2D_{cs}_{ch}_b{bins[0]}"
                fn=lambda img,b=bins,c=ch,cs_=cs: Histogram2D(b,bins,c,cs_).compute(img)
                configs.append((name,lambda img,bins=b,ch=ch,cs=cs: Histogram2D(bins,ch,cs).compute(img)))
    return configs

def make_3d_grid():
    configs=[]
    for cs in ['LAB','HLS','HSV']:
        for bins in [(8,8,8),(16,16,16)]:
            name=f"3D_{cs}_b{bins[0]}"
            configs.append((name,lambda img,bins=bins,cs=cs: Histogram3D(bins,cs).compute(img)))
    return configs

def make_block_grid():
    configs=[]
    for cs in ['LAB','HLS','HSV']:
        for bins in [(8,8,8)]:
            for grid in [(2,2),(3,3)]:
                name=f"BLOCK_{cs}_b{bins[0]}_g{grid[0]}x{grid[1]}"
                configs.append((name,lambda img,bins=bins,grid=grid,cs=cs: 
                                BlockHistogram(bins,grid,cs).compute(img)))
    return configs

def make_pyr_grid():
    configs=[]
    for cs in ['LAB','HLS','HSV']:
        for bins in [(8,8,8)]:
            for levels in [2,3]:
                for w in ['uniform','geometric']:
                    tag='geo' if w=='geometric' else 'uni'
                    name=f"PYR_{cs}_b{bins[0]}_L{levels}_{tag}"
                    configs.append((name,lambda img,bins=bins,cs=cs,levels=levels,w=w:
                                    SpatialPyramidHistogram(bins,levels,cs,w).compute(img)))
    return configs

# --- main ---
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["2d","3d","block","pyr","all"],default="all")
    ap.add_argument("--db_path",default=DB_PATH_DEFAULT)
    ap.add_argument("--qsd_path",default=QSD_PATH_DEFAULT)
    ap.add_argument("--gt_path",default=GT_PATH_DEFAULT)
    ap.add_argument("--recompute_db",action="store_true")
    args=ap.parse_args()

    configs=[]
    if args.mode in ["2d","all"]:   configs+=make_2d_grid()
    if args.mode in ["3d","all"]:   configs+=make_3d_grid()
    if args.mode in ["block","all"]:configs+=make_block_grid()
    if args.mode in ["pyr","all"]:  configs+=make_pyr_grid()

    ts=time.strftime("%Y%m%d_%H%M%S")
    prefix=f"{args.mode}_{ts}"
    df=evaluate_configs(configs,args.db_path,args.qsd_path,args.gt_path,prefix,args.recompute_db)
    top=df.sort_values("MAP@5",ascending=False).head(15)
    top.to_csv(os.path.join(RESULTS_DIR,f"{prefix}_top.csv"),index=False)
    print(top)

if __name__=="__main__":
    main()

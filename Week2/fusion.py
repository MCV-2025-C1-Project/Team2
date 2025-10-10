
"""
Combines best Week 2 histograms (2D, 3D, Block, Pyramid)
and searches for weighted multi-similarity combinations.
"""

import os, itertools, pickle, numpy as np, pandas as pd, argparse
from tqdm import tqdm
from PIL import Image
from helper_functions_main import pil_to_cv2
from week2_histograms import Histogram2D, Histogram3D, BlockHistogram, SpatialPyramidHistogram
from similarity_measures import (
    euclidean_distance_matrix, l1_distance_matrix, x2_distance_matrix,
    histogram_intersection_matrix, hellinger_kernel_matrix, cosine_similarity_matrix,
    bhattacharyya_distance_matrix, correlation_matrix, kl_divergence_matrix, normalize_hist
)
from image_retrieval import load_ground_truth
from mapk import mapk

SIM_FUNCS = [
    euclidean_distance_matrix, l1_distance_matrix, x2_distance_matrix,
    histogram_intersection_matrix, hellinger_kernel_matrix, cosine_similarity_matrix,
    bhattacharyya_distance_matrix, correlation_matrix, kl_divergence_matrix
]

DB_PATH   = "../Data/BBDD/"
QSD_PATH  = "../Data/Week1/qsd1_w1/"
GT_PATH   = "../Data/Week1/qsd1_w1/gt_corresps.pkl"
CACHE_DIR = "hp_cache"
RESULTS_DIR = "hp_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def filename_to_id(fname):
    import re, os
    m = re.search(r"\d+", os.path.splitext(os.path.basename(fname))[0])
    return int(m.group()) if m else None

def load_cv2_img(path):
    from PIL import Image
    pil = Image.open(path).convert("RGB")
    return pil_to_cv2(pil)

def compute_desc(img, name):
    """Reconstruct descriptor function from config name string"""
    if name.startswith("2D_"):
        parts = name.split("_")
        cs = parts[1]; bins = (int(parts[3][1:]), int(parts[3][1:]))
        ch = (0,1) if "ch(0,1)" in name or "(0,1)" in name else (1,2)
        return Histogram2D(bins, ch, cs).compute(img)
    elif name.startswith("3D_"):
        parts = name.split("_"); cs=parts[1]; bins=(int(parts[2][1:]),)*3
        return Histogram3D(bins, cs).compute(img)
    elif name.startswith("BLOCK_"):
        parts=name.split("_"); cs=parts[1]; b=int(parts[2][1:]); g=parts[3].split("x")
        grid=(int(g[0][-1]),int(g[1]))
        return BlockHistogram((b,b,b), grid, cs).compute(img)
    elif name.startswith("PYR_"):
        parts=name.split("_"); cs=parts[1]; b=int(parts[2][1:]); L=int(parts[3][1:]); w='geometric' if 'geo' in name else 'uniform'
        return SpatialPyramidHistogram((b,b,b), L, cs, w).compute(img)
    else:
        raise ValueError(name)

def get_descs(db_imgs, q_imgs, db_path, q_path, conf_name):
    db_cache = os.path.join(CACHE_DIR, f"{conf_name}.pkl")
    with open(db_cache,"rb") as f: data=pickle.load(f)
    DB = np.stack(data["desc"])
    Q  = []
    for fimg in tqdm(q_imgs, desc=f"Query {conf_name}"):
        img = load_cv2_img(os.path.join(q_path,fimg))
        Q.append(normalize_hist(compute_desc(img, conf_name)))
    Q = np.stack(Q)
    return Q, DB

def evaluate_combination(q_imgs, db_imgs, gt, desc1, desc2, sim_weights):
    """Combine two descriptors with weighted similarity fusion."""
    Q1, DB1 = get_descs(db_imgs, q_imgs, DB_PATH, QSD_PATH, desc1)
    Q2, DB2 = get_descs(db_imgs, q_imgs, DB_PATH, QSD_PATH, desc2)
    sims=[]
    for sim_func, (w1,w2) in zip(SIM_FUNCS, sim_weights):
        S1 = sim_func(Q1, DB1); S2 = sim_func(Q2, DB2)
        # invert similarity metrics where higher=better
        if sim_func.__name__ in ['histogram_intersection_matrix','hellinger_kernel_matrix',
                                 'cosine_similarity_matrix','correlation_matrix']:
            S1=-S1; S2=-S2
        sims.append(w1*S1 + w2*S2)
    total = np.sum(sims, axis=0)
    idxs=np.argsort(total,axis=1)
    preds=[[filename_to_id(db_imgs[i]) for i in idxs[q,:10]] for q in range(len(q_imgs))]
    m1,m5=mapk(gt,preds,1),mapk(gt,preds,5)
    return m1,m5

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="hp_results/all_results.csv",
                        help="CSV from week2_hyperparam_search with MAP scores")
    args=parser.parse_args()

    df=pd.read_csv(args.csv_path)
    best2D  = df[df['config'].str.startswith("2D_") ].sort_values("MAP@5",ascending=False).iloc[0]['config']
    best3D  = df[df['config'].str.startswith("3D_") ].sort_values("MAP@5",ascending=False).iloc[0]['config']
    bestBLK = df[df['config'].str.startswith("BLOCK_")].sort_values("MAP@5",ascending=False).iloc[0]['config']
    bestPYR = df[df['config'].str.startswith("PYR_") ].sort_values("MAP@5",ascending=False).iloc[0]['config']
    print("\n=== Best individual configs ===")
    print(best2D, best3D, bestBLK, bestPYR)

    combos=[(best2D,best3D),(best2D,bestBLK),(best3D,bestPYR),
            ("2D_CIELAB_best","2D_HSV_best")]  # example manual combos

    weight_levels=[0.0,0.5,1.0]
    q_imgs=sorted([f for f in os.listdir(QSD_PATH) if f.endswith(".jpg")])
    db_imgs=sorted([f for f in os.listdir(DB_PATH) if f.endswith(".jpg")])
    gt=load_ground_truth(GT_PATH)
    results=[]
    for desc1,desc2 in combos:
        print(f"\nCombining {desc1} + {desc2}")
        for sim_weights in itertools.product(itertools.product(weight_levels,repeat=2), repeat=len(SIM_FUNCS)):
            sim_weights=np.array(sim_weights).reshape(len(SIM_FUNCS),2)
            m1,m5=evaluate_combination(q_imgs,db_imgs,gt,desc1,desc2,sim_weights)
            results.append({"desc1":desc1,"desc2":desc2,"weights":str(sim_weights.tolist()),
                            "MAP@1":m1,"MAP@5":m5})
    dfres=pd.DataFrame(results)
    dfres.to_csv(os.path.join(RESULTS_DIR,"fusion_search_results.csv"),index=False)
    print(dfres.sort_values("MAP@5",ascending=False).head(10))

if __name__=="__main__":
    main()

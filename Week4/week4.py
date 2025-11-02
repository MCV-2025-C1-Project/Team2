# ============================================================
# Retrieval system (HOG baseline + SIFT local features + RANSAC)
# - Memory safe SIFT (max_side) + resumable DB kp/desc cache
# - Classic unknown rule for RANSAC (no adaptive)
# - Split gating by min keypoints
# - Keeps your apk/mapk EXACTLY unchanged
# ============================================================

# -------------------------
# Config (edit these)
# -------------------------
QUERY_FOLDER = "../Data/Week4/qsd1_w4/"   # query images
DB_FOLDER    = "../Data/BBDD/"            # database images
GT_PATH      = "../Data/Week4/qsd1_w4/gt_corresps.pkl"
RESULTS_DIR  = "results"

# Choose descriptor for DB/query extraction path: "SIFT" or "HOG"
DESCRIPTOR   = "SIFT"     # set "HOG" to run HOG baseline

# Run which retrieval?
USE_RANSAC   = True       # True -> SIFT + RANSAC; False -> SIFT (no RANSAC)

# Hyperparams
LOWE_RATIO   = 0.76   # Lowe ratio for tentative matches
MIN_TENTATIVE = 8     # require at least this many tentative matches for RANSAC
MAX_TENTATIVE = 2000  # cap forward matches to avoid explosion in repetitive texture

# Image sizing / feature extraction
SIFT_MAX_SIDE = 1200  # downscale large images before SIFT to save memory (use same for DB & queries)
SIFT_NFEATURES = 1200 # SIFT nfeatures

# Unknown detection (non-RANSAC scoring; not used when USE_RANSAC=True)
SCORE_MODE   = "norm_count"            # "norm_count" | "count" | "ratio_margin" | "dist_weighted"
T_ABS        = 0.15                    # absolute floor on normalized score
T_RATIO      = 1.15                    # best/second ratio
ADAPTIVE     = False                   # keep False for now
SIGMA        = 200.0                   # for dist_weighted

# RANSAC thresholds (classic unknown rule)
T_INL        = 16     # min inliers to accept as known (try {12,16,20})
T_INL_RATIO  = 1.20   # best_inliers / second_inliers (try {1.15..1.35})
RANSAC_REPRJ = 3.0
RANSAC_ITERS = 2000
RANSAC_CONF  = 0.995

# Split gating
MIN_KP_SPLIT = 60     # if a split crop has < MIN_KP_SPLIT keypoints → treat as single

# Evaluation
K_VALS       = (1, 5)
SKIP_UNKNOWN = False   # False to include unknown queries in mAP

# -------------------------
# Imports
# -------------------------
import os
import gc
import cv2
import numpy as np
import pickle
from typing import List, Tuple, Union
from tqdm import tqdm

# External helpers (as in your repo)
from noise_filter import preprocess_image
import background_remover_w2 as background_remover
from image_split_team5 import segment_multiple_paintings

# Ensure results folder
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# IO helpers
# -------------------------
def load_images_from_folder(folder: str):
    """
    Loads all .jpg images from a folder, sorted by filename.
    Returns:
        names: list[str]
        imgs:  list[np.ndarray]
    """
    names = sorted([f for f in os.listdir(folder) if f and f.lower().endswith(".jpg")])
    imgs = []
    for name in names:
        path = os.path.join(folder, name)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Could not read {path}")
        imgs.append(img)
    return names, imgs

def validate_split(is_split, imgs, min_size_ratio=0.2):
    """
    Validate result of segment_multiple_paintings.
    Returns:
        (bool, list|np.ndarray): (ok_split, [img1, img2]) or (False, single_img)
    """
    if not is_split:
        return False, imgs
    if not isinstance(imgs, (list, tuple)) or len(imgs) != 2:
        return False, imgs

    img1, img2 = imgs
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    total_width = w1 + w2
    total_height = max(h1, h2)

    valid_left   = (w1 / total_width) > min_size_ratio and (h1 / total_height) > min_size_ratio
    valid_right  = (w2 / total_width) > min_size_ratio and (h2 / total_height) > min_size_ratio

    keep = []
    if valid_left:  keep.append(img1)
    if valid_right: keep.append(img2)

    if len(keep) == 2:
        return True, keep
    elif len(keep) == 1:
        return False, keep[0]
    else:
        return False, imgs

# -------------------------
# Utility
# -------------------------
def resize_max_side(img, max_side=SIFT_MAX_SIDE):
    """Downscale image so its longest side == max_side (keeps aspect)."""
    if img is None:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_size = (int(round(w*scale)), int(round(h*scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# -------------------------
# Descriptors (HOG + SIFT)
# -------------------------
def compute_hog_descriptor(
    img_bgr,
    size=(128, 128),
    cell=(4, 4),
    block=(16, 16),
    block_stride=(8, 8),
    nbins=9,
):
    """
    Compute a fixed-length HOG descriptor (L2-normalized).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    hog = cv2.HOGDescriptor(
        _winSize=size, _blockSize=block, _blockStride=block_stride,
        _cellSize=cell, _nbins=nbins
    )
    desc = hog.compute(gray).flatten().astype(np.float32)
    n = np.linalg.norm(desc)
    if n > 0: desc /= n
    return desc

def compute_sift_desc(img_bgr, nfeatures=SIFT_NFEATURES, rootsift=True, img_name="", max_side=SIFT_MAX_SIDE):
    """
    Return only SIFT descriptors (variable-length). Memory-safe via resize.
    """
    img_bgr = resize_max_side(img_bgr, max_side=max_side)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    _, des = sift.detectAndCompute(gray, None)
    if des is None:
        return None
    des = des.astype(np.float32)
    if rootsift:
        des /= (des.sum(axis=1, keepdims=True) + 1e-12)
        des = np.sqrt(des, out=des)
    return des

def compute_sift_kpdesc(img_bgr, nfeatures=SIFT_NFEATURES, rootsift=True, max_side=SIFT_MAX_SIDE):
    """
    Return (kp_xy: Nx2 float32, desc: Nx128 float32) for RANSAC branch. Memory-safe via resize.
    """
    img_bgr = resize_max_side(img_bgr, max_side=max_side)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return np.zeros((0,2), np.float32), None
    des = des.astype(np.float32)
    if rootsift:
        des /= (des.sum(axis=1, keepdims=True) + 1e-12)
        des = np.sqrt(des, out=des)
    kp_xy = np.array([k.pt for k in kps], dtype=np.float32)
    return kp_xy, des

# -------------------------
# Evaluation (KEEP YOURS)
# -------------------------
def apk(actual, predicted, k=10):
    """
    Average precision at k (unchanged).
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10, skip_unknown=True):
    """
    Your robust mapk version (unchanged behavior).
    """
    map_score = 0.0
    total_pics = 0
    for idx, p in enumerate(predicted):
        if len(p) == 2:
            if len(actual[idx]) == 2:
                apk_score1 = apk([actual[idx][0]], p[0], k)
                apk_score2 = apk([actual[idx][1]], p[1], k)
                map_score += apk_score1 + apk_score2
                total_pics += 2
            else:
                apk_score1 = apk(actual[idx], p[0], k)
                map_score += apk_score1
                total_pics += 1
        elif actual[idx] == [-1] and skip_unknown:
            continue
        else:
            apk_score1 = apk(actual[idx], p, k)
            map_score += apk_score1
            total_pics += 1
    return map_score / total_pics if total_pics > 0 else 0.0

# -------------------------
# Matching primitives (SIFT)
# -------------------------
_FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=64))

def mutual_ratio_matches(desQ, desD, ratio=LOWE_RATIO, flann=None):
    """
    Mutual (cross-checked) matches after Lowe ratio test. Returns [(qi, di, d1), ...].
    Caps forward matches to avoid explosion in repetitive texture.
    """
    if desQ is None or desD is None or len(desQ) == 0 or len(desD) == 0:
        return []
    flann = flann or _FLANN
    knn_qd = flann.knnMatch(desQ, desD, k=2)
    knn_dq = flann.knnMatch(desD, desQ, k=2)

    fwd = []
    for m, n in knn_qd:
        if m.distance < ratio * n.distance:
            fwd.append((m.queryIdx, m.trainIdx, m.distance))
            if len(fwd) >= MAX_TENTATIVE:
                break

    back_best = {}
    for m, n in knn_dq:
        if m.distance < ratio * n.distance:
            back_best[m.queryIdx] = (m.trainIdx, m.distance)

    mutual = []
    for qi, di, d in fwd:
        bb = back_best.get(di, None)
        if bb is not None and bb[0] == qi:
            mutual.append((qi, di, d))
    return mutual

def ransac_inliers(kpQ_xy, kpD_xy, pairs, reproj=RANSAC_REPRJ, maxIters=RANSAC_ITERS, conf=RANSAC_CONF):
    """
    RANSAC homography; returns (inlier_count, inlier_ratio).
    Guard against too few tentative matches.
    """
    if len(pairs) < MIN_TENTATIVE:
        return 0, 0.0
    ptsQ = np.float32([kpQ_xy[q] for q, _ in pairs])
    ptsD = np.float32([kpD_xy[d] for _, d in pairs])
    H, mask = cv2.findHomography(ptsQ, ptsD, cv2.RANSAC, reproj, maxIters=maxIters, confidence=conf)
    if mask is None:
        return 0, 0.0
    inl = int(mask.sum())
    return inl, inl / max(1, len(pairs))

def rank_db_ransac(kpQ_xy, desQ, db_kps, db_descs, ratio=LOWE_RATIO):
    ranks = []
    for i, (kpD_xy, desD) in enumerate(zip(db_kps, db_descs)):
        mutual = mutual_ratio_matches(desQ, desD, ratio=ratio, flann=_FLANN)
        pairs  = [(qi, di) for (qi, di, _) in mutual]
        inl, inl_ratio = ransac_inliers(kpQ_xy, kpD_xy, pairs)
        ranks.append((i, inl, inl_ratio))
    ranks.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return ranks

def good_match_count(desQ, desD, ratio=LOWE_RATIO):
    """
    Classic count (no mutual) for your old baseline/sweeps if needed.
    """
    if desQ is None or desD is None or len(desQ)==0 or len(desD)==0:
        return 0
    knn = _FLANN.knnMatch(desQ, desD, k=2)
    good = 0
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good += 1
    return good

def rank_db_with_scores(desQ, db_descs, ratio=LOWE_RATIO):
    scores = []
    for i, desD in enumerate(db_descs):
        c = good_match_count(desQ, desD, ratio)
        scores.append((i, c))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores  # [(db_idx, count), ...]

def decide_unknown_from_scores(scores, T_abs=12, T_ratio=1.4):
    if not scores:
        return True
    best = scores[0][1]
    second = scores[1][1] if len(scores) > 1 else 0
    if best < T_abs:
        return True
    if second > 0 and (best / (second + 1e-9)) < T_ratio:
        return True
    return False

# -------------------------
# Unknown detection (RANSAC)
# -------------------------
def decide_unknown_inliers(ranked, T_inl=T_INL, T_ratio=T_INL_RATIO):
    """
    Classic rule on inlier counts (no adaptive):
      unknown if best_inliers < T_inl OR best/second < T_ratio
    """
    if not ranked: return True
    b = ranked[0][1]
    s = ranked[1][1] if len(ranked) > 1 else 0
    if b < T_inl: return True
    if s > 0 and (b / (s + 1e-9)) < T_ratio: return True
    return False

# -------------------------
# Pipelines
# -------------------------
def build_db_descriptors(descriptor="SIFT"):
    """
    Build/load DB descriptors. Returns (db_names, db_imgs, db_descs)
    For SIFT: list of np.ndarray (Ni x 128)
    For HOG:  list of 1D vectors
    """
    desc_path = os.path.join(RESULTS_DIR, f"descriptors_db_{descriptor.lower()}.pkl")
    if os.path.exists(desc_path):
        with open(desc_path, "rb") as f:
            data = pickle.load(f)
        # If images not cached, reload them
        if "gt_imgs" not in data or data["gt_imgs"] is None:
            names, imgs = load_images_from_folder(DB_FOLDER)
            return names, imgs, data["desc_gt"]
        else:
            return data["gt_names"], data["gt_imgs"], data["desc_gt"]

    names, imgs = load_images_from_folder(DB_FOLDER)
    descs = []
    for im, nm in tqdm(zip(imgs, names), total=len(names), desc=f"DB {descriptor}"):
        if descriptor == "SIFT":
            d = compute_sift_desc(im, rootsift=True, img_name=nm, max_side=SIFT_MAX_SIDE)
        elif descriptor == "HOG":
            d = compute_hog_descriptor(im)
        else:
            raise ValueError("descriptor must be SIFT or HOG")
        descs.append(d)
    with open(desc_path, "wb") as f:
        pickle.dump({"desc_gt": descs, "gt_names": names, "gt_imgs": imgs}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return names, imgs, descs

def build_db_kp_desc_resumable(db_imgs, cache_path="results/db_kpdesc.pkl", max_side=SIFT_MAX_SIDE):
    """
    Build (or resume) DB keypoints + descriptors, saving checkpoints periodically.
    """
    db_kps, db_descs_local, start = [], [], 0
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            db_kps = data.get("db_kps", [])
            db_descs_local = data.get("db_descs", [])
            start = len(db_kps)
            print(f"↻ Resuming DB kp/desc from {start}/{len(db_imgs)}")
        except Exception as e:
            print(f"⚠️ Cache read failed ({e}), rebuilding...")

    for idx in tqdm(range(start, len(db_imgs)), desc="DB kp/desc"):
        im = db_imgs[idx]
        try:
            kp_xy, des = compute_sift_kpdesc(im, rootsift=True, max_side=max_side)
        except cv2.error:
            # Retry once with smaller side if memory fails
            kp_xy, des = compute_sift_kpdesc(im, rootsift=True, max_side=int(max_side*0.75))
        db_kps.append(kp_xy)
        db_descs_local.append(des)

        # periodic checkpoint + GC
        if (idx + 1) % 25 == 0 or idx == len(db_imgs) - 1:
            with open(cache_path, "wb") as f:
                pickle.dump({"db_kps": db_kps, "db_descs": db_descs_local}, f, protocol=pickle.HIGHEST_PROTOCOL)
            gc.collect()

    return db_kps, db_descs_local

def build_query_descriptors_sift(q_imgs, q_names):
    """
    Split + background removal -> per-crop SIFT descriptors.
    Returns desc_query: list of [desc_single] or [desc_left, desc_right]
    """
    desc_query = []
    for img, img_name in tqdm(zip(q_imgs, q_names), total=len(q_names), desc="Query SIFT desc"):
        img_p = preprocess_image(img)
        is_split, split_imgs = segment_multiple_paintings(img_p)
        ok_split, parts = validate_split(is_split, split_imgs)

        if ok_split:
            left, right = parts
            iml, maskL, *_ = background_remover.remove_background_morphological_gradient(left)
            imr, maskR, *_ = background_remover.remove_background_morphological_gradient(right)
            left_c  = background_remover.crop_to_mask_rectangle(left,  maskL)
            right_c = background_remover.crop_to_mask_rectangle(right, maskR)
            dL = compute_sift_desc(left_c,  rootsift=True, img_name=img_name+"_L", max_side=SIFT_MAX_SIDE)
            dR = compute_sift_desc(right_c, rootsift=True, img_name=img_name+"_R", max_side=SIFT_MAX_SIDE)

            # split gating by min keypoints (approx via descriptor rows)
            if (dL is None or (hasattr(dL, "shape") and len(dL) < MIN_KP_SPLIT)) or \
               (dR is None or (hasattr(dR, "shape") and len(dR) < MIN_KP_SPLIT)):
                # fallback to single
                imx, maskX, *_ = background_remover.remove_background_morphological_gradient(img_p)
                crop = background_remover.crop_to_mask_rectangle(img_p, maskX)
                dX = compute_sift_desc(crop, rootsift=True, img_name=img_name, max_side=SIFT_MAX_SIDE)
                desc_query.append([dX])
            else:
                desc_query.append([dL, dR])
        else:
            imx, maskX, *_ = background_remover.remove_background_morphological_gradient(img_p)
            crop = background_remover.crop_to_mask_rectangle(img_p, maskX)
            dX = compute_sift_desc(crop, rootsift=True, img_name=img_name, max_side=SIFT_MAX_SIDE)
            desc_query.append([dX])
    return desc_query

def build_query_kp_desc(q_imgs, q_names):
    """
    Same as above, but returns kp+desc for RANSAC branch.
    Returns:
      q_kps:   list of [kp_single] or [kp_left, kp_right]
      q_descs: list of [des_single] or [des_left, des_right]
    """
    q_kps, q_descs = [], []
    for img, img_name in tqdm(zip(q_imgs, q_names), total=len(q_names), desc="Query SIFT kp/desc"):
        img_p = preprocess_image(img)
        is_split, split_imgs = segment_multiple_paintings(img_p)
        ok_split, parts = validate_split(is_split, split_imgs)

        if ok_split:
            left, right = parts
            iml, maskL, *_ = background_remover.remove_background_morphological_gradient(left)
            imr, maskR, *_ = background_remover.remove_background_morphological_gradient(right)
            left_c  = background_remover.crop_to_mask_rectangle(left,  maskL)
            right_c = background_remover.crop_to_mask_rectangle(right, maskR)
            kpL, dL = compute_sift_kpdesc(left_c,  rootsift=True, max_side=SIFT_MAX_SIDE)
            kpR, dR = compute_sift_kpdesc(right_c, rootsift=True, max_side=SIFT_MAX_SIDE)

            # split gating by min keypoints
            if len(kpL) < MIN_KP_SPLIT or len(kpR) < MIN_KP_SPLIT:
                imx, maskX, *_ = background_remover.remove_background_morphological_gradient(img_p)
                crop = background_remover.crop_to_mask_rectangle(img_p, maskX)
                kpX, dX = compute_sift_kpdesc(crop, rootsift=True, max_side=SIFT_MAX_SIDE)
                q_kps.append([kpX]); q_descs.append([dX])
            else:
                q_kps.append([kpL, kpR]); q_descs.append([dL, dR])
        else:
            imx, maskX, *_ = background_remover.remove_background_morphological_gradient(img_p)
            crop = background_remover.crop_to_mask_rectangle(img_p, maskX)
            kpX, dX = compute_sift_kpdesc(crop, rootsift=True, max_side=SIFT_MAX_SIDE)
            q_kps.append([kpX]); q_descs.append([dX])
    return q_kps, q_descs

# HOG query (simple per-image descriptor after preprocessing + masking)
def build_query_descriptors_hog(q_imgs, q_names):
    desc_query = []
    for img, img_name in tqdm(zip(q_imgs, q_names), total=len(q_names), desc="Query HOG desc"):
        img_p = preprocess_image(img)
        imx, maskX, *_ = background_remover.remove_background_morphological_gradient(img_p)
        crop = background_remover.crop_to_mask_rectangle(img_p, maskX)
        dX = compute_hog_descriptor(crop)
        desc_query.append([dX])
    return desc_query

# -------------------------
# Predictors (shape preserved)
# -------------------------
def predict_with_unknowns_non_ransac(desc_query, db_descs, *,
                                     topK=10, ratio=LOWE_RATIO,
                                     T_abs=12, T_ratio=1.4):
    """
    Classic count-based scoring + classic unknown rule.
    """
    predictions = []
    for desc_crops in tqdm(desc_query, desc="Predict (no RANSAC)"):
        crop_preds = []
        for desQ in desc_crops:
            if desQ is None or len(desQ) == 0:
                crop_preds.append([-1]); continue
            scores = rank_db_with_scores(desQ, db_descs, ratio=ratio)
            ranked_indices = [i for i, _ in scores]
            if decide_unknown_from_scores(scores, T_abs=T_abs, T_ratio=T_ratio):
                crop_preds.append([-1])
            else:
                crop_preds.append(ranked_indices[:topK])
        predictions.append(crop_preds)
    return predictions

def predict_with_unknowns_ransac(q_kps, desc_query, db_kps, db_descs, *,
                                 topK=10, ratio=LOWE_RATIO,
                                 T_inl=T_INL, T_ratio=T_INL_RATIO):
    """
    RANSAC ranking by inliers + classic unknown rule on inlier counts.
    """
    predictions = []
    for kp_crops, desc_crops in tqdm(zip(q_kps, desc_query), total=len(desc_query), desc="Predict (RANSAC)"):
        crop_preds = []
        for kpQ_xy, desQ in zip(kp_crops, desc_crops):
            if desQ is None or len(desQ) == 0 or kpQ_xy is None or len(kpQ_xy) == 0:
                crop_preds.append([-1]); continue
            ranked = rank_db_ransac(kpQ_xy, desQ, db_kps, db_descs, ratio=ratio)
            ranked_idx = [i for (i, _, __) in ranked]
            if decide_unknown_inliers(ranked, T_inl=T_inl, T_ratio=T_ratio):
                crop_preds.append([-1])
            else:
                crop_preds.append(ranked_idx[:topK])
        predictions.append(crop_preds)
    return predictions

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    print("📥 Loading DB images & descriptors...")
    db_names, db_imgs, db_descs = build_db_descriptors(descriptor=DESCRIPTOR)

    print("📥 Loading query images...")
    q_names, q_imgs = load_images_from_folder(QUERY_FOLDER)

    print("📥 Loading GT...")
    with open(GT_PATH, "rb") as f:
        gt_corresps = pickle.load(f)

    # ----- Branch: HOG baseline -----
    if DESCRIPTOR == "HOG":
        print("🧠 Building HOG descriptors for queries...")
        desc_query = build_query_descriptors_hog(q_imgs, q_names)

        # HOG retrieval via cosine similarity (no unknowns here)
        from sklearn.metrics.pairwise import cosine_similarity
        db_mat = np.vstack(db_descs)
        predictions = []
        for desc_crops in tqdm(desc_query, desc="Predict (HOG Cosine)"):
            crop_preds = []
            for dQ in desc_crops:
                if dQ is None or len(dQ) == 0:
                    crop_preds.append([-1]); continue
                sims = cosine_similarity([dQ], db_mat)[0]
                ranked = list(np.argsort(-sims))
                crop_preds.append(ranked[:10])
            predictions.append(crop_preds)

        for k in K_VALS:
            score = mapk(gt_corresps, predictions, k=k, skip_unknown=SKIP_UNKNOWN)
            print(f"✅ HOG mAP@{k} = {score:.4f}")

    # ----- Branch: SIFT local pipeline -----
    else:
        if USE_RANSAC:
            print("🧠 Building SIFT kp+desc for DB (cached) & queries...")
            cache_path = os.path.join(RESULTS_DIR, "db_kpdesc.pkl")

            # Ensure DB imgs are available (cache from descriptors may not include them)
            if db_imgs is None:
                print("📷 Reloading DB images because they were not stored in cache...")
                _, db_imgs = load_images_from_folder(DB_FOLDER)

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        data = pickle.load(f)
                    db_kps = data.get("db_kps")
                    db_descs_kp = data.get("db_descs")
                    if not isinstance(db_kps, list) or not isinstance(db_descs_kp, list) or len(db_kps) != len(db_descs_kp):
                        print("⚠️ Invalid cache, rebuilding...")
                        db_kps, db_descs_kp = build_db_kp_desc_resumable(db_imgs, cache_path=cache_path, max_side=SIFT_MAX_SIDE)
                except Exception as e:
                    print(f"⚠️ Cache read failed ({e}), rebuilding...")
                    db_kps, db_descs_kp = build_db_kp_desc_resumable(db_imgs, cache_path=cache_path, max_side=SIFT_MAX_SIDE)
            else:
                db_kps, db_descs_kp = build_db_kp_desc_resumable(db_imgs, cache_path=cache_path, max_side=SIFT_MAX_SIDE)

            q_kps, q_descs = build_query_kp_desc(q_imgs, q_names)

            print("🚀 Predicting with RANSAC…")
            pred = predict_with_unknowns_ransac(
                q_kps, q_descs, db_kps, db_descs_kp,
                topK=10, ratio=LOWE_RATIO,
                T_inl=T_INL, T_ratio=T_INL_RATIO
            )
        else:
            print("🧠 Building SIFT descriptors for queries (no RANSAC)…")
            desc_query = build_query_descriptors_sift(q_imgs, q_names)

            print("🚀 Predicting without RANSAC (classic counts + thresholds)…")
            pred = predict_with_unknowns_non_ransac(
                desc_query, db_descs,
                topK=10, ratio=LOWE_RATIO,
                T_abs=12, T_ratio=1.4
            )

        # Evaluate with YOUR mapk:
        for k in K_VALS:
            score = mapk(gt_corresps, pred, k=k, skip_unknown=SKIP_UNKNOWN)
            print(f"✅ SIFT mAP@{k} = {score:.4f}")


# =========================
# RANSAC hyperparameter sweep
# =========================
import csv
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

def ransac_inliers_params(kpQ_xy, kpD_xy, pairs, reproj=3.0, maxIters=2000, conf=0.995, min_tentative=8):
    if len(pairs) < min_tentative:
        return 0, 0.0
    ptsQ = np.float32([kpQ_xy[q] for q, _ in pairs])
    ptsD = np.float32([kpD_xy[d] for _, d in pairs])
    H, mask = cv2.findHomography(ptsQ, ptsD, cv2.RANSAC, reproj, maxIters=maxIters, confidence=conf)
    if mask is None:
        return 0, 0.0
    inl = int(mask.sum())
    return inl, inl / max(1, len(pairs))

def rank_db_ransac_params(kpQ_xy, desQ, db_kps, db_descs, *, ratio, reproj, maxIters, conf, min_tentative, max_tentative=2000):
    # mutual+ratio with cap (copy of your mutual but parameterized)
    if desQ is None or len(desQ) == 0:
        return []
    ranks = []
    for i, (kpD_xy, desD) in enumerate(zip(db_kps, db_descs)):
        if desD is None or len(desD) == 0:
            ranks.append((i, 0, 0.0)); continue
        # forward/back ratio tests
        knn_qd = _FLANN.knnMatch(desQ, desD, k=2)
        knn_dq = _FLANN.knnMatch(desD, desQ, k=2)

        fwd = []
        for m, n in knn_qd:
            if m.distance < ratio * n.distance:
                fwd.append((m.queryIdx, m.trainIdx, m.distance))
                if len(fwd) >= max_tentative:
                    break

        back_best = {}
        for m, n in knn_dq:
            if m.distance < ratio * n.distance:
                back_best[m.queryIdx] = (m.trainIdx, m.distance)

        pairs = []
        for qi, di, _ in fwd:
            bb = back_best.get(di, None)
            if bb is not None and bb[0] == qi:
                pairs.append((qi, di))

        inl, inl_ratio = ransac_inliers_params(
            kpQ_xy, kpD_xy, pairs,
            reproj=reproj, maxIters=maxIters, conf=conf, min_tentative=min_tentative
        )
        ranks.append((i, inl, inl_ratio))
    ranks.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return ranks

def predict_with_unknowns_ransac_params(q_kps, q_descs, db_kps, db_descs, *,
                                        ratio, T_inl, T_ratio, reproj, maxIters, conf, min_tentative,
                                        topK=10):
    predictions = []
    tq = tqdm(zip(q_kps, q_descs), total=len(q_descs), desc="Predict (RANSAC sweep)")
    for kp_crops, desc_crops in tq:
        crop_preds = []
        for kpQ_xy, desQ in zip(kp_crops, desc_crops):
            if desQ is None or len(desQ) == 0 or kpQ_xy is None or len(kpQ_xy) == 0:
                crop_preds.append([-1]); continue
            ranked = rank_db_ransac_params(
                kpQ_xy, desQ, db_kps, db_descs,
                ratio=ratio, reproj=reproj, maxIters=maxIters, conf=conf, min_tentative=min_tentative
            )
            ranked_idx = [i for (i, _, __) in ranked]
            # classic unknown rule on inliers
            if not ranked:
                crop_preds.append([-1])
            else:
                best = ranked[0][1]
                second = ranked[1][1] if len(ranked) > 1 else 0
                if best < T_inl or (second > 0 and (best / (second + 1e-9)) < T_ratio):
                    crop_preds.append([-1])
                else:
                    crop_preds.append(ranked_idx[:topK])
        predictions.append(crop_preds)
    return predictions

def ensure_kpdesc_ready():
    """Ensure we have db_kps/db_descs_kp and q_kps/q_descs in memory."""
    global db_kps, db_descs_kp, q_kps, q_descs, db_imgs
    # DB kp/desc
    cache_path = os.path.join(RESULTS_DIR, "db_kpdesc.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            db_kps = data.get("db_kps")
            db_descs_kp = data.get("db_descs")
        except Exception:
            db_kps = None; db_descs_kp = None
    if db_kps is None or db_descs_kp is None or len(db_kps) != len(db_descs_kp):
        if db_imgs is None:
            _, db_imgs = load_images_from_folder(DB_FOLDER)
        db_kps, db_descs_kp = build_db_kp_desc_resumable(db_imgs, cache_path=cache_path, max_side=SIFT_MAX_SIDE)

    # Query kp/desc
    if 'q_kps' not in globals() or 'q_descs' not in globals() or q_kps is None or q_descs is None:
        print("🔁 Building query kp/desc for sweep...")
        q_kps, q_descs = build_query_kp_desc(q_imgs, q_names)

def sweep_ransac(q_kps, q_descs, db_kps, db_descs, gt_corresps,
                 ratios=(0.74, 0.76, 0.78),
                 T_inls=(12, 16, 20, 24),
                 T_ratios=(1.15, 1.25, 1.35),
                 reprojs=(2.0, 3.0, 4.0),
                 iters=(1000, 2000),
                 min_tents=(6, 8, 10),
                 topK=10,
                 save_csv=os.path.join(RESULTS_DIR, "sweep_ransac.csv"),
                 make_heatmaps=True):
    """
    Returns: sorted list of dicts with metrics and params (best first).
    """
    combos = list(product(ratios, T_inls, T_ratios, reprojs, iters, min_tents))
    pbar = tqdm(total=len(combos), desc=f"Sweep {len(combos)} configs", ncols=100)
    results = []
    for (ratio, T_inl, T_ratio, reproj, maxIters, minTent) in combos:
        pred = predict_with_unknowns_ransac_params(
            q_kps, q_descs, db_kps, db_descs,
            ratio=ratio, T_inl=T_inl, T_ratio=T_ratio,
            reproj=reproj, maxIters=maxIters, conf=RANSAC_CONF,
            min_tentative=minTent, topK=topK
        )
        m1 = mapk(gt_corresps, pred, k=1, skip_unknown=SKIP_UNKNOWN)
        m5 = mapk(gt_corresps, pred, k=5, skip_unknown=SKIP_UNKNOWN)
        results.append({
            "mAP@1": m1, "mAP@5": m5,
            "ratio": ratio, "T_inl": T_inl, "T_ratio": T_ratio,
            "reproj": reproj, "iters": maxIters, "min_tent": minTent
        })
        # save in a csv after each iteration
        if save_csv:
            fieldnames = ["mAP@5", "mAP@1", "ratio", "T_inl", "T_ratio", "reproj", "iters", "min_tent"]
            with open(save_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in results:
                    w.writerow({
                        "mAP@5": f"{r['mAP@5']:.6f}",
                        "mAP@1": f"{r['mAP@1']:.6f}",
                        "ratio": r["ratio"],
                        "T_inl": r["T_inl"],
                        "T_ratio": r["T_ratio"],
                        "reproj": r["reproj"],
                        "iters": r["iters"],
                        "min_tent": r["min_tent"],
                    })
        pbar.set_postfix({"best_mAP@5": f"{max(r['mAP@5'] for r in results):.3f}, "
                                     f"best_config: ratio={results[np.argmax([r['mAP@5'] for r in results])]['ratio']}, "
                                     f"T_inl={results[np.argmax([r['mAP@5'] for r in results])]['T_inl']}, "
                                     f"T_ratio={results[np.argmax([r['mAP@5'] for r in results])]['T_ratio']}, "
                                     f"reproj={results[np.argmax([r['mAP@5'] for r in results])]['reproj']}, "
                                     f"iters={results[np.argmax([r['mAP@5'] for r in results])]['iters']}, "
                                     f"min_tent={results[np.argmax([r['mAP@5'] for r in results])]['min_tent']}"})
        pbar.update(1)
    pbar.close()

    # sort by mAP@5 then mAP@1
    results.sort(key=lambda d: (d["mAP@5"], d["mAP@1"]), reverse=True)

    # Save CSV
    if save_csv:
        fieldnames = ["mAP@5", "mAP@1", "ratio", "T_inl", "T_ratio", "reproj", "iters", "min_tent"]
        with open(save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({
                    "mAP@5": f"{r['mAP@5']:.6f}",
                    "mAP@1": f"{r['mAP@1']:.6f}",
                    "ratio": r["ratio"],
                    "T_inl": r["T_inl"],
                    "T_ratio": r["T_ratio"],
                    "reproj": r["reproj"],
                    "iters": r["iters"],
                    "min_tent": r["min_tent"],
                })
        print(f"📄 Saved sweep CSV to {save_csv}")

    # Optional heatmaps per ratio for (T_inl x T_ratio) at best (reproj,iters,minTent)
    if make_heatmaps:
        os.makedirs(os.path.join(RESULTS_DIR, "heatmaps"), exist_ok=True)
        # find best triple (reproj, iters, minTent) globally
        best_global = max(results, key=lambda d: d["mAP@5"])
        best_reproj, best_iters, best_minTent = best_global["reproj"], best_global["iters"], best_global["min_tent"]

        T_inls_list = sorted(set(T_inls))
        T_ratios_list = sorted(set(T_ratios))
        for r in sorted(set(ratios)):
            grid = np.zeros((len(T_inls_list), len(T_ratios_list)), dtype=np.float32)
            for i, Ti in enumerate(T_inls_list):
                for j, Tr in enumerate(T_ratios_list):
                    row = [x for x in results if x["ratio"]==r and x["T_inl"]==Ti and x["T_ratio"]==Tr
                           and x["reproj"]==best_reproj and x["iters"]==best_iters and x["min_tent"]==best_minTent]
                    grid[i,j] = row[0]["mAP@5"] if row else np.nan
            plt.figure()
            plt.imshow(grid, aspect="auto")
            plt.colorbar(label="mAP@5")
            plt.xticks(range(len(T_ratios_list)), [str(v) for v in T_ratios_list])
            plt.yticks(range(len(T_inls_list)), [str(v) for v in T_inls_list])
            plt.xlabel("T_ratio (best/second)")
            plt.ylabel("T_inl (min inliers)")
            plt.title(f"mAP@5 heatmap, ratio={r}, reproj={best_reproj}, iters={best_iters}, minTent={best_minTent}")
            out_png = os.path.join(RESULTS_DIR, "heatmaps", f"heatmap_ratio-{r}.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()
            print(f"🖼️ Saved {out_png}")

    # Print top-10
    print("\nTop 10 configs by mAP@5:")
    for r in results[:10]:
        print(f"mAP@5={r['mAP@5']:.4f}  mAP@1={r['mAP@1']:.4f}  "
              f"ratio={r['ratio']}  T_inl={r['T_inl']}  T_ratio={r['T_ratio']}  "
              f"reproj={r['reproj']}  iters={r['iters']}  minTent={r['min_tent']}")
    return results

# -------- Run the sweep (call this at the end of __main__) --------
if __name__ == "__main__":
    # Make sure kp/desc are ready (reuses existing caches/builders from your script)
    ensure_kpdesc_ready()

    # Adjust grids as you like (start small, expand if helpful)
    results = sweep_ransac(
        q_kps, q_descs, db_kps, db_descs_kp, gt_corresps,
        ratios=(0.74, 0.78),
        T_inls=(12, 20),
        T_ratios=(1.10, 1.40),
        reprojs=(2.0, 3.0, 4.0),
        iters=(1000),
        min_tents=(6, 10),
        topK=10,
        save_csv=os.path.join(RESULTS_DIR, "sweep_ransac.csv"),
        make_heatmaps=True
    )

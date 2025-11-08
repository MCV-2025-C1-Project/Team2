# Imports
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from noise_filter import preprocess_image
import background_remover_w2 as background_remover
from image_split_team5 import segment_multiple_paintings
# from image_split_team5_v2 import segment_multiple_paintings

# =========================
# Matchers & Evaluation
# =========================

def plot_score_vs_features(
    predicted_scores,
    predicted_num_features,
    gt_corresps,
    title: str = "SIFT Scores vs. Num. Features",
    ax=None
):
    """
    Scatterplot showing correlation between predicted scores and number of features.
    Points are colored by ground truth: red = unknown (-1), green = known.

    Parameters
    ----------
    predicted_scores : list[list[float]] or np.ndarray
        Per-(sub)query predicted scores (same structure as gt_corresps).
    predicted_num_features : list[list[int]] or np.ndarray
        Per-(sub)query number of detected features.
    gt_corresps : list[list[int]]
        Per-(sub)query ground-truth indices; -1 denotes unknown.
    title : str
        Title for the plot.
    ax : matplotlib Axes
        Optional Axes to draw on.
    """

    # Flatten everything
    scores = np.array([s for sub in predicted_scores for s in sub], dtype=float)
    nfeatures = np.array([n for sub in predicted_num_features for n in sub], dtype=float)
    y_true = np.array([0 if x == -1 else 1 for sub in gt_corresps for x in sub], dtype=int)

    if scores.size == 0 or nfeatures.size == 0:
        raise ValueError("Empty arrays provided.")

    # Prepare colors
    colors = np.where(y_true == 0, "red", "green")

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    scatter = ax.scatter(nfeatures, scores, c=colors, alpha=0.7, edgecolor='k', linewidth=0.4)
    ax.set_xlabel("Number of Features (SIFT)")
    ax.set_ylabel("Predicted Score")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Known', markerfacecolor='green', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Unknown', markerfacecolor='red', markersize=7)
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()


def find_best_score_threshold(
    rates,
    gt_corresps,
    metric: str = "f1",     # "f1" | "accuracy" | "youden"
    plot: bool = False,
    title_xlabel: str = "Query Index",
    title_ylabel: str = "Normalized Score",
    ax=None
):
    """
    Determine the optimal threshold on match 'rates' to classify:
      known (label=1) vs unknown (label=0, gt == -1).

    Parameters
    ----------
    rates : list[list[float]]
        Per-(sub)query normalized scores (same structure as returned by compute_matches).
    gt_corresps : list[list[int]]
        Per-(sub)query ground-truth indices; -1 denotes unknown (same structure as your GT).
    metric : str
        Criterion to optimize: "f1" (default), "accuracy", or "youden" (TPR - FPR).
    plot : bool
        If True, draws a scatter of scores and a horizontal line at the best threshold.
    ax : matplotlib Axes
        Optional axes to plot on.

    Returns
    -------
    best_threshold : float
    decision_side : str        # ">=" or "<=" meaning 'predict known if score SIDE threshold'
    metrics : dict             # F1, precision, recall, accuracy at best threshold
    y_pred : np.ndarray        # predictions (1 known / 0 unknown) for the *flattened* scores
    """

    # Flatten scores and build binary labels: 1=known, 0=unknown
    scores = np.array([s for sub in rates for s in sub], dtype=float)
    y_true = np.array([0 if x == -1 else 1 for sub in gt_corresps for x in sub], dtype=int)

    if scores.size == 0:
        raise ValueError("Empty 'rates' provided.")

    # Decide direction automatically: do knowns tend to have higher scores?
    mean_known = scores[y_true == 1].mean() if np.any(y_true == 1) else -np.inf
    mean_unk   = scores[y_true == 0].mean() if np.any(y_true == 0) else  np.inf
    predict_known_if_higher = (mean_known >= mean_unk)
    side_symbol = ">=" if predict_known_if_higher else "<="

    def preds_from_threshold(th):
        if predict_known_if_higher:
            return (scores >= th).astype(int)
        else:
            return (scores <= th).astype(int)

    # Candidate thresholds: midpoints between sorted unique scores + two safe extremes
    uniq = np.unique(scores)
    if uniq.size == 1:
        # Degenerate case: all scores equal -> any threshold yields same preds
        th_edges = np.array([uniq[0]])
    else:
        th_edges = np.concatenate([[uniq.min() - 1e-12],
                                   (uniq[:-1] + uniq[1:]) / 2.0,
                                   [uniq.max() + 1e-12]])

    def youden(y_true, y_hat):
        # TPR - FPR
        tp = np.sum((y_true == 1) & (y_hat == 1))
        fn = np.sum((y_true == 1) & (y_hat == 0))
        fp = np.sum((y_true == 0) & (y_hat == 1))
        tn = np.sum((y_true == 0) & (y_hat == 0))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return tpr - fpr

    best_key = -np.inf
    best = {"threshold": None, "f1": None, "precision": None, "recall": None, "accuracy": None}
    best_pred = None

    for th in th_edges:
        y_hat = preds_from_threshold(th)
        f1  = f1_score(y_true, y_hat, zero_division=0)
        pre = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        acc = (y_true == y_hat).mean()

        key = {"f1": f1, "accuracy": acc, "youden": youden(y_true, y_hat)}[metric]
        if key > best_key:
            best_key = key
            best = {"threshold": th, "f1": f1, "precision": pre, "recall": rec, "accuracy": acc}
            best_pred = y_hat

    # Optional plot
    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        colors = ['red' if y == 0 else 'green' for y in y_true]
        ax.scatter(range(len(scores)), scores, c=colors, alpha=0.6)
        ax.axhline(best["threshold"], linestyle="--")
        ax.set_title(f"Scores with Best Threshold ({metric.upper()})")
        ax.set_xlabel(title_xlabel)
        ax.set_ylabel(title_ylabel)

    return float(best["threshold"]), side_symbol, best, best_pred

def create_matcher(descriptor_type: str):
    """
    Return an OpenCV matcher appropriate for the descriptor type.
    - SIFT/CSIFT: FLANN (KD-tree, L2)
    - ORB:        BFMatcher with Hamming distance
    """
    descriptor_type = descriptor_type.upper()
    if descriptor_type in ("SIFT", "CSIFT"):
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=64))  # KDTree
    elif descriptor_type == "ORB":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"Unsupported descriptor_type: {descriptor_type}")


def good_match_count(desQ, desD, matcher, ratio=0.76):
    """
    2-NN + Lowe ratio; returns number of 'good' matches.
    Works for both float (SIFT/CSIFT) and binary (ORB) descriptors.
    """
    if desQ is None or desD is None or len(desQ) == 0 or len(desD) == 0:
        return 0
    knn = matcher.knnMatch(desQ, desD, k=2)
    return sum(1 for m, n in knn if m.distance < ratio * n.distance)


def rank_db_for_query_multi(desc_crops, db_descs, matcher, ratio=0.76):
    scores = []
    for i, desD in enumerate(db_descs):
        best = max(good_match_count(desQ, desD, matcher, ratio) for desQ in desc_crops)
        scores.append((i, best))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual, predicted, k=10, skip_unknown=True):
    map_score, total_pics = 0.0, 0
    for idx, p in enumerate(predicted):
        if len(p) == 2:
            apk_score1 = apk([actual[idx][0]], p[0], k)
            apk_score2 = apk([actual[idx][1]], p[1], k)
            map_score += apk_score1 + apk_score2
            total_pics += 2
        elif actual[idx] == [-1] and skip_unknown:
            continue
        else:
            apk_score = apk(actual[idx], p, k)
            map_score += apk_score
            total_pics += 1
    return map_score / total_pics if total_pics > 0 else 0.0


def compute_matches(desc_query, db_descs, gt_corresps, matcher, ratio=0.76, rate_threshold=0.125, descriptor_type=None):
    """
    desc_query:  list where each element is [desc_single] or [desc_left, desc_right]
    db_descs:    list of descriptor arrays, one per DB image
    gt_corresps: list; each entry is an int, list of ints, or [-1]
    """
    predicted = []
    predicted_norm_scores = []
    predicted_scores = []
    predicted_num_features = []
    min_score_rate_correct = float('inf')
    max_score_rate_incorrect = float('-inf')

    # if the descriptor_type is provided, check if the pickle file of the predicted results exists
    if descriptor_type:
        pickle_path = f"results/predicted_results_{descriptor_type}_rate{rate_threshold:.3f}.pkl"
        if os.path.exists(pickle_path):
            print(f"✅ Loading predicted results from {pickle_path}")
            with open(pickle_path, "rb") as f:
                predicted = pickle.load(f)
            return predicted

    for qi, desc_crops in tqdm(enumerate(desc_query), total=len(desc_query), desc="Processing queries"):
        q_gt = gt_corresps[qi]
        if len(q_gt) == 2 and len(desc_crops) == 2:
            list_ranked_indices = []
            list_ranked_norm_scores = []
            list_ranked_scores = []
            list_ranked_num_features = []
            for desc_crop in desc_crops:
                ranked = rank_db_for_query_multi([desc_crop], db_descs, matcher, ratio=ratio)
                ranked_indices, ranked_scores = zip(*ranked)
                rate = ranked_scores[0] / desc_crop.shape[0] if desc_crop is not None and desc_crop.shape[0] > 0 else 0

                if ranked_indices[0] in q_gt:
                    min_score_rate_correct = min(min_score_rate_correct, rate)
                else:
                    max_score_rate_incorrect = max(max_score_rate_incorrect, rate)
                
                if rate < rate_threshold:
                    ranked_indices = [-1]
                list_ranked_indices.append(ranked_indices)
                list_ranked_norm_scores.append(rate)
                list_ranked_scores.append(ranked_scores[0])
                list_ranked_num_features.append(desc_crop.shape[0] if desc_crop is not None else 0)

                # print the rate, gt and predicted for each crop
                # print(f"Query {qi} GT: {q_gt}, Predicted: {ranked_indices[0]}, Rate: {rate:.4f}")

        else:
            ranked = rank_db_for_query_multi(desc_crops, db_descs, matcher, ratio=ratio)
            list_ranked_indices, list_ranked_scores = zip(*ranked)
            base = desc_crops[0] if (len(desc_crops) > 0 and desc_crops[0] is not None) else None
            rate = (list_ranked_scores[0] / base.shape[0]) if (base is not None and base.shape[0] > 0) else 0
            
            if list_ranked_indices[0] in q_gt:
                min_score_rate_correct = min(min_score_rate_correct, rate)
            else:
                max_score_rate_incorrect = max(max_score_rate_incorrect, rate)
            
            if rate < rate_threshold:
                list_ranked_indices = [-1]
            
            list_ranked_norm_scores = [rate]
            list_ranked_scores = [list_ranked_scores[0]]
            list_ranked_num_features = [base.shape[0] if base is not None else 0]

            # print the rate, gt and predicted for the query
            # print(f"Query {qi} GT: {q_gt}, Predicted: {list_ranked_indices[0]}, Rate: {rate:.4f}")

        predicted.append(list_ranked_indices)
        predicted_norm_scores.append(list_ranked_norm_scores)
        predicted_scores.append(list_ranked_scores)
        predicted_num_features.append(list_ranked_num_features)

    print(f"Min score for correct matches: {min_score_rate_correct}")
    print(f"Max score for incorrect matches: {max_score_rate_incorrect}")

    # save the predicted results to a file
    if descriptor_type:
        with open(f"results/predicted_results_{descriptor_type}_rate{rate_threshold:.3f}.pkl", "wb") as f:
            pickle.dump(predicted, f)
    return predicted, predicted_norm_scores, predicted_scores, predicted_num_features


# =========================
# I/O & Preprocessing
# =========================

def load_images_from_folder(folder, exts=(".jpg",)):
    names = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    imgs = []
    for name in names:
        path = os.path.join(folder, name)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Could not read {path}")
            continue
        imgs.append(img)
    return names, imgs


def validate_split(is_split, imgs, min_size_ratio=0.2):
    if not is_split:
        return False, imgs
    if not isinstance(imgs, (list, tuple)) or len(imgs) != 2:
        return False, imgs

    img1, img2 = imgs
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    total_width = w1 + w2
    total_height = max(h1, h2)

    valid_left   = (w1 / total_width)  > min_size_ratio
    valid_right  = (w2 / total_width)  > min_size_ratio
    valid_h1     = (h1 / total_height) > min_size_ratio
    valid_h2     = (h2 / total_height) > min_size_ratio

    valid_imgs = []
    if valid_left and valid_h1:
        valid_imgs.append(img1)
    if valid_right and valid_h2:
        valid_imgs.append(img2)

    if len(valid_imgs) == 2:
        print("Both segmented images are valid after size check.")
        return True, valid_imgs
    elif len(valid_imgs) == 1:
        print("HADSJHADSHJDJHFDSAJHADFBDFBBBJDFJHDFHSHDSHDASHHFADFBABJDBJDBJFADBJDAFJDFJFBBJDJBDSFJD")
        return False, valid_imgs[0]
    else:
        return False, imgs


# =========================
# Descriptors (SIFT / ORB / Color-SIFT)
# =========================

def _to_uint8(img):
    """Normalize/convert single-channel float image to uint8 safely."""
    if img.dtype == np.uint8:
        return img
    m, M = np.min(img), np.max(img)
    if M <= m + 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    out = (255.0 * (img - m) / (M - m)).astype(np.uint8)
    return out

def compute_sift_desc(
    img_bgr,
    nfeatures=1200,
    rootsift=False,
    img_name="",
    show_img=False,
    save_vis=False,
    vis_dir="results/query_keypoints",
    info=False,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        if info:
            print(f"⚠️ No SIFT descriptors for {img_name}.")
        return None

    des = des.astype(np.float32)
    if rootsift:
        des /= (des.sum(axis=1, keepdims=True) + 1e-12)
        des = np.sqrt(des, out=des)

    if show_img or save_vis:
        vis = cv2.drawKeypoints(gray, kp, img_bgr.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, f"{img_name}_keypoints.jpg"), vis)
        if show_img:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"SIFT Keypoints - {img_name}")
            plt.axis('off')
            plt.show()

    if info:
        print(f"SIFT descriptors for {img_name}: {des.shape[0]} keypoints.")
    return des


def compute_orb_desc(
    img_bgr,
    nfeatures=2000,
    fast_threshold=20,
    score_type="HARRIS",   # or "FAST"
    img_name="",
    show_img=False,
    save_vis=False,
    vis_dir="results/query_keypoints",
    info=False,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.ORB_HARRIS_SCORE if score_type.upper() == "HARRIS" else cv2.ORB_FAST_SCORE
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=score,
        patchSize=31,
        fastThreshold=fast_threshold
    )
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        if info:
            print(f"⚠️ No ORB descriptors for {img_name}.")
        return None

    if show_img or save_vis:
        vis = cv2.drawKeypoints(gray, kp, img_bgr.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, f"{img_name}_keypoints.jpg"), vis)
        if show_img:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"ORB Keypoints - {img_name}")
            plt.axis('off')
            plt.show()

    if info:
        print(f"ORB descriptors for {img_name}: {des.shape[0]} keypoints.")
    return des


def compute_color_sift_desc(
    img_bgr,
    nfeatures=1200,
    color_space="OPPONENT",   # "OPPONENT" | "RGB" | "HSV"
    rootsift=False,
    img_name="",
    show_img=False,
    save_vis=False,
    vis_dir="results/query_keypoints",
    info=False,
):
    """
    Color-SIFT: detect keypoints once (intensity), then compute SIFT on 3 channels
    and concatenate per-keypoint (3 * 128 = 384 dims). Float32, L2.
    """
    # 1) Keypoints on intensity (grayscale or opponent O3)
    bgr = img_bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp = sift.detect(gray, None)
    if kp is None or len(kp) == 0:
        if info:
            print(f"⚠️ No keypoints for Color-SIFT: {img_name}.")
        return None

    # 2) Build 3 channels per chosen space as uint8 single-channel images
    color_space = color_space.upper()
    if color_space == "RGB":
        r, g, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        chs = [r, g, b]
    elif color_space == "HSV":
        h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
        chs = [h, s, v]
    else:
        # Opponent color space (van de Sande et al. 2010)
        b, g, r = cv2.split(bgr.astype(np.float32))
        o1 = (r - g) / np.sqrt(2.0)
        o2 = (r + g - 2.0 * b) / np.sqrt(6.0)
        o3 = (r + g + b) / np.sqrt(3.0)
        chs = [_to_uint8(o1), _to_uint8(o2), _to_uint8(o3)]

    # 3) Compute SIFT on each channel using the SAME keypoints; then concat
    desc_list = []
    for ch in chs:
        _kp, des_ch = sift.compute(ch, kp)  # use same keypoints
        if des_ch is None:
            if info:
                print(f"⚠️ Channel produced no descriptors for {img_name}.")
            return None
        desc_list.append(des_ch.astype(np.float32))

    # Ensure same #rows for all channels
    min_rows = min(d.shape[0] for d in desc_list)
    if min_rows == 0:
        if info:
            print(f"⚠️ No descriptors after channel alignment for {img_name}.")
        return None
    desc_list = [d[:min_rows] for d in desc_list]
    des = np.concatenate(desc_list, axis=1)  # (N, 384)

    # 4) RootSIFT (on concatenated descriptor)
    if rootsift:
        des /= (des.sum(axis=1, keepdims=True) + 1e-12)
        des = np.sqrt(des, out=des)

    # (Optional) visualization using original gray & kp
    if show_img or save_vis:
        vis = cv2.drawKeypoints(gray, kp[:min_rows], bgr.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, f"{img_name}_keypoints.jpg"), vis)
        if show_img:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Color-SIFT Keypoints - {img_name} ({color_space})")
            plt.axis('off')
            plt.show()

    if info:
        print(f"Color-SIFT ({color_space}) for {img_name}: {des.shape[0]} keypoints, {des.shape[1]} dims.")
    return des


# =========================
# Descriptor Pipelines
# =========================

def build_or_load_db_descriptors(
    db_folder,
    desc_db_path,
    descriptor_type="SIFT",
    nfeatures=1200,
    rootsift=True,
    color_space="OPPONENT",
    orb_fast_threshold=20,
    orb_score_type="HARRIS",
    use_cache=True,
    progress=True,
):
    if use_cache and os.path.exists(desc_db_path):
        print(f"✅ Loading DB descriptors from {desc_db_path}")
        with open(desc_db_path, "rb") as f:
            data = pickle.load(f)
            return data["desc_gt"], data["gt_names"]

    print(f"🧠 Computing {descriptor_type} descriptors for database images...")
    db_names, db_imgs = load_images_from_folder(db_folder)
    db_descs = []
    iterator = tqdm(zip(db_imgs, db_names), total=len(db_names)) if progress else zip(db_imgs, db_names)

    descriptor_type = descriptor_type.upper()
    for img, name in iterator:
        stem = os.path.splitext(name)[0]
        if descriptor_type == "SIFT":
            d = compute_sift_desc(img, nfeatures=nfeatures, rootsift=rootsift, img_name=stem)
        elif descriptor_type == "ORB":
            d = compute_orb_desc(img, nfeatures=nfeatures, fast_threshold=orb_fast_threshold,
                                 score_type=orb_score_type, img_name=stem)
        elif descriptor_type == "CSIFT":
            d = compute_color_sift_desc(img, nfeatures=nfeatures, color_space=color_space,
                                        rootsift=rootsift, img_name=stem)
        else:
            raise ValueError(f"Unsupported descriptor_type: {descriptor_type}")
        db_descs.append(d)

    with open(desc_db_path, "wb") as f:
        pickle.dump({"desc_gt": db_descs, "gt_names": db_names}, f)
    print(f"✅ Saved DB descriptors to {desc_db_path}")
    return db_descs, db_names


def single_img_removal(
    img,
    img_name,
    background_remover,
    descriptor_type="SIFT",
    nfeatures=1200,
    rootsift=True,
    color_space="OPPONENT",
    orb_fast_threshold=20,
    orb_score_type="HARRIS",
    bg_crop_min_ratio=0.10,
    show_img=False,
    info=False,
):
    im, mask, *_ = background_remover.remove_background_morphological_gradient(img)
    cropped = background_remover.crop_to_mask_rectangle(img, mask)

    if (cropped.shape[1] / img.shape[1] < bg_crop_min_ratio) or (cropped.shape[0] / img.shape[0] < bg_crop_min_ratio):
        cropped = img

    stem = os.path.splitext(img_name)[0] + "_single"
    descriptor_type = descriptor_type.upper()
    if descriptor_type == "SIFT":
        return compute_sift_desc(cropped, nfeatures=nfeatures, rootsift=rootsift,
                                 img_name=stem, show_img=show_img, save_vis=show_img, info=info)
    elif descriptor_type == "ORB":
        return compute_orb_desc(cropped, nfeatures=nfeatures, fast_threshold=orb_fast_threshold,
                                score_type=orb_score_type, img_name=stem,
                                show_img=show_img, save_vis=show_img, info=info)
    elif descriptor_type == "CSIFT":
        return compute_color_sift_desc(cropped, nfeatures=nfeatures, color_space=color_space,
                                       rootsift=rootsift, img_name=stem,
                                       show_img=show_img, save_vis=show_img, info=info)
    else:
        raise ValueError(f"Unsupported descriptor_type: {descriptor_type}")


def compute_query_descriptors(
    q_imgs,
    q_names,
    preprocess_image,
    segment_multiple_paintings,
    validate_split_fn=validate_split,
    background_remover=None,
    descriptor_type="SIFT",
    nfeatures=1200,
    rootsift=True,
    color_space="OPPONENT",
    orb_fast_threshold=20,
    orb_score_type="HARRIS",
    crop_min_ratio_single=0.20,
    bg_crop_min_ratio=0.10,
    show_img=False,
    info=False,
    save=False,
):
    
    #check if the pickle file of the query descriptors exists
    if save:
        pickle_path = f"results/query_descriptors_{descriptor_type}.pkl"
        if os.path.exists(pickle_path):
            print(f"✅ Loading query descriptors from {pickle_path}")
            with open(pickle_path, "rb") as f:
                desc_query = pickle.load(f)
            return desc_query
    desc_query = []
    descriptor_type = descriptor_type.upper()

    for img, img_name in tqdm(zip(q_imgs, q_names), total=len(q_names), desc="Processing queries"):
        original_h, original_w = img.shape[:2]
        img_proc = preprocess_image(img)
        is_split, split_imgs = segment_multiple_paintings(img_proc)
        ok_split, parts = validate_split_fn(is_split, split_imgs)

        if ok_split:
            left_artwork, right_artwork = parts

            if background_remover is not None:
                iml, left_mask, *_  = background_remover.remove_background_morphological_gradient(left_artwork)
                imr, right_mask, *_ = background_remover.remove_background_morphological_gradient(right_artwork)
                left_cropped  = background_remover.crop_to_mask_rectangle(left_artwork, left_mask)
                right_cropped = background_remover.crop_to_mask_rectangle(right_artwork, right_mask)
            else:
                left_cropped, right_cropped = left_artwork, right_artwork

            too_small = (
                (left_cropped.shape[1]  / original_w < crop_min_ratio_single) or
                (left_cropped.shape[0]  / original_h < crop_min_ratio_single) or
                (right_cropped.shape[1] / original_w < crop_min_ratio_single) or
                (right_cropped.shape[0] / original_h < crop_min_ratio_single)
            )

            if too_small:
                if info:
                    print("Processing image", img_name, "as single due to small crop size")
                desc_single = single_img_removal(
                    img_proc, img_name, background_remover,
                    descriptor_type=descriptor_type, nfeatures=nfeatures, rootsift=rootsift, 
                    bg_crop_min_ratio=bg_crop_min_ratio, color_space=color_space,
                    orb_fast_threshold=orb_fast_threshold, orb_score_type=orb_score_type,
                    show_img=show_img, info=info
                )
                desc_query.append([desc_single])
            else:
                stem = os.path.splitext(img_name)[0]
                if descriptor_type == "SIFT":
                    desc_left = compute_sift_desc(
                        left_cropped, nfeatures=nfeatures, rootsift=rootsift,
                        img_name=stem + "_left", show_img=show_img, save_vis=show_img, info=info
                    )
                    desc_right = compute_sift_desc(
                        right_cropped, nfeatures=nfeatures, rootsift=rootsift,
                        img_name=stem + "_right", show_img=show_img, save_vis=show_img, info=info
                    )
                elif descriptor_type == "ORB":
                    desc_left = compute_orb_desc(
                        left_cropped, nfeatures=nfeatures, fast_threshold=orb_fast_threshold,
                        score_type=orb_score_type, img_name=stem + "_left",
                        show_img=show_img, save_vis=show_img, info=info
                    )
                    desc_right = compute_orb_desc(
                        right_cropped, nfeatures=nfeatures, fast_threshold=orb_fast_threshold,
                        score_type=orb_score_type, img_name=stem + "_right",
                        show_img=show_img, save_vis=show_img, info=info
                    )
                elif descriptor_type == "CSIFT":
                    desc_left = compute_color_sift_desc(
                        left_cropped, nfeatures=nfeatures, color_space=color_space,
                        rootsift=rootsift, img_name=stem + "_left",
                        show_img=show_img, save_vis=show_img, info=info
                    )
                    desc_right = compute_color_sift_desc(
                        right_cropped, nfeatures=nfeatures, color_space=color_space,
                        rootsift=rootsift, img_name=stem + "_right",
                        show_img=show_img, save_vis=show_img, info=info
                    )
                else:
                    raise ValueError(f"Unsupported descriptor_type: {descriptor_type}")
                desc_query.append([desc_left, desc_right])
        else:
            desc_single = single_img_removal(
                img_proc, img_name, background_remover,
                descriptor_type=descriptor_type, nfeatures=nfeatures, rootsift=rootsift, 
                bg_crop_min_ratio=bg_crop_min_ratio, color_space=color_space,
                orb_fast_threshold=orb_fast_threshold, orb_score_type=orb_score_type,
                show_img=show_img, info=info
            )
            desc_query.append([desc_single])
    
    # if save, store descriptors to a pickle file
    if save:
        pickle_path = f"results/query_descriptors_{descriptor_type}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(desc_query, f)
        print(f"✅ Saved query descriptors to {pickle_path}")
    return desc_query


def evaluate_unknown_threshold(gt_corresps, pred, threshold_name, descriptor, normalize=True):
    """Evaluate and plot confusion matrix for known/unknown detection."""
    y_true, y_pred = [], []
    for i, gt in enumerate(gt_corresps):
        if len(gt) == 2:
            # print(f"Query {i} GT: {gt}, Predicted Left: {pred[i][0][0]}, Predicted Right: {pred[i][1][0]}")
            y_true += [1 if gt[0] != -1 else -1, 1 if gt[1] != -1 else -1]
            y_pred += [1 if pred[i][0][0] != -1 else -1, 1 if pred[i][1][0] != -1 else -1]
        else:
            # print(f"Query {i} GT: {gt}, Predicted: {pred[i][0]}")
            y_true.append(1 if gt[0] != -1 else -1)
            y_pred.append(1 if pred[i][0] != -1 else -1)

    # Confusion matrix
    labels = [-1, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({descriptor}) – threshold={threshold_name}")
    plt.tight_layout()
    plt.savefig(f"results/plots/confusion_matrix_{descriptor}_thr{threshold_name}.png", dpi=200)
    plt.close()

    # Metrics
    f1 = f1_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    print(f"Threshold {threshold_name} → Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    return f1

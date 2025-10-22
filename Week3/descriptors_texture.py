#  (LBP + DCT combined)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

#  CONFIGURATION

QSD1_PATH = 'data/qsd1_w3/'       # folder with query images
BBDD_PATH = 'data/bbdd/'          # folder with reference paintings
GT_PATH   = 'data/qsd1_w3/gt.npy' # ground truth indices

# Descriptor parameters
LBP_P = 8         # neighbors
LBP_R = 1         # radius
GRID  = (4, 4)    # grid for spatial LBP
DCT_N = 64        # number of DCT coefficients

# FEATURE EXTRACTORS

def lbp_descriptor(img, P=8, R=1, grid=(4,4)):
    """Compute LBP histogram descriptor divided into grid blocks."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    h_step, w_step = h // grid[0], w // grid[1]
    desc = []

    for i in range(grid[0]):
        for j in range(grid[1]):
            block = gray[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            lbp = local_binary_pattern(block, P, R, method='uniform')
            hist, _ = np.histogram(lbp.ravel(),
                                   bins=np.arange(0, P + 3),
                                   range=(0, P + 2))
            hist = hist.astype('float')
            hist /= hist.sum() + 1e-6
            desc.extend(hist)
    return np.array(desc, dtype=np.float32)


def dct_descriptor(img, N=64):
    """Compute DCT descriptor using first N coefficients."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    dct = cv2.dct(np.float32(gray))
    dct_flat = np.abs(dct).flatten()
    desc = dct_flat[:N]
    desc /= np.linalg.norm(desc) + 1e-6
    return desc.astype(np.float32)


def hybrid_descriptor(img, lbp_weight=0.5, dct_weight=0.5):
    """Combine normalized LBP + DCT into a single hybrid descriptor."""
    lbp_desc = lbp_descriptor(img, P=LBP_P, R=LBP_R, grid=GRID)
    dct_desc = dct_descriptor(img, N=DCT_N)

    # Normalize individually
    lbp_desc /= np.linalg.norm(lbp_desc) + 1e-6
    dct_desc /= np.linalg.norm(dct_desc) + 1e-6

    # Weighted concatenation
    combined = np.concatenate([lbp_weight * lbp_desc,
                               dct_weight * dct_desc])
    combined /= np.linalg.norm(combined) + 1e-6
    return combined


# UTILITY FUNCTIONS

def load_images_from_folder(folder):
    imgs = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            imgs.append(cv2.imread(os.path.join(folder, fname)))
    return imgs


def compute_descriptors(imgs, method='hybrid'):
    descs = []
    for img in tqdm(imgs, desc=f'Computing {method.upper()} features'):
        if method == 'lbp':
            desc = lbp_descriptor(img)
        elif method == 'dct':
            desc = dct_descriptor(img)
        elif method == 'hybrid':
            desc = hybrid_descriptor(img)
        else:
            raise ValueError("method must be 'lbp', 'dct', or 'hybrid'")
        descs.append(desc)
    return np.array(descs)


def rank_images(query_desc, db_descs):
    sims = cosine_similarity([query_desc], db_descs)[0]
    ranked_idx = np.argsort(-sims)
    return ranked_idx, sims[ranked_idx]


def evaluate_retrieval(results, gt):
    correct = 0
    for i in range(len(gt)):
        if gt[i][0] in results[i][:1]:
            correct += 1
    return correct / len(gt)


# MAIN PIPELINE

if __name__ == '__main__':
    # Load images and ground truth
    queries = load_images_from_folder(QSD1_PATH)
    db_imgs = load_images_from_folder(BBDD_PATH)
    gt = np.load(GT_PATH, allow_pickle=True)

    # Choose method ('lbp', 'dct', 'hybrid')
    method = 'hybrid'

    # Compute features
    db_descs = compute_descriptors(db_imgs, method)
    query_descs = compute_descriptors(queries, method)

    # Retrieval
    results = []
    for qdesc in tqdm(query_descs, desc='Retrieving'):
        ranked_idx, _ = rank_images(qdesc, db_descs)
        results.append(ranked_idx[:10])

    # Evaluate performance
    prec1 = evaluate_retrieval(results, gt)
    print(f"{method.upper()} Precision@1 = {prec1:.3f}")


    # VISUALIZATION

    idx = 0  # first query
    qimg = queries[idx]
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB))
    plt.title("Query")
    plt.axis('off')

    for i in range(5):
        match_idx = results[idx][i]
        plt.subplot(1, 6, i+2)
        plt.imshow(cv2.cvtColor(db_imgs[match_idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Top {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

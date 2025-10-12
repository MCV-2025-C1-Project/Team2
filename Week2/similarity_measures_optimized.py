import numpy as np 
from scipy.stats import wasserstein_distance


import numpy as np

# Euclidean distance
def euclidean_distance(h1, h2):
    return np.sqrt(np.sum((h1 - h2) ** 2)) #lower-better

# L1 distance
def l1_distance(h1, h2):
    return np.sum(np.abs(h1 - h2)) #lower-better

# x^2 distance
def x2_distance(h1, h2, eps=1e-10):
    return 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps)) #lower-better

# Histogram intersection similarity
def histogram_intersection(h1, h2):
    return np.sum(np.minimum(h1, h2)) #higher-better

# Hellinger kernel similarity
def hellinger_kernel(h1, h2):
    return np.sum(np.sqrt(h1 * h2)) #higher-better

# Normalize
def normalize_hist(h):
    return h.astype("float") / (h.sum() + 1e-10)

def measure_similarities(query_hist, museum_hist, distance_func, top_k):
    results = []
    for q_idx, q_hist in enumerate(query_hist):
        q_hist = normalize_hist(q_hist)

        distances = []
        for m_idx, m_hist in enumerate(museum_hist):
            m_hist = normalize_hist(m_hist)
            score = distance_func(q_hist, m_hist)
            distances.append((m_idx, score))

        # Sorting based on distance or similarity
        if distance_func in [histogram_intersection, hellinger_kernel]:
            # higher-better
            distances = sorted(distances, key=lambda x: -x[1])
        else:
            # lower-better
            distances = sorted(distances, key=lambda x: x[1])

        results.append(distances[:top_k])  # adds top K to results
    return results


def euclidean_distance_matrix(Q, M): 
    # ||q - m||^2 = ||q||^2 + ||m||^2 - 2 q.m 
    q_norms = np.sum(Q**2, axis=1, keepdims=True) # shape (nq, 1) 
    m_norms = np.sum(M**2, axis=1, keepdims=True).T # shape (1, nm) 
    return np.sqrt(q_norms + m_norms - 2 * np.dot(Q, M.T)) 
    
    
def l1_distance_matrix(Q, M): 
    return np.sum(np.abs(Q[:, None, :] - M[None, :, :]), axis=2) 
    

def x2_distance_matrix(Q, M, eps=1e-10): 
    num = (Q[:, None, :] - M[None, :, :]) ** 2 
    denom = Q[:, None, :] + M[None, :, :] + eps 
    return 0.5 * np.sum(num / denom, axis=2) 
    
    
def histogram_intersection_matrix(Q, M): 
    return np.sum(np.minimum(Q[:, None, :], M[None, :, :]), axis=2) 
    
    
def hellinger_kernel_matrix(Q, M): 
    return np.sum(np.sqrt(Q[:, None, :] * M[None, :, :]), axis=2) 


def cosine_similarity_matrix(Q, M, eps=1e-10):
    # q.m/∥q∥∥m∥
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + eps)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)
    return np.dot(Q_norm, M_norm.T)

def bhattacharyya_distance_matrix(Q, M, eps=1e-10):
    BC = np.dot(np.sqrt(Q), np.sqrt(M).T)  # Bhattacharyya coefficient
    return -np.log(BC + eps)  # smaller - more similar

def correlation_matrix(Q, M, eps=1e-10):
    Q_mean = Q - Q.mean(axis=1, keepdims=True)
    M_mean = M - M.mean(axis=1, keepdims=True)
    num = np.dot(Q_mean, M_mean.T)
    denom = np.sqrt(np.sum(Q_mean**2, axis=1, keepdims=True)) * \
            np.sqrt(np.sum(M_mean**2, axis=1, keepdims=True).T + eps)
    return num / (denom + eps)  # higher - better

def kl_divergence_matrix(Q, M, eps=1e-10):
    """Symmetrized KL divergence"""
    Q_ = Q[:, None, :] + eps
    M_ = M[None, :, :] + eps
    kl_qm = np.sum(Q_ * np.log(Q_ / M_), axis=2)
    kl_mq = np.sum(M_ * np.log(M_ / Q_), axis=2)
    return 0.5 * (kl_qm + kl_mq)  # smaller = more similar

def emd_matrix(Q, M):
    """Earth Mover's Distance (using Wasserstein distance)"""
    nq, nm, nbins = Q.shape[0], M.shape[0], Q.shape[1]
    D = np.zeros((nq, nm))
    bins = np.arange(nbins)
    for i in range(nq):
        for j in range(nm):
            D[i, j] = wasserstein_distance(bins, bins, u_weights=Q[i], v_weights=M[j])
    return D  # smaller = more similar
    
    
#Normalization 
def normalize_hist(h): 
    return h.astype("float64") / (h.sum() + 1e-10) 
    
     
def measure_similarities(query_hist, museum_hist, distance_func, top_k): 
    Q = np.array([normalize_hist(h) for h in query_hist]) 
    M = np.array([normalize_hist(h) for h in museum_hist]) 
    # Map function names to matrix functions 
    func_map = { 
        euclidean_distance: euclidean_distance_matrix, 
        l1_distance: l1_distance_matrix, 
        x2_distance: x2_distance_matrix, 
        histogram_intersection: histogram_intersection_matrix, 
        hellinger_kernel: hellinger_kernel_matrix,
        cosine_similarity: cosine_similarity_matrix,
        bhattacharyya_distance: bhattacharyya_distance_matrix,
        correlation: correlation_matrix,
        kl_divergence: kl_divergence_matrix,
        emd: emd_matrix,
        } 
        
    D = func_map[distance_func](Q, M) # (nq, nm) matrix 
    
    # Sorting depending on metric type 
    if distance_func in [histogram_intersection, hellinger_kernel, cosine_similarity, correlation]: 
        # higher - better - sort descending 
        indices = np.argsort(-D, axis=1)[:, :top_k] 
        scores = -np.sort(-D, axis=1)[:, :top_k] 
    else: 
        # lower - better - sort ascending 
        indices = np.argsort(D, axis=1)[:, :top_k] 
        scores = np.sort(D, axis=1)[:, :top_k] 
        
    # Return list of lists [(m_idx, score), ...] 
    results = [ 
        list(zip(idx_row, score_row)) for idx_row, 
        score_row in zip(indices, scores) 
        ] 
    
    return results

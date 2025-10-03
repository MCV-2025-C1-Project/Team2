import numpy as np 
from similarity_measures import *


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
        } 
        
    D = func_map[distance_func](Q, M) # (nq, nm) matrix 
    
    # Sorting depending on metric type 
    if distance_func in [histogram_intersection, hellinger_kernel, cosine_similarity]: 
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

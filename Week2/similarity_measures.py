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

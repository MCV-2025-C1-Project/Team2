"""
Image Retrieval System - Week 2
Content-Based Image Retrieval using optimized algorithms from Week 1

This module provides a simple interface to retrieve similar images from the BBDD database
using the best performing methods identified in Week 1.
"""

import os
import pickle
import numpy as np
import cv2
from PIL import Image
import copy
from tqdm import tqdm

from histograms import HSV_Concat_Histogram, CIELAB_Concat_Histogram, HLS_Concat_Histogram
from week2_histograms import Histogram2D, Histogram3D, BlockHistogram, SpatialPyramidHistogram

from similarity_measures_optimized import (
    euclidean_distance_matrix, l1_distance_matrix, x2_distance_matrix,
    histogram_intersection_matrix, hellinger_kernel_matrix, cosine_similarity_matrix,
    bhattacharyya_distance_matrix, correlation_matrix, kl_divergence_matrix,
    normalize_hist
)
from helper_functions_main import pil_to_cv2, create_histogram_with_bins
from mapk import mapk


def load_ground_truth(gt_path="Data/Week1/qsd1_w1/gt_corresps.pkl"):
    """Load ground truth correspondences"""
    if os.path.exists(gt_path):
        with open(gt_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Ground truth file not found: {gt_path}")
        return None


class ImageRetrieval:
    """
    Image Retrieval System using optimized methods from Week 1
    """

    def __init__(self, database_path="../Data/BBDD/", cache_path="cache/"):
        self.database_path = database_path
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)

        # Hardcoded best methods from Week 1
        self.method1_config = {
            "descriptors": ["CIELAB", "HLS"],  # dual descriptor
            "weights1": [0.0, 0.0, 0.5],
            "weights2": [1.0, 0.5, 0.0],
            "bins": 256,
            "similarity_indices": [1, 3, 8]
        }

        self.method2_config = {
            "descriptors": ["CIELAB", "HSV"],
            "weights1": [1.0, 0.5, 0.0],
            "weights2": [0.0, 0.0, 0.5],
            "bins": 256,
            "similarity_indices": [1, 3, 8]
        }

        self.method3_config = {  # 3D RGB spatial pyramid
            "descriptors": ["3D_RGB_PYRAMID"],
            "weights1": [1.0],
            "weights2": [0.0],
            "bins": 8,
            "similarity_indices": [1]
        }

        self.method4_config = {  # 2D HS histogram
            "descriptors": ["2D_HS"],
            "weights1": [1.0],
            "weights2": [0.0],
            "bins": 32,
            "similarity_indices": [1]
        }

        self.method5_config = {  # Block-based 3D RGB histogram
            "descriptors": ["3D_RGB_BLOCK"],
            "weights1": [1.0],
            "weights2": [0.0],
            "bins": 8,
            "similarity_indices": [1]
        }

        # Similarity functions
        self.similarity_functions = [
            euclidean_distance_matrix,  # 0
            l1_distance_matrix,         # 1
            x2_distance_matrix,         # 2
            histogram_intersection_matrix,  # 3
            hellinger_kernel_matrix,    # 4
            cosine_similarity_matrix,   # 5
            bhattacharyya_distance_matrix,  # 6
            correlation_matrix,         # 7
            kl_divergence_matrix        # 8
        ]

        # Initialize database
        self.database_images = []
        self.database_histograms = {}
        self._load_database()

    def _load_database(self):
        """Load database images and compute/load histograms"""
        print("Loading database...")
        self.database_images = sorted([f for f in os.listdir(self.database_path) if f.endswith('.jpg')])
        print(f"Found {len(self.database_images)} images in database")

        cache_file = os.path.join(self.cache_path, "database_histograms.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("image_list") == self.database_images:
                    self.database_histograms = cached["histograms"]
                    print("Loaded histograms from cache")
                    return
                else:
                    print("Cache outdated, recomputing...")
            except Exception as e:
                print(f"Cache load failed: {e}")

        self._compute_database_histograms()
        with open(cache_file, "wb") as f:
            pickle.dump({"image_list": self.database_images, "histograms": self.database_histograms}, f)
        print("Saved histograms to cache")

    def _compute_database_histograms(self):
        """Compute histograms for all database images"""
        print("Computing histograms for database images...")

        hsv_histograms, lab_histograms, hls_histograms = [], [], []
        pyramid_histograms, hs2d_histograms, block_histograms = [], [], []

        for img_name in tqdm(self.database_images, desc="Computing histograms"):
            img_path = os.path.join(self.database_path, img_name)
            img_pil = Image.open(img_path)
            img_cv = pil_to_cv2(img_pil)
            h, w = img_cv.shape[:2]

            # HSV
            hsv_hist = HSV_Concat_Histogram(h, w)
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_hist = np.bincount(img_hsv[:, :, 0].flatten(), minlength=256)
            s_hist = np.bincount(img_hsv[:, :, 1].flatten(), minlength=256)
            v_hist = np.bincount(img_hsv[:, :, 2].flatten(), minlength=256)
            hsv_hist.setHist(h_hist, s_hist, v_hist)
            hsv_hist.h_hist, hsv_hist.s_hist, hsv_hist.v_hist = h_hist, s_hist, v_hist
            hsv_hist.calculate_concat_hist()
            hsv_hist.normalize()
            hsv_histograms.append(copy.copy(hsv_hist))

            # CIELAB
            lab_hist = CIELAB_Concat_Histogram(h, w)
            img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l_hist = np.bincount(img_lab[:, :, 0].flatten(), minlength=256)
            a_hist = np.bincount(img_lab[:, :, 1].flatten(), minlength=256)
            b_hist = np.bincount(img_lab[:, :, 2].flatten(), minlength=256)
            lab_hist.setHist(l_hist, a_hist, b_hist)
            lab_hist.l_hist, lab_hist.a_hist, lab_hist.b_hist = l_hist, a_hist, b_hist
            lab_hist.calculate_concat_hist()
            lab_hist.normalize()
            lab_histograms.append(copy.copy(lab_hist))

            # HLS
            hls_hist = HLS_Concat_Histogram(h, w)
            img_hls = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
            h_hist = np.bincount(img_hls[:, :, 0].flatten(), minlength=256)
            l_hist = np.bincount(img_hls[:, :, 1].flatten(), minlength=256)
            s_hist = np.bincount(img_hls[:, :, 2].flatten(), minlength=256)
            hls_hist.setHist(h_hist, l_hist, s_hist)
            hls_hist.h_hist, hls_hist.l_hist, hls_hist.s_hist = h_hist, l_hist, s_hist
            hls_hist.calculate_concat_hist()
            hls_hist.normalize()
            hls_histograms.append(copy.copy(hls_hist))

            # Week 2 descriptors
            pyramid_histograms.append(SpatialPyramidHistogram((8, 8, 8), 3, "RGB").compute(img_cv))
            hs2d_histograms.append(Histogram2D((32, 32), "HSV").compute(img_cv))
            block_histograms.append(BlockHistogram((8, 8, 8), (2, 2), "RGB").compute(img_cv))

        self.database_histograms = {
            "HSV": hsv_histograms,
            "CIELAB": lab_histograms,
            "HLS": hls_histograms,
            "3D_RGB_PYRAMID": pyramid_histograms,
            "2D_HS": hs2d_histograms,
            "3D_RGB_BLOCK": block_histograms,
        }

    def _compute_query_histogram(self, image_path, descriptor_type):
        """Compute histogram for a query image"""
        img_pil = Image.open(image_path)
        img_cv = pil_to_cv2(img_pil)
        h, w = img_cv.shape[:2]

        if descriptor_type == "HSV":
            hist = HSV_Concat_Histogram(h, w)
            img_conv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_hist = np.bincount(img_conv[:, :, 0].flatten(), minlength=256)
            s_hist = np.bincount(img_conv[:, :, 1].flatten(), minlength=256)
            v_hist = np.bincount(img_conv[:, :, 2].flatten(), minlength=256)
            hist.setHist(h_hist, s_hist, v_hist)
            hist.h_hist, hist.s_hist, hist.v_hist = h_hist, s_hist, v_hist

        elif descriptor_type == "CIELAB":
            hist = CIELAB_Concat_Histogram(h, w)
            img_conv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l_hist = np.bincount(img_conv[:, :, 0].flatten(), minlength=256)
            a_hist = np.bincount(img_conv[:, :, 1].flatten(), minlength=256)
            b_hist = np.bincount(img_conv[:, :, 2].flatten(), minlength=256)
            hist.setHist(l_hist, a_hist, b_hist)
            hist.l_hist, hist.a_hist, hist.b_hist = l_hist, a_hist, b_hist

        elif descriptor_type == "HLS":
            hist = HLS_Concat_Histogram(h, w)
            img_conv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
            h_hist = np.bincount(img_conv[:, :, 0].flatten(), minlength=256)
            l_hist = np.bincount(img_conv[:, :, 1].flatten(), minlength=256)
            s_hist = np.bincount(img_conv[:, :, 2].flatten(), minlength=256)
            hist.setHist(h_hist, l_hist, s_hist)
            hist.h_hist, hist.l_hist, hist.s_hist = h_hist, l_hist, s_hist

        elif descriptor_type == "3D_RGB_PYRAMID":
            return SpatialPyramidHistogram((8, 8, 8), 3, "RGB").compute(img_cv)

        elif descriptor_type == "2D_HS":
            return Histogram2D((32, 32), "HSV").compute(img_cv)

        elif descriptor_type == "3D_RGB_BLOCK":
            return BlockHistogram((8, 8, 8), (2, 2), "RGB").compute(img_cv)

        hist.calculate_concat_hist()
        hist.normalize()
        return hist

    def _prepare_for_similarity(self, query_hist, desc_type, bins):
        """Convert query and database histograms to normalized arrays"""
        if isinstance(query_hist, np.ndarray):
            Query = np.array([normalize_hist(query_hist)])
            DB = np.array([normalize_hist(h) for h in self.database_histograms[desc_type]])
        else:
            query_bins = create_histogram_with_bins(query_hist, bins)
            db_hist = [create_histogram_with_bins(h, bins) for h in self.database_histograms[desc_type]]
            Query = np.array([normalize_hist(query_bins)])
            DB = np.array([normalize_hist(h) for h in db_hist])
        return Query, DB

    def _retrieve_with_method(self, query_image_path, method_config, k=10):
        descriptors = method_config["descriptors"]
        bins = method_config["bins"]
        sim_indices = method_config["similarity_indices"]

        if len(descriptors) == 2:
            desc1, desc2 = descriptors
            w1, w2 = np.array(method_config["weights1"]), np.array(method_config["weights2"])

            qh1 = self._compute_query_histogram(query_image_path, desc1)
            qh2 = self._compute_query_histogram(query_image_path, desc2)
            Q1, DB1 = self._prepare_for_similarity(qh1, desc1, bins)
            Q2, DB2 = self._prepare_for_similarity(qh2, desc2, bins)

            s1 = np.stack([self.similarity_functions[i](Q1, DB1) for i in sim_indices], axis=-1)
            s2 = np.stack([self.similarity_functions[i](Q2, DB2) for i in sim_indices], axis=-1)

            for i, si in enumerate(sim_indices):
                func = self.similarity_functions[si]
                if func.__name__ in ['histogram_intersection_matrix', 'hellinger_kernel_matrix',
                                     'cosine_similarity_matrix', 'correlation_matrix']:
                    s1[0, :, i] = -s1[0, :, i]
                    s2[0, :, i] = -s2[0, :, i]

            combined = (np.dot(s1[0], w1) + np.dot(s2[0], w2)) / 2.0
        else:
            desc = descriptors[0]
            w1 = np.array(method_config["weights1"])
            qh = self._compute_query_histogram(query_image_path, desc)
            Q, DB = self._prepare_for_similarity(qh, desc, bins)
            s = np.stack([self.similarity_functions[i](Q, DB) for i in sim_indices], axis=-1)
            for i, si in enumerate(sim_indices):
                func = self.similarity_functions[si]
                if func.__name__ in ['histogram_intersection_matrix', 'hellinger_kernel_matrix',
                                     'cosine_similarity_matrix', 'correlation_matrix']:
                    s[0, :, i] = -s[0, :, i]
            combined = np.dot(s[0], w1)

        top_k = np.argsort(combined)[:k]
        return [self.database_images[idx] for idx in top_k]

    def retrieve_similar_images(self, query_image_path, method="method1", k=10):
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Query image not found: {query_image_path}")
        config = getattr(self, f"{method}_config", None)
        if not config:
            raise ValueError(f"Unknown method: {method}")
        return self._retrieve_with_method(query_image_path, config, k)

    def evaluate_on_qsd1(self, query_path="Data/Week1/qsd1_w1/", gt_path="Data/Week1/qsd1_w1/gt_corresps.pkl", k_values=[1, 5]):
        print("Loading ground truth...")
        gt = load_ground_truth(gt_path)
        if gt is None or not os.path.exists(query_path):
            print("Missing ground truth or query folder.")
            return None

        query_imgs = sorted([f for f in os.listdir(query_path) if f.endswith(".jpg")])
        print(f"Found {len(query_imgs)} queries.")

        results = {m: {"predictions": [], "map_scores": {}} for m in
                   ["method1", "method2", "method3", "method4", "method5"]}

        for method in results:
            print(f"\n=== Evaluating {method.upper()} ===")
            preds = []
            for img in tqdm(query_imgs, desc=method):
                qpath = os.path.join(query_path, img)
                sims = self.retrieve_similar_images(qpath, method=method, k=max(k_values))
                ids = [int(os.path.splitext(s)[0].split("_")[-1]) for s in sims]
                preds.append(ids)
            results[method]["predictions"] = preds

        print("\n=== Calculating MAP Scores ===")
        for k in k_values:
            print(f"\nMAP@{k}:")
            for method in results:
                score = mapk(gt, results[method]["predictions"], k)
                results[method]["map_scores"][f"MAP@{k}"] = score
                print(f"  {method.upper()}: {score:.4f}")

        best_method = max(results, key=lambda m: results[m]["map_scores"]["MAP@5"])
        results["best_method"] = best_method
        print(f"\n Best performing method: {best_method}")
        return results


def main():
    retriever = ImageRetrieval()

    query_path = "../Data/Week1/qst1_w1/00001.jpg"
    if os.path.exists(query_path):
        print("\n=== RETRIEVAL EXAMPLES ===")
        for method in ["method1", "method2", "method3", "method4", "method5"]:
            print(f"\n--- {method.upper()} ---")
            res = retriever.retrieve_similar_images(query_path, method=method, k=5)
            for i, img in enumerate(res, 1):
                print(f"{i}. {img}")
    else:
        print(f"Query not found: {query_path}")

    print("\n=== EVALUATION ON QSD1_W1 DATASET ===")
    retriever.evaluate_on_qsd1()


if __name__ == "__main__":
    main()

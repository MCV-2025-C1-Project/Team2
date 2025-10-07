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
from similarity_measures_optimized import (
    euclidean_distance_matrix, l1_distance_matrix, x2_distance_matrix,
    histogram_intersection_matrix, hellinger_kernel_matrix, cosine_similarity_matrix,
    bhattacharyya_distance_matrix, correlation_matrix, kl_divergence_matrix,
    normalize_hist
)
from helper_functions_main import pil_to_cv2, create_histogram_with_bins
from mapk import mapk


def load_ground_truth(gt_path="../Data/Week1/qsd1_w1/gt_corresps.pkl"):
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
        """
        Initialize the Image Retrieval System
        
        Args:
            database_path (str): Path to the BBDD database folder
            cache_path (str): Path to cache folder for storing computed histograms
        """
        self.database_path = database_path
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        
        # Hardcoded best methods from Week 1
        self.method1_config = {
            "descriptors": ["CIELAB", "HLS"],  # CIELAB Concat + HLS Concat
            "weights1": [0.0, 0.0, 0.5],       # [L1, Hist Int, KL Divergence]
            "weights2": [1.0, 0.5, 0.0],      
            "bins": 256,
            "similarity_indices": [1, 3, 8]   
        }
        
        self.method2_config = {
            "descriptors": ["CIELAB", "HSV"],  # CIELAB Concat + HSV Concat
            "weights1": [1.0, 0.5, 0.0],      
            "weights2": [0.0, 0.0, 0.5],      
            "bins": 256,
            "similarity_indices": [1, 3, 8]   
        }

        
        # Similarity functions
        self.similarity_functions = [
            euclidean_distance_matrix,    # 0
            l1_distance_matrix,          # 1
            x2_distance_matrix,          # 2
            histogram_intersection_matrix, # 3
            hellinger_kernel_matrix,     # 4
            cosine_similarity_matrix,    # 5
            bhattacharyya_distance_matrix, # 6
            correlation_matrix,          # 7
            kl_divergence_matrix         # 8
        ]
        
        # Initialize database
        self.database_images = []
        self.database_histograms = {}
        self._load_database()
    
    def _load_database(self):
        """Load database images and compute/load histograms"""
        print("Loading database...")
        
        # Get all jpg images from database
        self.database_images = [f for f in os.listdir(self.database_path) if f.endswith('.jpg')]
        self.database_images.sort()
        
        print(f"Found {len(self.database_images)} images in database")
        
        # Try to load cached histograms
        cache_file = os.path.join(self.cache_path, "database_histograms.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                # Validate cache
                if cached_data.get('image_list') == self.database_images:
                    self.database_histograms = cached_data['histograms']
                    print("Loaded histograms from cache")
                    return
                else:
                    print("Cache outdated, recomputing histograms...")
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        # Compute histograms
        self._compute_database_histograms()
        
        # Save to cache
        cache_data = {
            'image_list': self.database_images,
            'histograms': self.database_histograms
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print("Saved histograms to cache")
    
    def _compute_database_histograms(self):
        """Compute histograms for all database images"""
        print("Computing histograms for database images...")
        
        hsv_histograms = []
        lab_histograms = []
        hls_histograms = []
        
        for img_name in tqdm(self.database_images, desc="Computing histograms"):
            img_path = os.path.join(self.database_path, img_name)
            
            # Load and convert image
            img_pil = Image.open(img_path)
            img_cv = pil_to_cv2(img_pil)
            height, width = img_cv.shape[:2]
            
            # HSV Histogram
            hsv_hist = HSV_Concat_Histogram(height, width)
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_hist = np.bincount(img_hsv[:,:,0].flatten(), minlength=256)
            s_hist = np.bincount(img_hsv[:,:,1].flatten(), minlength=256)
            v_hist = np.bincount(img_hsv[:,:,2].flatten(), minlength=256)
            hsv_hist.setHist(h_hist, s_hist, v_hist)
            # Set attributes for create_histogram_with_bins compatibility
            hsv_hist.h_hist = h_hist
            hsv_hist.s_hist = s_hist
            hsv_hist.v_hist = v_hist
            hsv_hist.calculate_concat_hist()
            hsv_hist.normalize()
            hsv_histograms.append(copy.copy(hsv_hist))
            
            # CIELAB Histogram
            lab_hist = CIELAB_Concat_Histogram(height, width)
            img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l_hist = np.bincount(img_lab[:,:,0].flatten(), minlength=256)
            a_hist = np.bincount(img_lab[:,:,1].flatten(), minlength=256)
            b_hist = np.bincount(img_lab[:,:,2].flatten(), minlength=256)
            lab_hist.setHist(l_hist, a_hist, b_hist)
            # Set attributes for create_histogram_with_bins compatibility
            lab_hist.l_hist = l_hist
            lab_hist.a_hist = a_hist
            lab_hist.b_hist = b_hist
            lab_hist.calculate_concat_hist()
            lab_hist.normalize()
            lab_histograms.append(copy.copy(lab_hist))
            
            # HLS Histogram
            hls_hist = HLS_Concat_Histogram(height, width)
            img_hls = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
            h_hist = np.bincount(img_hls[:,:,0].flatten(), minlength=256)
            l_hist = np.bincount(img_hls[:,:,1].flatten(), minlength=256)
            s_hist = np.bincount(img_hls[:,:,2].flatten(), minlength=256)
            hls_hist.setHist(h_hist, l_hist, s_hist)
            # Set attributes for create_histogram_with_bins compatibility
            hls_hist.h_hist = h_hist
            hls_hist.l_hist = l_hist
            hls_hist.s_hist = s_hist
            hls_hist.calculate_concat_hist()
            hls_hist.normalize()
            hls_histograms.append(copy.copy(hls_hist))

        
        self.database_histograms = {
            'HSV': hsv_histograms,
            'CIELAB': lab_histograms,
            'HLS': hls_histograms
        }
    
    def _compute_query_histogram(self, image_path, descriptor_type):
        """Compute histogram for a query image"""
        # Load and convert image
        img_pil = Image.open(image_path)
        img_cv = pil_to_cv2(img_pil)
        height, width = img_cv.shape[:2]
        
        if descriptor_type == "HSV":
            hist_obj = HSV_Concat_Histogram(height, width)
            img_converted = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_hist = np.bincount(img_converted[:,:,0].flatten(), minlength=256)
            s_hist = np.bincount(img_converted[:,:,1].flatten(), minlength=256)
            v_hist = np.bincount(img_converted[:,:,2].flatten(), minlength=256)
            hist_obj.setHist(h_hist, s_hist, v_hist)
            # Set attributes for create_histogram_with_bins compatibility
            hist_obj.h_hist = h_hist
            hist_obj.s_hist = s_hist
            hist_obj.v_hist = v_hist
            
        elif descriptor_type == "CIELAB":
            hist_obj = CIELAB_Concat_Histogram(height, width)
            img_converted = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l_hist = np.bincount(img_converted[:,:,0].flatten(), minlength=256)
            a_hist = np.bincount(img_converted[:,:,1].flatten(), minlength=256)
            b_hist = np.bincount(img_converted[:,:,2].flatten(), minlength=256)
            hist_obj.setHist(l_hist, a_hist, b_hist)
            # Set attributes for create_histogram_with_bins compatibility
            hist_obj.l_hist = l_hist
            hist_obj.a_hist = a_hist
            hist_obj.b_hist = b_hist
            
        elif descriptor_type == "HLS":
            hist_obj = HLS_Concat_Histogram(height, width)
            img_converted = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
            h_hist = np.bincount(img_converted[:,:,0].flatten(), minlength=256)
            l_hist = np.bincount(img_converted[:,:,1].flatten(), minlength=256)
            s_hist = np.bincount(img_converted[:,:,2].flatten(), minlength=256)
            hist_obj.setHist(h_hist, l_hist, s_hist)
            # Set attributes for create_histogram_with_bins compatibility
            hist_obj.h_hist = h_hist
            hist_obj.l_hist = l_hist
            hist_obj.s_hist = s_hist
        
        hist_obj.calculate_concat_hist()
        hist_obj.normalize()
        return hist_obj
    
    def _retrieve_with_method(self, query_image_path, method_config, k=10):
        """Retrieve similar images using a specific method"""
        desc1_type, desc2_type = method_config["descriptors"]
        weights1 = np.array(method_config["weights1"])
        weights2 = np.array(method_config["weights2"])
        bins = method_config["bins"]
        sim_indices = method_config["similarity_indices"]
        
        # Compute query histograms
        query_hist1 = self._compute_query_histogram(query_image_path, desc1_type)
        query_hist2 = self._compute_query_histogram(query_image_path, desc2_type)
        
        # Apply bin reduction if needed
        query_bins1 = create_histogram_with_bins(query_hist1, bins)
        query_bins2 = create_histogram_with_bins(query_hist2, bins)
        
        # Get database histograms and apply bin reduction
        db_hist1 = [create_histogram_with_bins(h, bins) for h in self.database_histograms[desc1_type]]
        db_hist2 = [create_histogram_with_bins(h, bins) for h in self.database_histograms[desc2_type]]
        
        # Normalize
        Query1 = np.array([normalize_hist(query_bins1)])
        Query2 = np.array([normalize_hist(query_bins2)])
        DB1 = np.array([normalize_hist(h) for h in db_hist1])
        DB2 = np.array([normalize_hist(h) for h in db_hist2])
        
        # Compute similarity scores
        scores1 = np.stack([self.similarity_functions[i](Query1, DB1) for i in sim_indices], axis=-1)
        scores2 = np.stack([self.similarity_functions[i](Query2, DB2) for i in sim_indices], axis=-1)
        
        # Apply sign correction for similarity measures (intersection, hellinger, cosine, correlation)
        for i, sim_idx in enumerate(sim_indices):
            func = self.similarity_functions[sim_idx]
            if func.__name__ in ['histogram_intersection_matrix', 'hellinger_kernel_matrix', 
                                'cosine_similarity_matrix', 'correlation_matrix']:
                scores1[0, :, i] = -scores1[0, :, i]
                scores2[0, :, i] = -scores2[0, :, i]
        
        # Apply weights and combine
        weighted1 = np.dot(scores1[0], weights1)
        weighted2 = np.dot(scores2[0], weights2)
        combined = (weighted1 + weighted2) / 2.0
        
        # Get top k indices
        top_k_indices = np.argsort(combined)[:k]
        
        # Return image filenames (no scores needed)
        results = []
        for idx in top_k_indices:
            results.append(self.database_images[idx])
        
        return results
    
    def retrieve_similar_images(self, query_image_path, method="method1", k=10):
        """
        Retrieve top-k similar images from the database
        
        Args:
            query_image_path (str): Path to the query image
            method (str): Method to use ("method1" or "method2")
            k (int): Number of similar images to retrieve (default: 10)
            
        Returns:
            list: List of image filenames, ordered by similarity (most similar first)
        """
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Query image not found: {query_image_path}")
        
        if method == "method1":
            config = self.method1_config
        elif method == "method2":
            config = self.method2_config
        else:
            raise ValueError("Method must be 'method1' or 'method2'")
                
        results = self._retrieve_with_method(query_image_path, config, k)
        
        return results
    
    def get_method_info(self, method="method1"):
        """Get information about a specific method"""
        if method == "method1":
            config = self.method1_config
        elif method == "method2":
            config = self.method2_config
        else:
            raise ValueError("Method must be 'method1' or 'method2'")
        
        return {
            "method": method,
            "descriptors": config["descriptors"],
            "weights1": config["weights1"],
            "weights2": config["weights2"],
            "bins": config["bins"],
            "similarity_measures": ["L1", "Histogram Intersection", "KL Divergence"]  # Based on indices used
        }

    def evaluate_on_qsd1(self, query_path="../Data/Week1/qsd1_w1/", gt_path="../Data/Week1/qsd1_w1/gt_corresps.pkl", k_values=[1, 5]):
        """
        Evaluate both methods on qsd1_w1 dataset and calculate MAP scores
        
        Args:
            query_path (str): Path to query images folder
            gt_path (str): Path to ground truth file
            k_values (list): List of K values for MAP@K calculation
            
        Returns:
            dict: Results with MAP scores for both methods
        """
        print("Loading ground truth...")
        ground_truth = load_ground_truth(gt_path)
        if ground_truth is None:
            return None
        
        # Get query images
        if not os.path.exists(query_path):
            print(f"Query path not found: {query_path}")
            return None
            
        query_images = [f for f in os.listdir(query_path) if f.endswith('.jpg')]
        query_images.sort()
        
        print(f"Found {len(query_images)} query images")
        print(f"Ground truth has {len(ground_truth)} entries")
        
        results = {
            "method1": {"predictions": [], "map_scores": {}},
            "method2": {"predictions": [], "map_scores": {}}
        }
        
        # Evaluate Method 1
        print("\n=== Evaluating Method 1 (CIELAB + HLS) ===")
        method1_predictions = []
        for img_name in tqdm(query_images, desc="Method 1"):
            img_path = os.path.join(query_path, img_name)
            similar_images = self.retrieve_similar_images(img_path, method="method1", k=max(k_values))
            # Convert to image IDs (remove extension and convert to int)
            image_ids = [int(os.path.splitext(img)[0].split('_')[-1]) for img in similar_images]
            method1_predictions.append(image_ids)
        
        results["method1"]["predictions"] = method1_predictions
        
        # Evaluate Method 2
        print("\n=== Evaluating Method 2 (CIELAB + HSV) ===")
        method2_predictions = []
        for img_name in tqdm(query_images, desc="Method 2"):
            img_path = os.path.join(query_path, img_name)
            similar_images = self.retrieve_similar_images(img_path, method="method2", k=max(k_values))
            # Convert to image IDs (remove extension and convert to int)
            image_ids = [int(os.path.splitext(img)[0].split('_')[-1]) for img in similar_images]
            method2_predictions.append(image_ids)
        
        results["method2"]["predictions"] = method2_predictions
        
        # Calculate MAP scores
        print("\n=== Calculating MAP Scores ===")
        for k in k_values:
            # Method 1
            map1_k = mapk(ground_truth, method1_predictions, k)
            results["method1"]["map_scores"][f"MAP@{k}"] = map1_k
            
            # Method 2
            map2_k = mapk(ground_truth, method2_predictions, k)
            results["method2"]["map_scores"][f"MAP@{k}"] = map2_k
            
            print(f"MAP@{k}:")
            print(f"  Method 1 (CIELAB + HLS): {map1_k:.4f}")
            print(f"  Method 2 (CIELAB + HSV): {map2_k:.4f}")
        
        # Determine best method
        best_method = "method1" if results["method1"]["map_scores"]["MAP@5"] > results["method2"]["map_scores"]["MAP@5"] else "method2"
        results["best_method"] = best_method
        
        print(f"\nBest performing method: {best_method}")
        
        return results


def main():
    """Example usage of the Image Retrieval System"""
    # Initialize the retrieval system
    retriever = ImageRetrieval()
    
    # Example query (you would replace this with actual query image path)
    query_path = "../Data/Week1/qst1_w1/00001.jpg"  # Example path
    
    if os.path.exists(query_path):
        # Retrieve using method1
        print("=== METHOD 1 ===")
        results1 = retriever.retrieve_similar_images(query_path, method="method1", k=10)
        
        print("Top 10 similar images (Method 1):")
        for i, image_name in enumerate(results1, 1):
            print(f"{i}. {image_name}")
        
        print("\n=== METHOD 2 ===")
        results2 = retriever.retrieve_similar_images(query_path, method="method2", k=10)
        
        print("Top 10 similar images (Method 2):")
        for i, image_name in enumerate(results2, 1):
            print(f"{i}. {image_name}")
        
        # Show method information
        print("\n=== METHOD INFO ===")
        print("Method 1:", retriever.get_method_info("method1"))
        print("Method 2:", retriever.get_method_info("method2"))
    
    else:
        print(f"Query image not found: {query_path}")
        print("Please provide a valid query image path")

    # Evaluate on qsd1_w1 dataset
    print("\n=== EVALUATION ON QSD1_W1 DATASET ===")
    evaluation_results = retriever.evaluate_on_qsd1()


if __name__ == "__main__":
    main()
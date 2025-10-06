import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def pil_to_cv2(img):
    """Convert PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def extract_histograms(hist_objects, attr="normalized"):
    """Extract histogram arrays from histogram objects."""
    return [np.array(getattr(h, attr)) for h in hist_objects]


def create_histogram_with_bins(hist_obj, new_bins=256):
    """Create histogram with specified number of bins by reducing from 256 bins."""
    if new_bins == 256:
        return hist_obj.normalized.copy()
    
    original_hist = hist_obj.hist if hasattr(hist_obj, 'hist') else hist_obj.normalized
    
    if hasattr(hist_obj, 'red_hist'):
        # Multi-channel histogram (RGB, HSV, LAB, etc.)
        channels = []
        if hasattr(hist_obj, 'red_hist'):
            channels.extend([hist_obj.red_hist, hist_obj.green_hist, hist_obj.blue_hist])
        elif hasattr(hist_obj, 'h_hist'):
            if hasattr(hist_obj, 's_hist'):
                channels.extend([hist_obj.h_hist, hist_obj.s_hist, hist_obj.v_hist])
            else:
                channels.extend([hist_obj.h_hist, hist_obj.l_hist, hist_obj.s_hist])
        elif hasattr(hist_obj, 'l_hist'):
            channels.extend([hist_obj.l_hist, hist_obj.a_hist, hist_obj.b_hist])
        elif hasattr(hist_obj, 'y_hist'):
            if hasattr(hist_obj, 'cb_hist'):
                channels.extend([hist_obj.y_hist, hist_obj.cb_hist, hist_obj.cr_hist])
            elif hasattr(hist_obj, 'u_hist'):
                channels.extend([hist_obj.y_hist, hist_obj.u_hist, hist_obj.v_hist])
        elif hasattr(hist_obj, 'x_hist'):
            channels.extend([hist_obj.x_hist, hist_obj.y_hist, hist_obj.z_hist])
        
        reduced_channels = []
        for channel in channels:
            if new_bins >= 256:
                reduced_channels.append(channel)
            else:
                bin_ratio = 256 // new_bins
                reduced = np.add.reduceat(channel, np.arange(0, 256, bin_ratio))[:new_bins]
                reduced_channels.append(reduced)
        result = np.concatenate(reduced_channels)
    else:
        # Single-channel histogram (grayscale)
        if new_bins >= 256:
            result = original_hist.copy()
        else:
            bin_ratio = 256 // new_bins
            result = np.add.reduceat(original_hist, np.arange(0, 256, bin_ratio))[:new_bins]
    
    # Normalize
    total = np.sum(result)
    if total > 0:
        result = result / total
    return result


def visualize_query_results(test_images, test_directory, train_directory, test_results, query_idx=0, top_n=5):
    """
    Visualize a query image and the top retrieved images from both algorithms
    
    Args:
        query_idx: Index of the query image to visualize (0-based)
        top_n: Number of top retrieved images to show for each algorithm
    """
    if query_idx >= len(test_images):
        print(f"Query index {query_idx} out of range. Available queries: 0-{len(test_images)-1}")
        return
    
    query_name = test_images[query_idx]
    print(f"Query {query_idx}: {query_name}")
    
    # Load query image
    query_img = Image.open(os.path.join(test_directory, query_name))
    
    # Get results from both algorithms
    alg1_results = test_results[0][query_idx][:top_n] if len(test_results) > 0 else []
    alg2_results = test_results[1][query_idx][:top_n] if len(test_results) > 1 else []
    
    # Create figure
    fig_width = max(top_n + 1, 6)
    fig, axes = plt.subplots(3, fig_width, figsize=(fig_width * 2, 6))
    
    # Handle case where we have fewer than 3 rows or single algorithm
    if len(test_results) == 1:
        fig, axes = plt.subplots(2, fig_width, figsize=(fig_width * 2, 4))
        axes = np.array(axes).reshape(2, -1)
    else:
        axes = np.array(axes).reshape(3, -1)
    
    # Show query image in first column of first row
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f"Query {query_idx}\\n{query_name}", fontsize=10)
    axes[0, 0].axis('off')
    
    # Hide remaining cells in first row
    for j in range(1, fig_width):
        axes[0, j].axis('off')
    
    # Show Algorithm 1 results
    if len(test_results) > 0:
        for i, img_id in enumerate(alg1_results):
            if i < fig_width:
                train_img_name = f"bbdd_{img_id:05d}.jpg"
                train_img_path = os.path.join(train_directory, train_img_name)
                
                if os.path.exists(train_img_path):
                    train_img = Image.open(train_img_path)
                    axes[1, i].imshow(train_img)
                    axes[1, i].set_title(f"ID: {img_id}\\n{train_img_name}", fontsize=8)
                else:
                    axes[1, i].text(0.5, 0.5, f"ID: {img_id}\\nNot found", 
                                  ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')
        
        # Hide remaining cells in Algorithm 1 row
        for i in range(len(alg1_results), fig_width):
            axes[1, i].axis('off')
    
    # Show Algorithm 2 results (if available)
    if len(test_results) > 1:
        for i, img_id in enumerate(alg2_results):
            if i < fig_width:
                train_img_name = f"bbdd_{img_id:05d}.jpg"
                train_img_path = os.path.join(train_directory, train_img_name)
                
                if os.path.exists(train_img_path):
                    train_img = Image.open(train_img_path)
                    axes[2, i].imshow(train_img)
                    axes[2, i].set_title(f"ID: {img_id}\\n{train_img_name}", fontsize=8)
                else:
                    axes[2, i].text(0.5, 0.5, f"ID: {img_id}\\nNot found", 
                                  ha='center', va='center', transform=axes[2, i].transAxes)
                axes[2, i].axis('off')
        
        # Hide remaining cells in Algorithm 2 row
        for i in range(len(alg2_results), fig_width):
            axes[2, i].axis('off')
    
    # Add row labels
    if len(test_results) > 0:
        alg1_label = f"Algorithm 1"
        fig.text(0.0, 0.45, alg1_label, fontsize=10, rotation=90, va='center')
    
    if len(test_results) > 1:
        alg2_label = f"Algorithm 2"
        fig.text(0.0, 0.15, alg2_label, fontsize=10, rotation=90, va='center')
    
    plt.suptitle(f"Image Retrieval Results for Query {query_idx}", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print(f"Algorithm 1 - Top {top_n} results: {alg1_results}")
    if len(test_results) > 1:
        print(f"Algorithm 2 - Top {top_n} results: {alg2_results}")

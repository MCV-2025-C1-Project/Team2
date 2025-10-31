import numpy as np
import os
import cv2
from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes, gaussian_filter
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings('ignore')

import background_remover_w2 as background_remover

"""from src.data.extract import read_image
from src.metrics.mask_metrics import (
    compute_confusion_counts_mask,
    precision_mask,
    recall_mask,
    f1_mask
)
from src.visualization.viz import (
    apply_mask_to_image_mask,
    create_visualization_mask,
    visualize_color_pipeline_steps_mask,
    create_summary_plots_mask,
    save_results_mask
)
from src.descriptors.grayscale import convert_img_to_gray_scale
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.hsv import convert_img_to_hsv"""


def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0, radius: int = 2) -> np.ndarray:
    """
    Apply Gaussian filter to denoise image.

    Args:
        image (np.ndarray): Input image (H, W, C).
        sigma (float): Standard deviation for Gaussian kernel.
        radius (int): Radius of Gaussian kernel (truncate parameter).

    Returns:
        np.ndarray: Filtered image.
    """
    # Apply Gaussian filter to each channel
    filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        filtered[:, :, c] = gaussian_filter(
            image[:, :, c].astype(np.float32),
            sigma=sigma,
            truncate=radius
        )

    return filtered.astype(image.dtype)


def _estimate_bg_from_borders(image_lab: np.ndarray, border: int = 20) -> Dict[str, np.ndarray]:
    """
    Estimate background color statistics from the image borders in LAB space.

    Args:
        image_lab (np.ndarray): Image in LAB color space.
        border (int): Width of border pixels to sample for background estimation.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'median' and 'mad' statistics for each channel.
    """
    H, W, _ = image_lab.shape
    b = max(1, min(border, min(H, W)//4))

    top    = image_lab[:b, :, :]
    bottom = image_lab[-b:, :, :]
    left   = image_lab[:, :b, :]
    right  = image_lab[:, -b:, :]

    borders = np.concatenate(
        [top.reshape(-1,3), bottom.reshape(-1,3), left.reshape(-1,3), right.reshape(-1,3)],
        axis=0
    )

    med = np.median(borders, axis=0)                            # (3,)
    mad = np.median(np.abs(borders - med), axis=0) + 1e-6       # (3,) avoid /0
    return {"median": med.astype(np.float32), "mad": mad.astype(np.float32)}

def _lab_robust_distance_weighted(L: np.ndarray, A: np.ndarray, B: np.ndarray,
                                  med: np.ndarray, mad: np.ndarray,
                                  wL: float = 0.6, wA: float = 1.0, wB: float = 1.0) -> np.ndarray:
    """
    Weighted distance to BG center in LAB, normalized by MAD per channel.

    Args:
        L (np.ndarray): Lightness channel.
        A (np.ndarray): A channel (green-red).
        B (np.ndarray): B channel (blue-yellow).
        med (np.ndarray): Median values for each LAB channel.
        mad (np.ndarray): MAD (Median Absolute Deviation) values for each LAB channel.
        wL (float): Weight for L channel to downweight illumination.
        wA (float): Weight for A channel.
        wB (float): Weight for B channel.

    Returns:
        np.ndarray: Weighted distance to background center.
    """
    dL = (L - med[0]) / mad[0]
    dA = (A - med[1]) / mad[1]
    dB = (B - med[2]) / mad[2]
    dist = np.sqrt((wL*dL)**2 + (wA*dA)**2 + (wB*dB)**2)
    return dist

def _hue_circular_distance(h: np.ndarray, h0: float) -> np.ndarray:
    """
    Circular distance on hue (OpenCV H in [0,180)).

    Args:
        h (np.ndarray): Hue values in range [0,180).
        h0 (float): Reference hue value.

    Returns:
        np.ndarray: Circular distance in degrees in range [0,90].
    """
    dh = np.abs(h - h0)
    dh = np.minimum(dh, 180.0 - dh)
    return dh


def _morphological_area_opening(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Area opening morphology to remove small blobs.

    Args:
        mask (np.ndarray): Binary mask to process.
        min_area (int): Minimum area threshold for blob removal.

    Returns:
        np.ndarray: Processed mask with small blobs removed.
    """
    if min_area <= 0:
        return mask

    # disk area ~ pi r^2  -> r ~ sqrt(min_area/pi)
    r = max(1, int(np.sqrt(float(min_area) / np.pi)))
    se = np.ones((2*r+1, 2*r+1), dtype=bool)

    # Remove small bright objects
    mask = binary_opening(mask, structure=se)

    # Fill small holes by the same area scale on the inverse
    inv = ~mask
    inv = binary_opening(inv, structure=se)
    mask = ~inv

    return mask

def _post_clean(mask: np.ndarray,
                open_size: int = 5,
                close_size: int = 7,
                min_area: int = 200) -> np.ndarray:
    """
    Post-process mask with morphological operations and area filtering.

    Args:
        mask (np.ndarray): Binary mask to clean.
        open_size (int): Size of opening structuring element.
        close_size (int): Size of closing structuring element.
        min_area (int): Minimum area threshold for component retention.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    if open_size > 0:
        mask = binary_opening(mask, structure=np.ones((open_size, open_size)))
    if close_size > 0:
        mask = binary_closing(mask, structure=np.ones((close_size, close_size)))

    mask = binary_fill_holes(mask)

    # Area opening for large component
    mask = _morphological_area_opening(mask, min_area=min_area)

    return mask


def _project_mask_to_signal(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Project a 2D mask to a 1D signal by summing along the specified axis.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        axis (int): Axis to sum along. 0 for column-wise (horizontal split), 1 for row-wise (vertical split).

    Returns:
        np.ndarray: 1D signal representing the projection.
    """
    # Sum along the axis and normalize to binary (0 or 1)
    signal = np.sum(mask, axis=axis)
    # Convert to binary: 0 if all zeros, 1 if at least some foreground
    signal = (signal > 0).astype(np.int32)
    return signal


def _project_mask_to_signal_thick(mask: np.ndarray, row_thickness: int = 50) -> np.ndarray:
    """
    Project mask to 1D signal using thick row detection (improved method).

    For each column, if ANY pixel in that column has foreground, signal=1, else signal=0.
    This provides more robust gap detection than single-pixel rows.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        row_thickness (int): Row thickness parameter (informational, actual check is per-column).

    Returns:
        np.ndarray: 1D signal array of length W.
    """
    H, W = mask.shape
    signal = []

    # Process column by column
    for col in range(W):
        # Check if any pixel in this column (across all rows) has foreground
        if np.any(mask[:, col] > 0):
            signal.append(1)
        else:
            signal.append(0)

    return np.array(signal)


def _detect_gap_pattern(signal: np.ndarray, min_gap_size: int = 10) -> tuple:
    """
    Detect if there are multiple paintings by analyzing the 1D signal pattern.

    Looks for patterns like:
    - Single painting: 0-1-0 or 0-0-0 or 1-0 or 0-1
    - Two paintings: 0-1-0-1-0 or 1-0-1 or 1-0-1-0 or 0-1-0-1

    Args:
        signal (np.ndarray): 1D binary signal.
        min_gap_size (int): Minimum size of gap (in pixels) to consider as separation.

    Returns:
        tuple: (num_paintings, gap_start, gap_end) where gap_start and gap_end define the gap position.
               Returns (1, -1, -1) if single painting detected.
    """
    # Find transitions from 0 to 1 (painting starts) and 1 to 0 (painting ends)
    diff = np.diff(np.concatenate([[0], signal, [0]]))
    starts = np.where(diff == 1)[0]  # Indices where painting starts
    ends = np.where(diff == -1)[0]   # Indices where painting ends

    num_regions = len(starts)

    if num_regions <= 1:
        # Single painting or no painting
        return (1, -1, -1)

    # Find gaps between regions
    gaps = []
    for i in range(num_regions - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start
        if gap_size >= min_gap_size:
            gaps.append((gap_start, gap_end, gap_size))

    if len(gaps) == 0:
        # No significant gap found
        return (1, -1, -1)

    # Find the largest gap (most likely the gap between two paintings)
    largest_gap = max(gaps, key=lambda x: x[2])
    gap_start, gap_end, _ = largest_gap

    return (2, gap_start, gap_end)


def _detect_vertical_gap(signal: np.ndarray, min_gap_size: int = 50) -> Tuple[int, int, int]:
    """
    Detect vertical gap (background region) in signal for vertical-only splitting.

    This is an improved version focused on detecting side-by-side paintings only.

    Args:
        signal (np.ndarray): 1D binary signal.
        min_gap_size (int): Minimum gap size to consider as split (pixels).

    Returns:
        Tuple[int, int, int]: (num_regions, gap_start, gap_end)
            - num_regions: 1 for single painting, 2 for two paintings
            - gap_start, gap_end: Gap boundaries (-1, -1 if single painting)
    """
    # Find transitions from 0 to 1 (region starts) and 1 to 0 (region ends)
    diff = np.diff(np.concatenate([[0], signal, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    num_regions = len(starts)

    if num_regions <= 1:
        # Single painting or no painting
        return (1, -1, -1)

    # Find gaps between regions
    gaps = []
    for i in range(num_regions - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start
        if gap_size >= min_gap_size:
            gaps.append((gap_start, gap_end, gap_size))

    if len(gaps) == 0:
        # No significant gap found
        return (1, -1, -1)

    # Return largest gap (most likely separation between paintings)
    largest_gap = max(gaps, key=lambda x: x[2])
    return (2, largest_gap[0], largest_gap[1])


def _find_split_position(gap_start: int, gap_end: int) -> int:
    """
    Find the split position at the middle of the gap.

    Args:
        gap_start (int): Start index of the gap.
        gap_end (int): End index of the gap.

    Returns:
        int: Split position (middle of the gap).
    """
    return (gap_start + gap_end) // 2


def crop_to_mask_bounds(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image and mask to the bounding box of the foreground.

    This function finds the tight bounding box around the foreground pixels
    and crops both the image and mask to this region.

    Args:
        image (np.ndarray): Image to crop (H, W, C).
        mask (np.ndarray): Binary mask (H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped (image, mask) tuple.
    """
    # Find bounding box of foreground
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No foreground, return original
        return image, mask

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped_image = image[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]

    return cropped_image, cropped_mask


def visualize_signal_analysis(image: np.ndarray,
                              initial_mask: np.ndarray,
                              min_gap_size: int = 10,
                              save_path: str = None) -> plt.Figure:
    """
    Visualize the signal analysis for detecting multiple paintings.

    Args:
        image (np.ndarray): Original image in BGR format.
        initial_mask (np.ndarray): Initial segmentation mask.
        min_gap_size (int): Minimum gap size for detection.
        save_path (str): Path to save the figure (optional).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    # Project to signals
    horizontal_signal = _project_mask_to_signal(initial_mask, axis=0)
    vertical_signal = _project_mask_to_signal(initial_mask, axis=1)

    # Detect gaps
    num_h, gap_h_start, gap_h_end = _detect_gap_pattern(horizontal_signal, min_gap_size)
    num_v, gap_v_start, gap_v_end = _detect_gap_pattern(vertical_signal, min_gap_size)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Initial mask
    axes[0, 1].imshow(initial_mask, cmap='gray')
    axes[0, 1].set_title("Initial Segmentation Mask")
    axes[0, 1].axis('off')

    # Horizontal signal (for detecting side-by-side paintings)
    axes[1, 0].plot(horizontal_signal, linewidth=2, color='blue')
    axes[1, 0].set_title(f"Horizontal Signal - Paintings: {num_h}")
    axes[1, 0].set_xlabel("Column Index")
    axes[1, 0].set_ylabel("Signal Value (0 or 1)")
    axes[1, 0].set_ylim([-0.1, 1.5])
    axes[1, 0].grid(True, alpha=0.3)
    if num_h == 2:
        axes[1, 0].axvline(gap_h_start, color='r', linestyle='--', label='Gap Start', linewidth=2)
        axes[1, 0].axvline(gap_h_end, color='g', linestyle='--', label='Gap End', linewidth=2)
        split_pos = _find_split_position(gap_h_start, gap_h_end)
        axes[1, 0].axvline(split_pos, color='purple', linestyle='-', linewidth=3, label=f'Split @ {split_pos}')
        axes[1, 0].legend()

    # Vertical signal (for detecting stacked paintings)
    axes[1, 1].plot(vertical_signal, linewidth=2, color='green')
    axes[1, 1].set_title(f"Vertical Signal - Paintings: {num_v}")
    axes[1, 1].set_xlabel("Row Index")
    axes[1, 1].set_ylabel("Signal Value (0 or 1)")
    axes[1, 1].set_ylim([-0.1, 1.5])
    axes[1, 1].grid(True, alpha=0.3)
    if num_v == 2:
        axes[1, 1].axvline(gap_v_start, color='r', linestyle='--', label='Gap Start', linewidth=2)
        axes[1, 1].axvline(gap_v_end, color='g', linestyle='--', label='Gap End', linewidth=2)
        split_pos = _find_split_position(gap_v_start, gap_v_end)
        axes[1, 1].axvline(split_pos, color='purple', linestyle='-', linewidth=3, label=f'Split @ {split_pos}')
        axes[1, 1].legend()

    plt.suptitle("Signal Analysis for Multi-Painting Detection", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def visualize_split_comparison(image: np.ndarray,
                                initial_mask: np.ndarray,
                                final_mask: np.ndarray,
                                split_info: dict,
                                save_path: str = None) -> plt.Figure:
    """
    Visualize comparison between initial segmentation and split-then-segment result.

    Args:
        image (np.ndarray): Original image in BGR format.
        initial_mask (np.ndarray): Segmentation before splitting.
        final_mask (np.ndarray): Segmentation after split-segment-merge.
        split_info (dict): Information about the split (type, position, etc.).
        save_path (str): Path to save the figure (optional).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create segmented images
    initial_segmented = image_rgb.copy()
    initial_segmented[initial_mask == 0] = 0

    final_segmented = image_rgb.copy()
    final_segmented[final_mask == 0] = 0

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Initial (before split)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(initial_mask, cmap='gray')
    axes[0, 1].set_title("Initial Mask (Before Split)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(initial_segmented)
    axes[0, 2].set_title("Initial Segmentation", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Final (after split-segment-merge)
    axes[1, 0].imshow(image_rgb)

    # Draw split line on the image
    if split_info['type'] == 'horizontal':
        split_pos = split_info['position']
        axes[1, 0].axvline(split_pos, color='red', linestyle='--', linewidth=3, label=f'Split at col {split_pos}')
        axes[1, 0].legend(loc='upper right')
    elif split_info['type'] == 'vertical':
        split_pos = split_info['position']
        axes[1, 0].axhline(split_pos, color='red', linestyle='--', linewidth=3, label=f'Split at row {split_pos}')
        axes[1, 0].legend(loc='upper right')

    split_type_str = split_info['type'].capitalize() if split_info['type'] != 'none' else 'No Split'
    axes[1, 0].set_title(f"Split Strategy: {split_type_str}", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title("Final Mask (After Split-Segment-Merge)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(final_segmented)
    axes[1, 2].set_title("Final Segmentation", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle("Comparison: Before Split vs After Split-Segment-Merge", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def detect_and_split_paintings(image: np.ndarray,
                                mask: np.ndarray,
                                row_thickness: int = 50,
                                min_gap_size: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect if image has multiple paintings and split vertically if needed (improved method).

    This is a simplified version that only performs vertical splitting (side-by-side paintings)
    using the improved thick row signal detection method.

    Args:
        image (np.ndarray): Original image.
        mask (np.ndarray): Segmentation mask.
        row_thickness (int): Row thickness for signal detection (for robust gap detection).
        min_gap_size (int): Minimum gap size to consider as split (pixels).

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (image_crop, mask_crop) tuples.
    """
    # Project mask to signal using thick row method
    signal = _project_mask_to_signal_thick(mask, row_thickness)

    # Detect vertical gap
    num_paintings, gap_start, gap_end = _detect_vertical_gap(signal, min_gap_size)

    if num_paintings == 1:
        # Single painting - return whole image and mask
        return [(image, mask)]

    # Two paintings - split at gap center
    split_col = (gap_start + gap_end) // 2

    left_image = image[:, :split_col]
    right_image = image[:, split_col:]

    left_mask = mask[:, :split_col]
    right_mask = mask[:, split_col:]

    return [(left_image, left_mask), (right_image, right_mask)]


def segment_multiple_paintings(image: np.ndarray,
                                border: int = 20,
                                dist_percentile: float = 92.0,
                                dist_margin: float = 0.5,
                                min_area: int = 200,
                                opening_size: int = 5,
                                closing_size: int = 7,
                                wL: float = 0.6, wA: float = 1.0, wB: float = 1.0,
                                sat_min: float = 30.0,
                                hue_percentile: float = 92.0,
                                hue_margin_deg: float = 6.0,
                                min_gap_size: int = 10,
                                return_debug_info: bool = False):
    """
    Segment multiple paintings in an image by detecting gaps and splitting.

    This function first performs an initial segmentation, then analyzes the mask
    to detect if there are multiple paintings (horizontal or vertical layout).
    If multiple paintings are detected, it splits the image and segments each part
    separately, then merges the results.

    Args:
        image (np.ndarray): Input image in BGR format.
        border (int): Width of border pixels for background estimation.
        dist_percentile (float): Percentile for LAB distance threshold.
        dist_margin (float): Margin added to LAB distance threshold.
        min_area (int): Minimum area for morphological filtering.
        opening_size (int): Size of morphological opening operation.
        closing_size (int): Size of morphological closing operation.
        wL (float): Weight for L channel in LAB distance.
        wA (float): Weight for A channel in LAB distance.
        wB (float): Weight for B channel in LAB distance.
        sat_min (float): Minimum saturation threshold for hue-based segmentation.
        hue_percentile (float): Percentile for hue distance threshold.
        hue_margin_deg (float): Margin in degrees added to hue distance threshold.
        min_gap_size (int): Minimum gap size (in pixels) to consider as separation between paintings.
        return_debug_info (bool): If True, return tuple of (mask, debug_info_dict).

    Returns:
        np.ndarray or tuple: Binary mask where foreground is 1 and background is 0.
                            If return_debug_info=True, returns (mask, debug_info_dict).
    """
    H, W = image.shape[:2]

    # Step 1: Perform initial segmentation
    initial_mask = segment_background(
        image, border=border, dist_percentile=dist_percentile, dist_margin=dist_margin,
        min_area=min_area, opening_size=opening_size, closing_size=closing_size,
        wL=wL, wA=wA, wB=wB, sat_min=sat_min, hue_percentile=hue_percentile,
        hue_margin_deg=hue_margin_deg
    )

    # Step 2: Analyze for horizontal split (paintings side by side)
    # Project along vertical axis (sum each column)
    horizontal_signal = _project_mask_to_signal(initial_mask, axis=0)
    num_h, gap_h_start, gap_h_end = _detect_gap_pattern(horizontal_signal, min_gap_size)

    # Step 3: Analyze for vertical split (paintings stacked)
    # Project along horizontal axis (sum each row)
    vertical_signal = _project_mask_to_signal(initial_mask, axis=1)
    num_v, gap_v_start, gap_v_end = _detect_gap_pattern(vertical_signal, min_gap_size)

    # Initialize debug info
    debug_info = {
        'initial_mask': initial_mask,
        'horizontal_signal': horizontal_signal,
        'vertical_signal': vertical_signal,
        'num_paintings_h': num_h,
        'num_paintings_v': num_v,
        'split_type': 'none',
        'split_position': None
    }

    # Step 4: Decide which split to use
    if num_h == 2 and num_v == 1:
        # Horizontal split (side by side)
        split_col = _find_split_position(gap_h_start, gap_h_end)

        # Split image
        left_image = image[:, :split_col]
        right_image = image[:, split_col:]

        return [True, [left_image, right_image]]

    elif num_v == 2 and num_h == 1:
        # Vertical split (stacked)
        split_row = _find_split_position(gap_v_start, gap_v_end)

        # Split image
        top_image = image[:split_row, :]
        bottom_image = image[split_row:, :]

        return [True, [top_image, bottom_image]]

    elif num_h == 2 and num_v == 2:
        # Both horizontal and vertical gaps detected - prioritize the larger gap
        h_gap_size = gap_h_end - gap_h_start
        v_gap_size = gap_v_end - gap_v_start

        if h_gap_size >= v_gap_size:
            # Use horizontal split
            split_col = _find_split_position(gap_h_start, gap_h_end)

            left_image = image[:, :split_col]
            right_image = image[:, split_col:]

            return [True, [left_image, right_image]]
        else:
            # Use vertical split
            split_row = _find_split_position(gap_v_start, gap_v_end)

            top_image = image[:split_row, :]
            bottom_image = image[split_row:, :]

            return [True, [top_image, bottom_image]]
    else:
        # Single painting or no clear separation detected
        return[False, image]


def segment_background(image: np.ndarray,
                                border: int = 20,
                                dist_percentile: float = 92.0,
                                dist_margin: float = 0.5,
                                min_area: int = 200,
                                opening_size: int = 5,
                                closing_size: int = 7,
                                wL: float = 0.6, wA: float = 1.0, wB: float = 1.0,
                                sat_min: float = 30.0,
                                hue_percentile: float = 92.0,
                                hue_margin_deg: float = 6.0,
                                apply_gaussian: bool = False,
                                sigma: float = 1.0,
                                radius: int = 2) -> np.ndarray:
    """
    Segment foreground from background using LAB and HSV color-based methods.

    Args:
        image (np.ndarray): Input image in BGR format.
        border (int): Width of border pixels for background estimation.
        dist_percentile (float): Percentile for LAB distance threshold.
        dist_margin (float): Margin added to LAB distance threshold.
        min_area (int): Minimum area for morphological filtering.
        opening_size (int): Size of morphological opening operation.
        closing_size (int): Size of morphological closing operation.
        wL (float): Weight for L channel in LAB distance.
        wA (float): Weight for A channel in LAB distance.
        wB (float): Weight for B channel in LAB distance.
        sat_min (float): Minimum saturation threshold for hue-based segmentation.
        hue_percentile (float): Percentile for hue distance threshold.
        hue_margin_deg (float): Margin in degrees added to hue distance threshold.
        apply_gaussian (bool): Whether to apply Gaussian filtering for denoising.
        sigma (float): Standard deviation for Gaussian kernel.
        radius (int): Radius of Gaussian kernel (truncate parameter).

    Returns:
        np.ndarray: Binary mask where foreground is 1 and background is 0.
    """
    # Apply Gaussian filter for denoising if requested
    if apply_gaussian:
        image = apply_gaussian_filter(image, sigma=sigma, radius=radius)

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]
    Hh = hsv[..., 0]  # [0,180)
    Ss = hsv[..., 1]  # [0,255]

    # 1) Background model from borders (LAB)
    stats = _estimate_bg_from_borders(lab, border=border)
    med = stats["median"]
    mad = stats["mad"]  # robust scales

    # Border mask (for threshold estimation)
    H, W = L.shape
    b = max(1, min(border, min(H, W)//4))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:b, :] = True; border_mask[-b:, :] = True
    border_mask[:, :b] = True; border_mask[:, -b:] = True

    # 3) Robust weighted LAB distance (downweight L)
    dist_lab = _lab_robust_distance_weighted(L, A, B, med, mad, wL=wL, wA=wA, wB=wB)

    # 4) Threshold from *border* distances (+ margin)
    border_dists = dist_lab[border_mask]
    if border_dists.size == 0:
        thr_lab = np.percentile(dist_lab, dist_percentile) + dist_margin
    else:
        thr_lab = np.percentile(border_dists, dist_percentile) + dist_margin
    mask_lab = dist_lab > thr_lab  # FG by LAB

    # 5) Hue fallback (HSV), purely color-based
    border_hues = Hh[border_mask]
    h0 = np.median(border_hues) if border_hues.size > 0 else np.median(Hh)
    hue_dist = _hue_circular_distance(Hh, h0)  # degrees in [0,90]
    border_hued = hue_dist[border_mask]
    thr_hue = (np.percentile(hue_dist, hue_percentile) + hue_margin_deg
               if border_hued.size == 0
               else np.percentile(border_hued, hue_percentile) + hue_margin_deg)
    mask_hue = (hue_dist > thr_hue) & (Ss >= sat_min)

    # Combine color cues
    mask = mask_lab | mask_hue

    # 6) Morph-only cleanup
    mask = _post_clean(mask, open_size=opening_size, close_size=closing_size, min_area=min_area)

    return mask.astype(np.uint8)

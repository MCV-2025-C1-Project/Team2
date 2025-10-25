# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, median_filter, morphological_gradient
from scipy import ndimage
import cv2
from sklearn.cluster import KMeans
import numpy as np
from scipy import ndimage
from skimage import morphology, segmentation



def convert_to_representation(image):
    """Convert RGB image to LAB and return all channels as float arrays."""
    im_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(im_lab)
    return L.astype(float), a.astype(float), b.astype(float)



def compute_edge_mask(im_lab, gradient_threshold=0.1):
    """Compute morphological gradient on LAB channels and combine."""
    L, a, b = im_lab
    
    # Compute morphological gradients per channel
    grad_l = morphological_gradient(L, structure=np.ones((5, 5)))
    grad_a = morphological_gradient(a, structure=np.ones((5, 5)))
    grad_b = morphological_gradient(b, structure=np.ones((5, 5)))

    # Combine gradients using Euclidean magnitude
    grad_combined = np.sqrt(grad_l**2 + grad_a**2 + grad_b**2)
    #grad_combined = grad_l
    #We apply a median filter
    #grad_combined = median_filter(grad_combined, size=5)

    

    # Threshold to highlight edges
    threshold = np.max(grad_combined) * gradient_threshold
    grad_bin = grad_combined > threshold



    # Apply opening to clean noise
    mask = grad_bin
    mask = binary_opening(grad_bin, structure=np.ones((3, 3)))

    return mask, grad_combined



def create_border_suppressed_mask(mask_bool, pixel_border, h, w):
    """Create a mask with border pixels suppressed for extreme point computation."""
    if not pixel_border or int(pixel_border) <= 0:
        return mask_bool
    
    mask_for_extremes = mask_bool.copy()
    pb = int(pixel_border)
    
    # Avoid border larger than dimensions
    pb_h = min(pb, h // 2)
    pb_w = min(pb, w // 2)
    
    # Suppress border pixels
    if pb_w > 0:
        mask_for_extremes[:, :pb_w] = False
        mask_for_extremes[:, w - pb_w:] = False
    if pb_h > 0:
        mask_for_extremes[:pb_h, :] = False
        mask_for_extremes[h - pb_h:, :] = False
    
    return mask_for_extremes


def compute_extreme_points(mask_for_extremes, h, w):
    """Compute leftmost, rightmost, topmost, and bottommost white pixels."""
    # Row-wise extremes (leftmost and rightmost white pixel per row)
    rows_any = mask_for_extremes.any(axis=1)
    left_all = mask_for_extremes.argmax(axis=1)
    right_all = w - 1 - np.argmax(mask_for_extremes[:, ::-1], axis=1)
    left = np.where(rows_any, left_all, -1)
    right = np.where(rows_any, right_all, -1)
    
    # Column-wise extremes (topmost and bottommost white pixel per column)
    cols_any = mask_for_extremes.any(axis=0)
    top_all = mask_for_extremes.argmax(axis=0)
    bottom_all = h - 1 - np.argmax(mask_for_extremes[::-1, :], axis=0)
    top = np.where(cols_any, top_all, -1)
    bottom = np.where(cols_any, bottom_all, -1)
    
    return left, right, top, bottom, rows_any, cols_any


def filter_extremes_by_median(extremes, threshold):
    """Filter extreme points by median deviation threshold."""
    filtered = extremes.copy()
    valid = filtered >= 0
    
    if valid.any():
        median_val = int(np.median(filtered[valid]))
        filtered[valid] = np.where(
            np.abs(filtered[valid] - median_val) <= threshold,
            filtered[valid],
            -1
        )
    
    return filtered


def create_border_mask(left, right, top, bottom, rows_any, cols_any, shape):
    """Create a border mask by marking filtered extreme points."""
    border_mask = np.zeros(shape, dtype=bool)
    
    # Mark left and right extremes for each row
    row_inds = np.where(rows_any)[0]
    for r in row_inds:
        if left[r] >= 0:
            border_mask[r, left[r]] = True
        if right[r] >= 0:
            border_mask[r, right[r]] = True
    
    # Mark top and bottom extremes for each column
    col_inds = np.where(cols_any)[0]
    for c in col_inds:
        if top[c] >= 0:
            border_mask[top[c], c] = True
        if bottom[c] >= 0:
            border_mask[bottom[c], c] = True
    
    return border_mask


def fit_x_of_y(y_indices, x_values):
    """Fit line x = a*y + b for vertical-like sides."""
    if len(y_indices) >= 2:
        a, b = np.polyfit(y_indices, x_values, 1)
    elif len(y_indices) == 1:
        a, b = 0.0, float(x_values[0])
    else:
        a, b = None, None
    return a, b


def fit_y_of_x(x_indices, y_values):
    """Fit line y = c*x + d for horizontal-like sides."""
    if len(x_indices) >= 2:
        c, d = np.polyfit(x_indices, y_values, 1)
    elif len(x_indices) == 1:
        c, d = 0.0, float(y_values[0])
    else:
        c, d = None, None
    return c, d


def intersect_lines(a, b, c, d):
    """Find intersection between lines x = a*y + b and y = c*x + d."""
    if a is None or c is None:
        return None
    
    denom = 1.0 - a * c
    if abs(denom) < 1e-6:
        return None
    
    x = (a * d + b) / denom
    y = c * x + d
    return (x, y)


def compute_polygon_corners(left, right, top, bottom, h, w):
    """Compute the four corners of the document polygon by fitting lines and finding intersections."""
    # Collect valid points for each side
    rows_left = np.where(left >= 0)[0]
    xs_left = left[rows_left] if rows_left.size > 0 else np.array([])
    
    rows_right = np.where(right >= 0)[0]
    xs_right = right[rows_right] if rows_right.size > 0 else np.array([])
    
    cols_top = np.where(top >= 0)[0]
    ys_top = top[cols_top] if cols_top.size > 0 else np.array([])
    
    cols_bottom = np.where(bottom >= 0)[0]
    ys_bottom = bottom[cols_bottom] if cols_bottom.size > 0 else np.array([])
    
    # Fit lines for each side
    a_left, b_left = fit_x_of_y(rows_left, xs_left)
    a_right, b_right = fit_x_of_y(rows_right, xs_right)
    c_top, d_top = fit_y_of_x(cols_top, ys_top)
    c_bottom, d_bottom = fit_y_of_x(cols_bottom, ys_bottom)
    
    # Compute four corner intersections
    corners = [
        intersect_lines(a_left, b_left, c_top, d_top),      # left-top
        intersect_lines(a_right, b_right, c_top, d_top),    # right-top
        intersect_lines(a_right, b_right, c_bottom, d_bottom),  # right-bottom
        intersect_lines(a_left, b_left, c_bottom, d_bottom)     # left-bottom
    ]
    
    # Apply fallback for None intersections
    final_corners = []
    for i, corner in enumerate(corners):
        if corner is None:
            # Use median of available points as fallback
            if i == 0:  # left-top
                x = np.median(xs_left) if xs_left.size > 0 else 0
                y = np.median(ys_top) if ys_top.size > 0 else 0
            elif i == 1:  # right-top
                x = np.median(xs_right) if xs_right.size > 0 else (w - 1)
                y = np.median(ys_top) if ys_top.size > 0 else 0
            elif i == 2:  # right-bottom
                x = np.median(xs_right) if xs_right.size > 0 else (w - 1)
                y = np.median(ys_bottom) if ys_bottom.size > 0 else (h - 1)
            else:  # left-bottom
                x = np.median(xs_left) if xs_left.size > 0 else 0
                y = np.median(ys_bottom) if ys_bottom.size > 0 else (h - 1)
            corner = (float(x), float(y))
        
        # Clip to image boundaries and convert to integers
        x_cl = int(np.clip(round(corner[0]), 0, w - 1))
        y_cl = int(np.clip(round(corner[1]), 0, h - 1))
        final_corners.append((x_cl, y_cl))
    
    return final_corners


def create_polygon_mask(corners, shape, fallback_mask):
    """Create a filled polygon mask from corner points."""
    h, w = shape
    polygon = np.array(corners, dtype=np.int32)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    
    try:
        cv2.fillPoly(poly_mask, [polygon], 1)
    except Exception:
        # Fallback to border mask if polygon is degenerate
        poly_mask = fallback_mask.astype(np.uint8)
    
    return poly_mask


def normalize_gradient(grad):
    """Normalize gradient values to 0-255 range for visualization."""
    grad_float = grad.astype(np.float32)
    grad_min = grad_float.min()
    grad_max = grad_float.max()
    grad_norm = 255 * (grad_float - grad_min) / ((grad_max - grad_min) + 1e-8)
    return grad_norm.astype(np.uint8)






def remove_background_morphological_gradient(im, thr=20, pixel_border=15, gradient_threshold=0.1):
    """
    Remove background from an image using morphological gradient detection and polygon fitting.
    
    Parameters:
    -----------
    image : ndarray
        Input RGB image
    thr : int
        Threshold for median filtering of extreme points
    pixel_border : int
        Number of border pixels to ignore when computing extreme points
    gradient_threshold : float
        Threshold for edge detection in morphological gradient
    
    Returns:
    --------
    tuple
        (original_image, mask, output_image, normalized_gradient)
    """
    # Preprocess
    
    im_lab = convert_to_representation(im)
    mask, grad = compute_edge_mask(im_lab, gradient_threshold)
    
    # Setup for border pixel processing
    mask_bool = mask.astype(bool)
    h, w = mask_bool.shape
    
    # Create mask with borders suppressed for extreme point computation
    mask_for_extremes = create_border_suppressed_mask(mask_bool, pixel_border, h, w)
    
    # Compute extreme points (leftmost, rightmost, topmost, bottommost)
    left, right, top, bottom, rows_any, cols_any = compute_extreme_points(
        mask_for_extremes, h, w
    )
    
    # Filter extremes by median deviation threshold
    left_filtered = filter_extremes_by_median(left, thr)
    right_filtered = filter_extremes_by_median(right, thr)
    top_filtered = filter_extremes_by_median(top, thr)
    bottom_filtered = filter_extremes_by_median(bottom, thr)
    
    # Create border mask from filtered extremes
    border_mask = create_border_mask(
        left_filtered, right_filtered, top_filtered, bottom_filtered,
        rows_any, cols_any, mask_bool.shape
    )


    
    # Fit lines and compute polygon corners
    final_corners = compute_polygon_corners(
        left_filtered, right_filtered, top_filtered, bottom_filtered, h, w
    )
    
    # Create polygon mask
    poly_mask = create_polygon_mask(final_corners, (h, w), border_mask)
    
    # Apply final mask to image
    output_image = im * poly_mask[:, :, np.newaxis]
    
    # Normalize gradient for visualization
    grad_norm = normalize_gradient(grad)

    ret1 = im.copy()
    ret2 = poly_mask.copy()
    ret3 = output_image.copy()
    ret4 = grad_norm.copy()

    """
    At this point:
    -im is the original image
    -poly_mask is the binary mask of the detected polygon
    -output_image is the image with background removed
    -grad_norm is the normalized gradient image for visualization
    """




    # ======================================================
    # 🔹 SEGUNDA FASE: ELIMINAR SOMBRAS
    # ======================================================

    NEARITY = 20  # Máximo número de píxeles "1s" que atravesamos antes de rendirnos

    scale = 1 # zoom in (mayor que 1)
    poly_mask = poly_mask.astype(np.uint8)

    # 1. Escalamos la máscara hacia arriba
    large = cv2.resize(poly_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    h, w = poly_mask.shape
    lh, lw = large.shape

    y_start = (lh - h) // 2
    x_start = (lw - w) // 2
    zoom_in_mask = large[y_start:y_start+h, x_start:x_start+w]

    # 1️⃣ Aplicamos closing 3x3 al gradiente binario
    grad_norm = np.where(grad_norm > np.max(grad_norm) * 0.1, 255, 0).astype(np.uint8)
    grad_norm = np.where(poly_mask > 0, grad_norm, 0).astype(np.uint8)
    grad_bin = cv2.morphologyEx(grad_norm, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))
    grad_bin = (grad_bin > 0).astype(np.uint8)
    grad_bin[grad_bin > 0] = 1

    h, w = grad_bin.shape
    zero_mask = (grad_bin == 0)

    shadows_candidates = np.zeros_like(poly_mask, dtype=np.int32)
    current_label = 1

    def is_cluster_in_upper_third(region, h):
        """Verifica si todos los píxeles del cluster están en el tercio superior."""
        coords = np.argwhere(region)
        if len(coords) == 0:
            return False
        max_y = np.max(coords[:, 0])
        upper_limit = h // 3
        return max_y < upper_limit

    def process_seed_and_label(r, c):
        nonlocal current_label
        if r < 0 or r >= h or c < 0 or c >= w:
            return
        if not zero_mask[r, c]:
            return
        region = segmentation.flood(zero_mask, (r, c), connectivity=1)
        
        # Filtrar clusters con menos de 100 píxeles
        region_size = np.sum(region)
        if region_size < 100:
            return
        
        # Filtrar clusters completamente en el tercio superior
        if is_cluster_in_upper_third(region, h):
            return
        
        shadows_candidates[region] = current_label
        current_label += 1

    # LEFT
    for r in range(h):
        state = 0  # 0 = esperando gradiente, 1 = dentro de gradiente
        gradient_count = 0
        for x in range(w):
            if grad_bin[r, x] > 0 and state == 0:
                state = 1  # hemos entrado en zona de gradiente
                gradient_count = 1
            elif grad_bin[r, x] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break  # nos rendimos
            elif grad_bin[r, x] == 0 and state == 1:
                # hemos salido de la zona de gradiente: 0-(1s)-0 detectado
                process_seed_and_label(r, x)
                break  # detenemos búsqueda para este r

    # RIGHT
    for r in range(h):
        state = 0
        gradient_count = 0
        for x in range(w - 1, -1, -1):
            if grad_bin[r, x] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[r, x] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[r, x] == 0 and state == 1:
                process_seed_and_label(r, x)
                break

    # TOP
    for c in range(w):
        state = 0
        gradient_count = 0
        for y in range(h):
            if grad_bin[y, c] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[y, c] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[y, c] == 0 and state == 1:
                process_seed_and_label(y, c)
                break

    # BOTTOM
    for c in range(w):
        state = 0
        gradient_count = 0
        for y in range(h - 1, -1, -1):
            if grad_bin[y, c] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[y, c] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[y, c] == 0 and state == 1:
                process_seed_and_label(y, c)
                break

    # ========== FUNCIÓN AUXILIAR PARA VERIFICAR AGUJEROS ==========
    def cluster_has_holes(cluster_mask):
        """Verifica si un cluster tiene agujeros (regiones de 0s rodeadas por 1s)."""
        # Invertir la máscara
        inverted = (~cluster_mask.astype(bool)).astype(np.uint8)
        
        # Etiquetar componentes conectadas en la máscara invertida
        num_labels, labels = cv2.connectedComponents(inverted, connectivity=8)
        
        # Si hay más de 2 componentes (background exterior + al menos un agujero), tiene agujeros
        # num_labels incluye el background (0), así que si > 2 hay agujeros
        return num_labels > 2

    # ========== SECCIÓN FÁCILMENTE COMENTABLE: FILTRADO POR DIÁMETROS ==========
    # Calcular diámetros máximos de poly_mask
    poly_rows = np.any(poly_mask > 0, axis=1)
    poly_cols = np.any(poly_mask > 0, axis=0)
    Vm = np.sum(poly_rows)  # Diámetro vertical máximo
    Hm = np.sum(poly_cols)  # Diámetro horizontal máximo

    # Filtrar clusters por diámetros
    unique_labels = np.unique(shadows_candidates)
    unique_labels = unique_labels[unique_labels > 0]

    shadows_candidates2 = shadows_candidates.copy()

    for label in unique_labels:
        cluster_mask = (shadows_candidates == label)
        
        # Calcular diámetros del cluster
        cluster_rows = np.any(cluster_mask, axis=1)
        cluster_cols = np.any(cluster_mask, axis=0)
        cluster_V = np.sum(cluster_rows)  # Diámetro vertical
        cluster_H = np.sum(cluster_cols)  # Diámetro horizontal
        
        # Verificar condiciones de diámetro
        meets_V = cluster_V >= (3/5) * Vm
        meets_H = cluster_H >= (3/5) * Hm
        
        # Verificar si el cluster tiene agujeros
        has_holes = cluster_has_holes(cluster_mask)
        
        # Debe cumplir al menos uno de los dos
        if not (meets_V or meets_H):
            shadows_candidates[cluster_mask] = 0
            shadows_candidates2[cluster_mask] = 0
            continue
        
        # Si cumple solo uno, verificar que el otro no supere 2/5
        if meets_V and not meets_H:
            if cluster_H > (3/5) * Hm:
                shadows_candidates[cluster_mask] = 0
                shadows_candidates2[cluster_mask] = 0
        elif meets_H and not meets_V:
            if cluster_V > (3/5) * Vm:
                shadows_candidates[cluster_mask] = 0
                shadows_candidates2[cluster_mask] = 0
        elif meets_V and meets_H:
            shadows_candidates[cluster_mask] = 0
        
        # Si el cluster tiene agujeros, no pasa a shadows_candidates (CF)
        # pero se queda en shadows_candidates2 (C)
        if has_holes:
            shadows_candidates[cluster_mask] = 0
    # ========== FIN SECCIÓN COMENTABLE ==========

    print(np.unique(grad_bin))
    print(np.unique(shadows_candidates))

    # ========== FUNCIONES AUXILIARES PARA ALGORITMO FINAL ==========
    def find_background_cluster(clusters_map):
        """Detecta si existe un cluster que toca el borde de la imagen."""
        h, w = clusters_map.shape
        unique_labels = np.unique(clusters_map)
        unique_labels = unique_labels[unique_labels > 0]
        
        for label in unique_labels:
            cluster_mask = (clusters_map == label)
            # Verificar si toca algún borde
            if (np.any(cluster_mask[0, :]) or np.any(cluster_mask[-1, :]) or 
                np.any(cluster_mask[:, 0]) or np.any(cluster_mask[:, -1])):
                return label
        return None

    def find_max_interior_quadrilateral(cluster_mask, original_mask, start_point=None):
        """
        Encuentra el cuadrilátero de área máxima contenido en el agujero de un cluster.
        Si start_point es None, usa el centro de la imagen.
        """
        h, w = cluster_mask.shape
        
        # Invertir el cluster para que el agujero sea 1 y el cluster sea 0
        hole_mask = (~cluster_mask.astype(bool)).astype(np.uint8)
        hole_mask = hole_mask & original_mask  # Intersección con máscara original
        
        if start_point is None:
            # Usar centro de la imagen
            center_y, center_x = h // 2, w // 2
        else:
            center_y, center_x = start_point
        
        # Verificar que el punto de inicio está en el agujero
        if not hole_mask[center_y, center_x]:
            # Si no está en el agujero, buscar el punto más cercano que sí esté
            coords = np.argwhere(hole_mask > 0)
            if len(coords) == 0:
                return np.zeros((h, w), dtype=np.uint8)
            distances = np.sum((coords - np.array([center_y, center_x]))**2, axis=1)
            closest = coords[np.argmin(distances)]
            center_y, center_x = closest[0], closest[1]
        
        # Expandir desde el centro hacia los bordes
        # Buscar extremos en cada dirección dentro del agujero
        left = center_x
        while left > 0 and hole_mask[center_y, left - 1]:
            left -= 1
        
        right = center_x
        while right < w - 1 and hole_mask[center_y, right + 1]:
            right += 1
        
        top = center_y
        while top > 0 and hole_mask[top - 1, center_x]:
            top -= 1
        
        bottom = center_y
        while bottom < h - 1 and hole_mask[bottom + 1, center_x]:
            bottom += 1
        
        # Ahora expandir el rectángulo completo
        # Buscar el rectángulo máximo que cabe en el agujero
        while left > 0:
            can_expand = True
            for y in range(top, bottom + 1):
                if not hole_mask[y, left - 1]:
                    can_expand = False
                    break
            if can_expand:
                left -= 1
            else:
                break
        
        while right < w - 1:
            can_expand = True
            for y in range(top, bottom + 1):
                if not hole_mask[y, right + 1]:
                    can_expand = False
                    break
            if can_expand:
                right += 1
            else:
                break
        
        while top > 0:
            can_expand = True
            for x in range(left, right + 1):
                if not hole_mask[top - 1, x]:
                    can_expand = False
                    break
            if can_expand:
                top -= 1
            else:
                break
        
        while bottom < h - 1:
            can_expand = True
            for x in range(left, right + 1):
                if not hole_mask[bottom + 1, x]:
                    can_expand = False
                    break
            if can_expand:
                bottom += 1
            else:
                break
        
        # Crear máscara del cuadrilátero
        result_mask = np.zeros((h, w), dtype=np.uint8)
        result_mask[top:bottom+1, left:right+1] = 1
        
        return result_mask

    def combine_cluster_masks(clusters_map):
        """Combina todos los clusters en una sola máscara."""
        return (clusters_map > 0).astype(np.uint8)

    # ========== ALGORITMO PRINCIPAL ==========
    C = shadows_candidates2  # Clusters originales
    CF = shadows_candidates  # Clusters filtrados

    # Paso 1: Buscar cluster background en C
    bg_cluster_label = find_background_cluster(C)

    # Contar clusters en C y CF
    unique_C = np.unique(C)
    unique_C = unique_C[unique_C > 0]
    num_clusters_C = len(unique_C)

    unique_CF = np.unique(CF)
    unique_CF = unique_CF[unique_CF > 0]
    num_clusters_CF = len(unique_CF)

    print(f"DEBUG: Background cluster: {bg_cluster_label}")
    print(f"DEBUG: Num clusters C: {num_clusters_C}, CF: {num_clusters_CF}")

    # Paso 2 y 3: Si no existe cluster background
    if bg_cluster_label is None:
        print("DEBUG: Caso 2-3: No hay cluster background, final_mask = máscara original")
        final_mask = poly_mask.copy()
    else:
        # Paso 4: Existe cluster background
        # Paso 5: Si es el único cluster en C
        if num_clusters_C == 1:
            print("DEBUG: Caso 5: Solo cluster background, final_mask = cuadrilátero máximo interior")
            bg_mask = (C == bg_cluster_label).astype(bool)
            final_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)
        # Paso 6: Más de un cluster en C pero NO hay clusters en CF
        elif num_clusters_C > 1 and num_clusters_CF == 0:
            print("DEBUG: Caso 6: Múltiples clusters en C, ninguno en CF")
            # Combinar todos los clusters
            all_clusters_mask = combine_cluster_masks(C).astype(bool)
            
            # Intentar encontrar cuadrilátero desde el centro
            h, w = poly_mask.shape
            center = (h // 2, w // 2)
            
            # Verificar si el centro está en un cluster
            if C[center[0], center[1]] > 0:
                print("DEBUG: Centro está en un cluster, usar cuadrilátero de background")
                bg_mask = (C == bg_cluster_label).astype(bool)
                final_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)
            else:
                # Buscar submáscara desde el centro
                candidate_mask = find_max_interior_quadrilateral(all_clusters_mask, poly_mask, center)
                
                # Verificar si contiene al menos 70% del área original
                area_original = np.sum(poly_mask)
                area_candidate = np.sum(candidate_mask)
                
                if area_candidate >= 0.7 * area_original:
                    print(f"DEBUG: Submáscara válida ({area_candidate/area_original*100:.1f}% del área)")
                    final_mask = candidate_mask
                else:
                    print(f"DEBUG: Submáscara muy pequeña ({area_candidate/area_original*100:.1f}%), usar background")
                    bg_mask = (C == bg_cluster_label).astype(bool)
                    final_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)
        # Paso 7: CF tiene exactamente un cluster
        elif num_clusters_CF == 1:
            print("DEBUG: Caso 7: Un cluster en CF, cuadrilátero máximo interior")
            cf_label = unique_CF[0]
            cf_mask = (CF == cf_label).astype(bool)
            final_mask = find_max_interior_quadrilateral(cf_mask, poly_mask)
        else:
            # Caso no definido: imagen toda blanca
            print("DEBUG: Caso no definido, final_mask = todo blanco")
            final_mask = np.ones((h, w), dtype=np.uint8)

    # Convertir shadows_candidates a visualización RGBA
    unique_labels = np.unique(shadows_candidates)
    unique_labels = unique_labels[unique_labels > 0]  # excluir el 0 (fondo)

    # Crear imagen RGBA (con canal alpha para transparencia)
    shadows_candidates_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Generar colores aleatorios para cada cluster
    np.random.seed(42)  # para reproducibilidad
    colors = np.random.randint(50, 255, size=(len(unique_labels) + 1, 3))

    for i, label in enumerate(unique_labels):
        mask_label = (shadows_candidates == label)
        shadows_candidates_rgba[mask_label, :3] = colors[i]  # RGB
        shadows_candidates_rgba[mask_label, 3] = 255  # Alpha (opaco)

    # Reemplazamos shadows_candidates por la versión RGBA visualizable
    shadows_candidates = shadows_candidates_rgba

    # Convertir shadows_candidates2 a visualización RGBA
    unique_labels = np.unique(shadows_candidates2)
    unique_labels = unique_labels[unique_labels > 0]  # excluir el 0 (fondo)

    # Crear imagen RGBA (con canal alpha para transparencia)
    shadows_candidates_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Generar colores aleatorios para cada cluster
    np.random.seed(42)  # para reproducibilidad
    colors = np.random.randint(50, 255, size=(len(unique_labels) + 1, 3))

    for i, label in enumerate(unique_labels):
        mask_label = (shadows_candidates2 == label)
        shadows_candidates_rgba[mask_label, :3] = colors[i]  # RGB
        shadows_candidates_rgba[mask_label, 3] = 255  # Alpha (opaco)

    # Reemplazamos shadows_candidates2 por la versión RGBA visualizable
    shadows_candidates2 = shadows_candidates_rgba

    return ret1, ret2, ret3, ret4, grad_bin, shadows_candidates, shadows_candidates2, final_mask
import cv2
import numpy as np

def calculate_variance(image):
    """Calculate the variance for each row and column."""
    row_variances = np.var(image, axis=1)
    col_variances = np.var(image, axis=0)
    return row_variances, col_variances

def smooth_variance(variances, window_size=5):
    """Smooth the variance using a moving average."""
    return np.convolve(variances, np.ones(window_size)/window_size, mode='same')

def find_valleys_in_column_variance(col_variances, threshold=10):
    """Find valleys between two peaks in the column variance with a stricter threshold to avoid outliers."""
    valleys = []
    for i in range(200, len(col_variances) - 200):
        if col_variances[i-1] > col_variances[i] and col_variances[i+1] > col_variances[i] and col_variances[i] < threshold:
            valleys.append(i)
    return valleys

def split_image_at_valley(image, valleys):
    """Split the image at the detected valley to separate two artworks."""
    if len(valleys) >= 1:
        valley_position = valleys[0]
        left_artwork = image[:, :valley_position]
        right_artwork = image[:, valley_position:]
        return (left_artwork, right_artwork)
    return image

def fit_polynomial(variances, degree=5):
    """Fit a polynomial to the smoothed variance."""
    x = np.arange(len(variances))
    coeffs = np.polyfit(x, variances, degree)
    poly = np.poly1d(coeffs)
    return poly(x)

def split_images(img):
    """Encapsula la lógica de detección y separación de imágenes.
       Devuelve (valleys, imagen_o_tupla)
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, col_variances = calculate_variance(gray_img)
    smoothed_col_variances = smooth_variance(col_variances, window_size=200)
    valleys = find_valleys_in_column_variance(smoothed_col_variances, threshold=750)

    if valleys:
        splitted_imgs = split_image_at_valley(img, valleys)
        
        return valleys, splitted_imgs
    else:
        return valleys, img
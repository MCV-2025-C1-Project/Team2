import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_images(img, show_detection=False):
    """
    Detecta si hay uno o dos cuadros en la imagen y los separa usando gradiente morfológico.
    
    Args:
        img: imagen BGR de OpenCV
        show_detection: si True, muestra el proceso de detección de bordes
        
    Returns:
        - Si hay 2 cuadros: tupla (left_artwork, right_artwork)
        - Si hay 1 cuadro: imagen completa
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # GRADIENTE MORFOLÓGICO en lugar de Canny
    # El gradiente morfológico = dilatación - erosión
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(blurred, kernel, iterations=1)
    eroded = cv2.erode(blurred, kernel, iterations=1)
    morph_gradient = cv2.subtract(dilated, eroded)
    
    # Binarizar el gradiente morfológico
    _, binary = cv2.threshold(morph_gradient, 30, 255, cv2.THRESH_BINARY)
    
    # Dilatar para conectar bordes cercanos
    kernel_connect = np.ones((7, 7), np.uint8)
    connected = cv2.dilate(binary, kernel_connect, iterations=2)
    
    # Visualizar detección de bordes si se solicita
    if show_detection:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(blurred, cmap='gray')
        plt.title('Blurred')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(morph_gradient, cmap='gray')
        plt.title('Morphological Gradient')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(connected, cmap='gray')
        plt.title('Binary + Dilated')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos detectados en una copia de la imagen
    if show_detection:
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.title(f'Contornos Detectados: {len(contours)} total')
        plt.axis('off')
        plt.show()
    
    # Filtrar contornos por área (ignorar contornos muy pequeños)
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * 0.05  # Al menos 5% del área total
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            # Filtrar por relación de aspecto razonable (no demasiado estrecho)
            if 0.2 < aspect_ratio < 5.0:
                valid_contours.append((x, y, w, h, area))
    
    # Ordenar por posición horizontal (izquierda a derecha)
    valid_contours.sort(key=lambda c: c[0])
    
    # Visualizar contornos válidos
    if show_detection and len(valid_contours) > 0:
        img_valid_contours = img.copy()
        for i, (x, y, w, h, area) in enumerate(valid_contours):
            cv2.rectangle(img_valid_contours, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(img_valid_contours, f'#{i+1}', (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_valid_contours, cv2.COLOR_BGR2RGB))
        plt.title(f'Contornos Válidos: {len(valid_contours)}')
        plt.axis('off')
        plt.show()
    
    # Si hay 2 o más contornos válidos, intentar separar
    if len(valid_contours) >= 2:
        # Tomar los dos contornos más grandes
        sorted_by_area = sorted(valid_contours, key=lambda c: c[4], reverse=True)
        top_two = sorted_by_area[:2]
        
        # Ordenar por posición horizontal
        top_two.sort(key=lambda c: c[0])
        
        x1, y1, w1, h1, _ = top_two[0]
        x2, y2, w2, h2, _ = top_two[1]
        
        # Verificar que están suficientemente separados horizontalmente
        separation = x2 - (x1 + w1)
        
        if separation > 20:  # Al menos 20 píxeles de separación
            # Añadir un pequeño margen
            margin = 10
            
            # Extraer imagen izquierda
            x1_start = max(0, x1 - margin)
            x1_end = min(img.shape[1], x1 + w1 + margin)
            y1_start = max(0, y1 - margin)
            y1_end = min(img.shape[0], y1 + h1 + margin)
            left_artwork = img[y1_start:y1_end, x1_start:x1_end]
            
            # Extraer imagen derecha
            x2_start = max(0, x2 - margin)
            x2_end = min(img.shape[1], x2 + w2 + margin)
            y2_start = max(0, y2 - margin)
            y2_end = min(img.shape[0], y2 + h2 + margin)
            right_artwork = img[y2_start:y2_end, x2_start:x2_end]
            
            return (left_artwork, right_artwork)
    
    # Si no se detectaron 2 cuadros separados, devolver la imagen completa
    return img


def split_images_simple(img, threshold=0.4):
    """
    Método alternativo más simple: divide la imagen verticalmente si detecta
    dos regiones con contenido significativo.
    
    Args:
        img: imagen BGR de OpenCV
        threshold: umbral para detectar contenido (0-1)
        
    Returns:
        - Si hay 2 cuadros: tupla (left_artwork, right_artwork)
        - Si hay 1 cuadro: imagen completa
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcular la suma vertical de intensidad (proyección vertical)
    vertical_projection = np.sum(gray, axis=0)
    
    # Normalizar
    vertical_projection = vertical_projection / np.max(vertical_projection)
    
    # Encontrar el centro de la imagen
    center = img.shape[1] // 2
    search_range = img.shape[1] // 4
    
    # Buscar el mínimo alrededor del centro (posible separación)
    left_bound = max(0, center - search_range)
    right_bound = min(img.shape[1], center + search_range)
    
    min_pos = left_bound + np.argmin(vertical_projection[left_bound:right_bound])
    min_value = vertical_projection[min_pos]
    
    # Si hay un valle significativo, dividir
    if min_value < threshold:
        left_artwork = img[:, :min_pos]
        right_artwork = img[:, min_pos:]
        
        # Verificar que ambas partes tengan tamaño razonable
        if left_artwork.shape[1] > img.shape[1] * 0.2 and right_artwork.shape[1] > img.shape[1] * 0.2:
            return (left_artwork, right_artwork)
    
    return img
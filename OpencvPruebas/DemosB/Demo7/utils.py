"""
Funciones auxiliares y de debug
"""

import cv2
import numpy as np
from config import *


def rotate_image(image, rotation):
    """
    Rota una imagen según el código de rotación
    0: sin rotación, 1: 90° derecha, 2: 180°, 3: 270° derecha
    """
    if rotation == 0:
        return image
    elif rotation == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def draw_grid(image, tile_size, reference_pos):
    """
    Dibuja la cuadrícula sobre la imagen para debug
    """
    debug_img = image.copy()
    h, w = image.shape[:2]
    
    ref_x, ref_y = reference_pos
    
    # Dibujar líneas verticales
    x = ref_x
    while x < w:
        cv2.line(debug_img, (x, 0), (x, h), DEBUG_COLOR_GRID, DEBUG_LINE_THICKNESS)
        x += tile_size
    x = ref_x
    while x > 0:
        cv2.line(debug_img, (x, 0), (x, h), DEBUG_COLOR_GRID, DEBUG_LINE_THICKNESS)
        x -= tile_size
    
    # Dibujar líneas horizontales
    y = ref_y
    while y < h:
        cv2.line(debug_img, (0, y), (w, y), DEBUG_COLOR_GRID, DEBUG_LINE_THICKNESS)
        y += tile_size
    y = ref_y
    while y > 0:
        cv2.line(debug_img, (0, y), (w, y), DEBUG_COLOR_GRID, DEBUG_LINE_THICKNESS)
        y -= tile_size
    
    # Destacar el cuadrado de referencia
    cv2.rectangle(debug_img, 
                  (ref_x, ref_y), 
                  (ref_x + tile_size, ref_y + tile_size), 
                  DEBUG_COLOR_REFERENCE, 
                  DEBUG_LINE_THICKNESS * 2)
    
    # Agregar texto indicando el cuadrado de referencia
    cv2.putText(debug_img, "REFERENCIA", 
                (ref_x + 5, ref_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, DEBUG_COLOR_REFERENCE, 2)
    
    return debug_img


def save_debug_image(image, filename=OUTPUT_DEBUG_IMAGE):
    """
    Guarda la imagen de debug
    """
    cv2.imwrite(filename, image)
    print(f"[DEBUG] Imagen guardada en: {filename}")


def show_debug_image(image, window_name="Debug"):
    """
    Muestra la imagen de debug (útil para desarrollo)
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_images(img1, img2):
    """
    Compara dos imágenes y retorna un score de similitud (0-1)
    Usa correlación de histogramas y comparación estructural
    """
    # Asegurar mismo tamaño
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Método 1: Comparación de histogramas
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return max(0, similarity)  # Asegurar que esté entre 0 y 1
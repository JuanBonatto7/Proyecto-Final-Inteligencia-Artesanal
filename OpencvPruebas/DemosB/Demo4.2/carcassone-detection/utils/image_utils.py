"""
Utilidades para procesamiento de imágenes.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image_keep_aspect(
    image: np.ndarray,
    max_width: int,
    max_height: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Redimensiona imagen manteniendo aspect ratio.
    
    Args:
        image: Imagen a redimensionar
        max_width: Ancho máximo
        max_height: Alto máximo (opcional)
        
    Returns:
        Tupla (imagen_redimensionada, escala_aplicada)
    """
    height, width = image.shape[:2]
    
    if width <= max_width and (max_height is None or height <= max_height):
        return image, 1.0
    
    scale = max_width / width
    
    if max_height is not None:
        scale_height = max_height / height
        scale = min(scale, scale_height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized, scale


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convierte imagen a escala de grises si es necesario.
    
    Args:
        image: Imagen de entrada
        
    Returns:
        Imagen en escala de grises
    """
    if len(image.shape) == 2:
        return image
    
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    return image


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Carga una imagen desde archivo.
    
    Args:
        path: Ruta al archivo de imagen
        
    Returns:
        Array numpy con la imagen o None si falla
    """
    image = cv2.imread(path)
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {path}")
        return None
    
    return image


def save_image(image: np.ndarray, path: str) -> bool:
    """
    Guarda una imagen a archivo.
    
    Args:
        image: Imagen a guardar
        path: Ruta destino
        
    Returns:
        True si se guardó exitosamente, False si falló
    """
    result = cv2.imwrite(path, image)
    
    if not result:
        print(f"Error: No se pudo guardar la imagen en {path}")
    
    return result


def crop_rectangle(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Recorta un rectángulo de la imagen dados sus 4 vértices.
    
    Args:
        image: Imagen fuente
        corners: Array (4, 2) con las coordenadas de las esquinas
        
    Returns:
        Imagen recortada y enderezada
    """
    # Ordenar esquinas: top-left, top-right, bottom-right, bottom-left
    rect = order_points(corners)
    
    (tl, tr, br, bl) = rect
    
    # Calcular ancho del nuevo rectángulo
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Calcular alto del nuevo rectángulo
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Coordenadas destino
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Calcular matriz de perspectiva y aplicar
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena puntos en orden: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array (4, 2) con coordenadas de puntos
        
    Returns:
        Array ordenado (4, 2)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left tendrá la suma más pequeña
    # Bottom-right tendrá la suma más grande
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right tendrá la diferencia más pequeña
    # Bottom-left tendrá la diferencia más grande
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect
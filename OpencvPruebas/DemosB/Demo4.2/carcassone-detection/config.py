"""
Configuración global del sistema de detección de Carcassonne (OPTIMIZADO).
"""

from typing import Tuple


class Config:
    """Configuración centralizada del sistema."""
    
    # Configuración de imagen
    MAX_IMAGE_WIDTH: int = 800
    
    # Configuración de Blur
    BLUR_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    BLUR_SIGMA: float = 0
    
    # Configuración de Canny Edge Detection
    CANNY_THRESHOLD_1: int = 50
    CANNY_THRESHOLD_2: int = 150
    CANNY_APERTURE_SIZE: int = 3
    
    # Configuración de Dilate
    DILATE_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    DILATE_ITERATIONS: int = 2  # Aumentado para conectar mejor las líneas
    
    # Configuración de Hough Line Transform (OPTIMIZADO)
    HOUGH_RHO: float = 1.0
    HOUGH_THETA: float = 3.14159265 / 180
    HOUGH_THRESHOLD: int = 80  # Aumentado para filtrar líneas débiles
    HOUGH_MIN_LINE_LENGTH: int = 100  # Aumentado para líneas más largas
    HOUGH_MAX_LINE_GAP: int = 15  # Aumentado para unir segmentos
    
    # Configuración de intersecciones (OPTIMIZADO)
    INTERSECTION_MIN_DISTANCE: float = 20.0  # Aumentado para agrupar mejor
    INTERSECTION_ANGLE_THRESHOLD: float = 10.0
    
    # Configuración de RANSAC (OPTIMIZADO)
    RANSAC_THRESHOLD: float = 3.0  # Más estricto
    RANSAC_MAX_ITERATIONS: int = 2000
    RANSAC_CONFIDENCE: float = 0.99
    
    # Configuración de detección de fichas
    MIN_TILE_SIZE: int = 30  # Reducido para detectar fichas más pequeñas
    MAX_TILE_SIZE: int = 400  # Aumentado para fichas grandes
    TILE_GRID_ROWS: int = 30  # Aumentado para tableros más grandes
    TILE_GRID_COLS: int = 30
    
    # Configuración de clasificación (OPTIMIZADO)
    FEATURE_DESCRIPTOR: str = "ORB"  # ORB, SIFT, AKAZE
    MATCH_RATIO_THRESHOLD: float = 0.7  # Más estricto para mejores matches
    MIN_MATCHES_THRESHOLD: int = 8  # Reducido para ser más permisivo
    
    # Rutas
    TILES_TEMPLATE_PATH: str = "data/tiles/templates/"
    
    # Visualización
    VISUALIZATION_WINDOW_NAME: str = "Carcassonne Detector"
    VISUALIZATION_WAIT_TIME: int = 0  # 0 = esperar tecla
    
    # Debug
    DEBUG_MODE: bool = True
    SHOW_INTERMEDIATE_STEPS: bool = True


# Letras de fichas del Carcassonne (A-Y)
TILE_TYPES = [chr(i) for i in range(ord('A'), ord('Y') + 1)]

# Rotaciones posibles (0, 1, 2, 3)
ROTATIONS = [0, 1, 2, 3]
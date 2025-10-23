"""
Configuración global del sistema de detección de Carcassonne.
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
    DILATE_ITERATIONS: int = 1
    
    # Configuración de Hough Line Transform
    HOUGH_RHO: float = 1.0
    HOUGH_THETA: float = 3.14159265 / 180
    HOUGH_THRESHOLD: int = 50
    HOUGH_MIN_LINE_LENGTH: int = 50
    HOUGH_MAX_LINE_GAP: int = 10
    
    # Configuración de intersecciones
    INTERSECTION_MIN_DISTANCE: float = 10.0
    INTERSECTION_ANGLE_THRESHOLD: float = 10.0
    
    # Configuración de RANSAC
    RANSAC_THRESHOLD: float = 5.0
    RANSAC_MAX_ITERATIONS: int = 2000
    RANSAC_CONFIDENCE: float = 0.99
    
    # Configuración de detección de fichas
    MIN_TILE_SIZE: int = 50
    MAX_TILE_SIZE: int = 300
    TILE_GRID_ROWS: int = 10
    TILE_GRID_COLS: int = 10
    
    # Configuración de clasificación
    FEATURE_DESCRIPTOR: str = "ORB"  # ORB, SIFT, AKAZE
    MATCH_RATIO_THRESHOLD: float = 0.75
    MIN_MATCHES_THRESHOLD: int = 10
    
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

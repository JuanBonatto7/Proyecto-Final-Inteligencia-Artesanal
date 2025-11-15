# ============================================================================
# config/settings.py
# ============================================================================

import numpy as np
from typing import Tuple, Dict

class Config:
    """Configuración global del sistema."""
    
    # Dimensiones de la imagen
    IMAGE_SIZE = (800, 800)
    
    # Colores RGB (basados en análisis de imágenes reales)
    COLORS: Dict[str, Tuple[int, int, int]] = {
        'FIELD': (0, 191, 98),       # Verde
        'CASTLE': (255, 145, 76),    # Naranja
        'ROAD': (63, 72, 204),       # Azul
        'VERTEX': (237, 28, 36),     # Rojo (antes "CHURCH", ahora vértices)
        'MEEPLE_1': (156, 0, 1),     # Rojo oscuro/morado
        'MEEPLE_2': (0, 0, 0),       # Negro
        'EMPTY': (255, 255, 255)     # Blanco (loseta vacía)
    }
    
    # Tolerancias para detección de colores
    COLOR_TOLERANCE = 10
    
    # Puntos por castillo cerrado que toca el campo
    POINTS_PER_CLOSED_CASTLE = 3
    
    # Configuración de conectividad
    CONNECTIVITY = 8  # Para cv2.connectedComponents (4 u 8)
    
    # Configuración de morfología
    MORPH_KERNEL_SIZE = 3
    
    # Umbral mínimo de área para considerar un campo válido (píxeles)
    MIN_FIELD_AREA = 100
    
    # Umbral mínimo de área para considerar un meeple válido (píxeles)
    MIN_MEEPLE_AREA = 10
    
    # Directorio de salida
    OUTPUT_DIR = 'output'
"""
Configuración del sistema de detección de Carcassonne
"""

# Dimensiones
TILE_SIZE = 100  # Tamaño esperado de una loceta en píxeles (ajustar según tu caso)

# Rutas
REFERENCIAS_DIR = "referencias"
INPUT_IMAGE = "tablero.jpg"
OUTPUT_DEBUG_IMAGE = "debug_tablero.jpg"

# Detección
SIMILARITY_THRESHOLD = 0.75  # Umbral de similitud para considerar un match (0-1)

# Debug
DEBUG_MODE = True  # Activar visualización de debug
DEBUG_COLOR_GRID = (0, 255, 0)  # Verde para la cuadrícula
DEBUG_COLOR_REFERENCE = (255, 0, 0)  # Azul para el cuadrado de referencia
DEBUG_LINE_THICKNESS = 2

# Locetas disponibles
TILES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'BLANCA']

# Rotaciones posibles
ROTATIONS = [0, 1, 2, 3]
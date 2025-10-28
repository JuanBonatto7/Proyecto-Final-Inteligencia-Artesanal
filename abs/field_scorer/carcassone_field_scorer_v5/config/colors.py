"""
Configuración de colores para el tablero de Carcassonne.
"""

# Colores en formato RGB
COLORS = {
    'FIELD': (34, 177, 76),           # Verde
    'CASTLE': (255, 127, 39),         # Naranja
    'ROAD': (63, 72, 204),            # Azul (caminos)
    'CHURCH': (237, 28, 36),          # Rojo
    'MEEPLE_1': (163, 73, 164),       # Violeta
    'MEEPLE_2': (0, 0, 0),            # Negro
}

# Tolerancia para la detección de colores
COLOR_TOLERANCE = 40

# Nombres de jugadores
PLAYER_NAMES = {
    'MEEPLE_1': 'Jugador 1',
    'MEEPLE_2': 'Jugador 2',
}

# Configuración avanzada para detección de campos
FIELD_DETECTION_CONFIG = {
    'barrier_expansion': 4,  # Cuánto expandir las barreras para separar campos
    'min_field_area': 100,   # Área mínima en píxeles para considerar un campo válido
    'meeple_detection_threshold': 15,  # Umbral de píxeles de meeple para contar como presente
}
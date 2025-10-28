"""
Configuración de colores para el tablero de Carcassonne.
Modifica estos valores según necesites.
"""

# Colores en formato RGB (valores enteros simples)
COLORS = {
    'FIELD': (34, 177, 76),           # Verde
    'CASTLE': (255, 127, 39),         # Naranja
    'ROAD': (63, 72, 204),            # Azul (caminos)
    'CHURCH': (237, 28, 36),          # Rojo
    'MEEPLE_1': (163, 73, 164),       # Violeta
    'MEEPLE_2': (0, 0, 0),            # Negro
}

# Tolerancia para la detección de colores
# Aumentada para manejar variaciones en la imagen
COLOR_TOLERANCE = 40

# Nombre de jugadores
PLAYER_NAMES = {
    'MEEPLE_1': 'Jugador 1',
    'MEEPLE_2': 'Jugador 2',
}

# Configuración avanzada para detección de campos
FIELD_DETECTION_CONFIG = {
    # Cuánto expandir las barreras (caminos/castillos) para separar campos
    # Valores más altos = mejor separación pero campos más pequeños
    'barrier_expansion': 4,
    
    # Área mínima en píxeles para considerar un campo válido
    # Filtra campos muy pequeños que probablemente son ruido
    'min_field_area': 100,
    
    # Umbral de píxeles de meeple para contar como presente
    'meeple_detection_threshold': 15,
}
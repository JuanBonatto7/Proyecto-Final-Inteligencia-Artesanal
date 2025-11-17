"""
Configuración de colores para el tablero de Carcassonne.
Modifica estos valores según necesites.
"""

# Colores en formato RGB (valores enteros simples)
COLORS = {
    'FIELD': (0, 191, 98),           # Verde (ajustado para imágenes generadas)
    'CASTLE': (255, 145, 76),         # Naranja (ajustado para imágenes generadas)
    'ROAD': (63, 72, 204),            # Azul (mantener original, ajustar si necesario)
    'CHURCH': (237, 28, 36),          # Rojo
    'MEEPLE_1': (163, 73, 164),       # Violeta
    'MEEPLE_2': (0, 0, 0),            # Negro
}

# Tolerancia para la detección de colores
# Aumentada para manejar variaciones en la imagen
COLOR_TOLERANCE = 30

# Tolerancia específica para meeples (más estricta)
MEEPLE_TOLERANCE = 20

# Nombre de jugadores
PLAYER_NAMES = {
    'MEEPLE_1': 'Jugador 1',
    'MEEPLE_2': 'Jugador 2',
}

# Umbral para detectar áreas blancas (fuera del tablero)
# Píxeles con R,G,B > WHITE_THRESHOLD se consideran blancos
# Valores: 0-255 (200 = detecta grises claros, 220 = solo blancos puros)
WHITE_THRESHOLD = 200

# Configuración avanzada para detección de campos
FIELD_DETECTION_CONFIG = {
    # Cuánto expandir las barreras (caminos/castillos) para separar campos
    # Valores más altos = mejor separación pero campos más pequeños
    'barrier_expansion': 2,

    # Área mínima en píxeles para considerar un campo válido
    # Filtra campos muy pequeños que probablemente son ruido
    'min_field_area': 100,

    # Umbral de píxeles de meeple para contar como presente
    'meeple_detection_threshold': 15,
}

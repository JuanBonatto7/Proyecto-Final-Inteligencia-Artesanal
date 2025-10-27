"""
Configuración de colores para el tablero de Carcassonne.
Modifica estos valores según necesites.
"""

# Colores en formato RGB (valores enteros simples)
COLORS = {
    'FIELD': (34, 177, 76),           # Verde
    'CASTLE': (255, 127, 39),         # Naranja (era CHURCH, intercambiado)
    'ROAD': (63, 72, 204),            # Azul
    'CHURCH': (237, 28, 36),          # Rojo (ajustar si es necesario)
    'MEEPLE_1': (163, 73, 164),       # Violeta
    'MEEPLE_2': (0, 0, 0),            # Negro
}

# Tolerancia para la detección de colores
COLOR_TOLERANCE = 30

# Nombre de jugadores
PLAYER_NAMES = {
    'MEEPLE_1': 'Jugador 1',
    'MEEPLE_2': 'Jugador 2',
}
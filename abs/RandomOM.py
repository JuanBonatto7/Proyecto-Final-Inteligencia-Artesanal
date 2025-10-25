import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

@dataclass
class Tile:
    """Representa una loseta del tablero de Carcassonne"""
    nombre: str
    orientacion: int  # 0, 1, 2, 3 (rotación en múltiplos de 90°)
    meeple: Tuple[int, int]  # (jugador, posición)


TILESET = {
    "A": {"borders": {"N": "field", "E": "field", "S": "road", "W": "field"}, "count": 2},
    "B": {"borders": {"N": "field", "E": "field", "S": "field", "W": "field"}, "count": 4},
    "C": {"borders": {"N": "city", "E": "city", "S": "city", "W": "city"}, "count": 1},
    "D": {"borders": {"N": "road", "E": "city", "S": "road", "W": "field"}, "count": 4},
    "E": {"borders": {"N": "city", "E": "field", "S": "field", "W": "field"}, "count": 5},
    "F": {"borders": {"N": "field", "E": "city", "S": "field", "W": "city"}, "count": 2},
    "G": {"borders": {"N": "city", "E": "field", "S": "city", "W": "field"}, "count": 1},
    "H": {"borders": {"N": "field", "E": "city", "S": "field", "W": "city"}, "count": 3},
    "I": {"borders": {"N": "field", "E": "city", "S": "city", "W": "field"}, "count": 2},
    "J": {"borders": {"N": "city", "E": "road", "S": "road", "W": "field"}, "count": 3},
    "K": {"borders": {"N": "road", "E": "city", "S": "field", "W": "road"}, "count": 3},
    "L": {"borders": {"N": "road", "E": "city", "S": "road", "W": "road"}, "count": 3},
    "M": {"borders": {"N": "city", "E": "field", "S": "field", "W": "city"}, "count": 3},
    "N": {"borders": {"N": "city", "E": "field", "S": "field", "W": "city"}, "count": 3},
    "O": {"borders": {"N": "city", "E": "road", "S": "road", "W": "city"}, "count": 2},
    "P": {"borders": {"N": "city", "E": "road", "S": "road", "W": "city"}, "count": 3},
    "Q": {"borders": {"N": "city", "E": "city", "S": "field", "W": "city"}, "count": 1},
    "R": {"borders": {"N": "city", "E": "city", "S": "field", "W": "city"}, "count": 3},
    "S": {"borders": {"N": "city", "E": "city", "S": "road", "W": "city"}, "count": 2},
    "T": {"borders": {"N": "city", "E": "city", "S": "road", "W": "city"}, "count": 1},
    "U": {"borders": {"N": "road", "E": "field", "S": "road", "W": "field"}, "count": 8},
    "V": {"borders": {"N": "field", "E": "field", "S": "road", "W": "road"}, "count": 9},
    "W": {"borders": {"N": "field", "E": "road", "S": "road", "W": "road"}, "count": 4},
    "X": {"borders": {"N": "road", "E": "road", "S": "road", "W": "road"}, "count": 1},
}

TOTAL_LOSETAS = sum(data["count"] for data in TILESET.values())


def rotar_bordes(bordes: Dict[str, str], rotaciones: int) -> Dict[str, str]:
    """
    Rota los bordes de una loseta según su orientación.
    rotaciones = 0 → sin rotar
    rotaciones = 1 → 90° horario
    rotaciones = 2 → 180°
    rotaciones = 3 → 270°
    """
    orden = ["N", "E", "S", "W"]
    rotaciones = rotaciones % 4
    return {
        orden[i]: bordes[orden[(i - rotaciones) % 4]]
        for i in range(4)
    }

# --- Función auxiliar: definir compatibilidad ---
def son_compatibles(a: str, b: str) -> bool:
    """
    Determina si dos bordes son compatibles.
    En Carcassonne, deben ser del mismo tipo (field, road, city).
    """
    return a == b


# --- Obtener vecinos ---
def obtener_vecinos_con_direccion(
    tablero: List[List[Optional["Tile"]]],
    fila: int,
    col: int
) -> List[Tuple[str, "Tile", str]]:
    """
    Obtiene los vecinos existentes de una posición.
    Devuelve tuplas: (mi_direccion, loseta_vecina, direccion_del_vecino_hacia_mi)
    """
    n = len(tablero)
    vecinos = []

    if fila > 0 and tablero[fila - 1][col] is not None:
        vecinos.append(("N", tablero[fila - 1][col], "S"))
    if fila < n - 1 and tablero[fila + 1][col] is not None:
        vecinos.append(("S", tablero[fila + 1][col], "N"))
    if col < n - 1 and tablero[fila][col + 1] is not None:
        vecinos.append(("E", tablero[fila][col + 1], "W"))
    if col > 0 and tablero[fila][col - 1] is not None:
        vecinos.append(("W", tablero[fila][col - 1], "E"))

    return vecinos


# --- Verificar compatibilidad ---
def verificar_compatibilidad(
    nombre_loseta: str,
    orientacion: int,
    tablero: List[List[Optional["Tile"]]],
    fila: int,
    col: int
) -> bool:
    """
    Verifica si una loseta con cierta orientación encaja con todos sus vecinos.
    """
    vecinos = obtener_vecinos_con_direccion(tablero, fila, col)

    # ✅ Permitir la primera ficha del tablero
    if len(vecinos) == 0:
        return True

    bordes_propuesta = rotar_bordes(TILESET[nombre_loseta]["borders"], orientacion)

    for mi_direccion, loseta_vecina, dir_vecino_hacia_mi in vecinos:
        bordes_vecino = rotar_bordes(
            TILESET[loseta_vecina.nombre]["borders"],
            loseta_vecina.orientacion
        )

        if not son_compatibles(
            bordes_propuesta[mi_direccion],
            bordes_vecino[dir_vecino_hacia_mi]
        ):
            return False

    return True


def encontrar_rotacion_valida(nombre_loseta: str,
                              tablero: List[List[Optional[Tile]]], 
                              fila: int, 
                              col: int) -> Optional[int]:
    """
    Busca una rotación válida para una loseta en una posición.
    
    Args:
        nombre_loseta: Tipo de loseta
        tablero: Tablero actual
        fila, col: Posición donde colocar
    
    Returns:
        Rotación válida (0-3) o None si ninguna funciona
    """
    for rot in [0, 1, 2, 3]:
        if verificar_compatibilidad(nombre_loseta, rot, tablero, fila, col):
            return rot
    return None


# =============================================================================
# GENERACIÓN DEL TABLERO
# =============================================================================

def generar_tablero(n: int = 5) -> List[List[Optional[Tile]]]:
    """
    Genera un tablero cuadrado n x n válido respetando el inventario y las reglas del juego.
    
    Args:
        n: Tamaño del tablero (default=5)
        
    Returns:
        List[List[Optional[Tile]]]: Matriz que representa el tablero con las losetas colocadas
    """
    # Inicializar tablero vacío
    tablero = [[None for _ in range(n)] for _ in range(n)]
    centro = n // 2
    
    # Crear inventario de losetas disponibles
    disponibles = {nombre: cantidad["count"] for nombre, cantidad in TILESET.items()}
    
    # Asignar meeples por jugador (3-5 por jugador)
    meeples_totales = {1: random.randint(3, 5), 2: random.randint(3, 5)}
    meeples_usados = {1: 0, 2: 0}
    
    # Colocar loseta inicial en el centro
    nombre_inicial = random.choice(list(disponibles.keys()))
    orientacion_inicial = random.choice([0, 1, 2, 3])
    tablero[centro][centro] = Tile(nombre_inicial, orientacion_inicial, (0, 0))
    disponibles[nombre_inicial] -= 1
    
    # Ordenar posiciones por distancia al centro
    posiciones = []
    for i in range(n):
        for j in range(n):
            if i == centro and j == centro:
                continue
            dist = max(abs(i - centro), abs(j - centro))
            posiciones.append((dist, i, j))
    
    posiciones.sort(key=lambda x: (x[0], random.random()))
    
    # Llenar tablero
    for _, fila, col in posiciones:
        # Verificar si hay vecinos
        vecinos = obtener_vecinos_con_direccion(tablero, fila, col)
        if len(vecinos) == 0:
            continue
        
        # Obtener losetas disponibles
        tipos_disponibles = [t for t, cant in disponibles.items() if cant > 0]
        if not tipos_disponibles:
            break
        
        random.shuffle(tipos_disponibles)
        
        # Intentar colocar una loseta válida
        for tipo in tipos_disponibles:
            rot_valida = encontrar_rotacion_valida(tipo, tablero, fila, col)
            
            if rot_valida is not None:
                # Decidir si colocar meeple (30% probabilidad)
                jugador = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 2])
                
                if jugador > 0 and meeples_usados[jugador] < meeples_totales[jugador]:
                    meeple = (jugador, random.randint(0, 8))
                    meeples_usados[jugador] += 1
                else:
                    meeple = (0, 0)
                
                # Colocar loseta
                tablero[fila][col] = Tile(tipo, rot_valida, meeple)
                disponibles[tipo] -= 1
                break
    
    return tablero



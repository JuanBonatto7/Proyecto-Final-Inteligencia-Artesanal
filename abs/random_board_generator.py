import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, deque

@dataclass
class Tile:
    """Represents a Carcassonne board tile"""
    type: str
    rotation: int
    meeple_info: Optional[Tuple[int, int]]

# =============================================================================
# TILE CONFIGURATION
# =============================================================================

GRID_SIZE = 3
ROTATION_STEP = 90
EDGE_ORDER = ["TOP", "RIGHT", "BOTTOM", "LEFT"]

EDGE_CENTERS = {
    "LEFT":   (1, 0),
    "TOP":    (0, 1),
    "RIGHT":  (1, 2),
    "BOTTOM": (2, 1),
}

FEATURE_CITY = "C"
FEATURE_ROAD = "R"
FEATURE_FIELD = "F"
FEATURE_MONASTERY = "M"
FEATURE_TOWN = "T"

@dataclass
class TileConfig:
    grid: List[str]
    feature_connections: List[List[str]]
    has_pennant: bool
    count: int

TILE_INFO: Dict[str, TileConfig] = {
    "A": TileConfig(grid=["F", "F", "F", "F", "M", "F", "F", "R", "F"], feature_connections=[], has_pennant=False, count=2),
    "B": TileConfig(grid=["F", "F", "F", "F", "M", "F", "F", "F", "F"], feature_connections=[], has_pennant=False, count=4),
    "C": TileConfig(grid=["C", "C", "C", "C", "C", "C", "C", "C", "C"], feature_connections=[["LEFT", "TOP", "RIGHT", "BOTTOM"]], has_pennant=True, count=1),
    "D": TileConfig(grid=["F", "R", "F", "F", "R", "C", "F", "R", "F"], feature_connections=[["TOP", "BOTTOM"]], has_pennant=False, count=4),
    "E": TileConfig(grid=["F", "C", "F", "F", "F", "F", "F", "F", "F"], feature_connections=[], has_pennant=False, count=5),
    "F": TileConfig(grid=["F", "F", "F", "C", "C", "C", "F", "F", "F"], feature_connections=[["LEFT", "RIGHT"]], has_pennant=True, count=2),
    "G": TileConfig(grid=["F", "C", "F", "F", "C", "F", "F", "C", "F"], feature_connections=[["TOP", "BOTTOM"]], has_pennant=False, count=1),
    "H": TileConfig(grid=["F", "F", "F", "C", "F", "C", "F", "F", "F"], feature_connections=[], has_pennant=False, count=3),
    "I": TileConfig(grid=["F", "F", "F", "F", "F", "C", "F", "C", "F"], feature_connections=[], has_pennant=False, count=2),
    "J": TileConfig(grid=["F", "C", "F", "F", "R", "R", "F", "R", "F"], feature_connections=[["RIGHT", "BOTTOM"]], has_pennant=False, count=3),
    "K": TileConfig(grid=["F", "R", "F", "R", "R", "C", "F", "F", "F"], feature_connections=[["LEFT", "TOP"]], has_pennant=False, count=3),
    "L": TileConfig(grid=["F", "R", "F", "R", "T", "C", "F", "R", "F"], feature_connections=[], has_pennant=False, count=3),
    "M": TileConfig(grid=["C", "C", "F", "C", "F", "F", "F", "F", "F"], feature_connections=[["LEFT", "TOP"]], has_pennant=True, count=2),
    "N": TileConfig(grid=["C", "C", "F", "C", "F", "F", "F", "F", "F"], feature_connections=[["LEFT", "TOP"]], has_pennant=False, count=3),
    "O": TileConfig(grid=["C", "C", "F", "C", "R", "R", "F", "R", "F"], feature_connections=[["LEFT", "TOP"], ["RIGHT", "BOTTOM"]], has_pennant=True, count=2),
    "P": TileConfig(grid=["C", "C", "F", "C", "R", "R", "F", "R", "F"], feature_connections=[["LEFT", "TOP"], ["RIGHT", "BOTTOM"]], has_pennant=False, count=3),
    "Q": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "F", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=True, count=1),
    "R": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "F", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=False, count=3),
    "S": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "R", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=True, count=2),
    "T": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "R", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=False, count=1),
    "U": TileConfig(grid=["F", "R", "F", "F", "R", "F", "F", "R", "F"], feature_connections=[["TOP", "BOTTOM"]], has_pennant=False, count=8),
    "V": TileConfig(grid=["F", "F", "F", "R", "R", "F", "F", "R", "F"], feature_connections=[["LEFT", "BOTTOM"]], has_pennant=False, count=9),
    "W": TileConfig(grid=["F", "F", "F", "R", "T", "R", "F", "R", "F"], feature_connections=[], has_pennant=False, count=4),
    "X": TileConfig(grid=["F", "R", "F", "R", "T", "R", "F", "R", "F"], feature_connections=[], has_pennant=False, count=1),
}

# =============================================================================
# GRID ROTATION UTILITIES
# =============================================================================

class GridRotator:
    @staticmethod
    def _rotate_grid_steps(flat_grid: List[str], steps: int) -> List[str]:
        """Rota una grilla 3x3 en pasos de 90° sentido horario."""
        steps = steps % 4
        if steps == 0:
            return flat_grid[:]
        if steps == 1:
            order = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        elif steps == 2:
            order = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        else:
            order = [2, 5, 8, 1, 4, 7, 0, 3, 6]
        return [flat_grid[i] for i in order]

    @staticmethod
    def rotate_grid(flat_grid: List[str], degrees: int) -> List[str]:
        """Rota una grilla dados los grados (0, 90, 180, 270)."""
        steps = ((degrees % 360) // ROTATION_STEP) % 4
        return GridRotator._rotate_grid_steps(flat_grid, steps)

    @staticmethod
    def rotate_edge(edge: str, steps: int) -> str:
        """Rota el nombre de un borde (TOP, RIGHT, etc.) n pasos."""
        index = EDGE_ORDER.index(edge)
        new_index = (index + (steps % 4)) % len(EDGE_ORDER)
        return EDGE_ORDER[new_index]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tile_edges(tile_type: str, rotation: int) -> Dict[str, str]:
    """Obtiene los bordes de un tile después de aplicar la rotación."""
    config = TILE_INFO[tile_type]
    rotated_grid = GridRotator.rotate_grid(config.grid, rotation)
    edges = {}
    for edge, (lr, lc) in EDGE_CENTERS.items():
        val = rotated_grid[lr * GRID_SIZE + lc]
        if val == FEATURE_CITY:
            edges[edge] = FEATURE_CITY
        elif val == FEATURE_ROAD:
            edges[edge] = FEATURE_ROAD
        else:
            edges[edge] = FEATURE_FIELD
    return edges

def tiles_match(tile1_type: str, tile1_rot: int, edge1: str,
                tile2_type: str, tile2_rot: int, edge2: str) -> bool:
    """Verifica si dos tiles coinciden en sus bordes adyacentes."""
    edges1 = get_tile_edges(tile1_type, tile1_rot)
    edges2 = get_tile_edges(tile2_type, tile2_rot)
    return edges1[edge1] == edges2[edge2]

def can_place_tile(board: List[List[Optional[Tile]]], row: int, col: int,
                   tile_type: str, rotation: int) -> bool:
    """Verifica si un tile puede colocarse en una posición dada."""
    n = len(board)
    
    # Verificar borde superior
    if row > 0 and board[row - 1][col] is not None:
        neighbor = board[row - 1][col]
        if not tiles_match(tile_type, rotation, "TOP",
                          neighbor.type, neighbor.rotation, "BOTTOM"):
            return False
    
    # Verificar borde inferior
    if row < n - 1 and board[row + 1][col] is not None:
        neighbor = board[row + 1][col]
        if not tiles_match(tile_type, rotation, "BOTTOM",
                          neighbor.type, neighbor.rotation, "TOP"):
            return False
    
    # Verificar borde izquierdo
    if col > 0 and board[row][col - 1] is not None:
        neighbor = board[row][col - 1]
        if not tiles_match(tile_type, rotation, "LEFT",
                          neighbor.type, neighbor.rotation, "RIGHT"):
            return False
    
    # Verificar borde derecho
    if col < n - 1 and board[row][col + 1] is not None:
        neighbor = board[row][col + 1]
        if not tiles_match(tile_type, rotation, "RIGHT",
                          neighbor.type, neighbor.rotation, "LEFT"):
            return False
    
    return True

def is_structure_closed(board, row, col, pos):
    """
    Determina si la estructura (ciudad o camino) a la que pertenece `pos` está cerrada.
    Considera conexiones internas dentro de la loseta (mismo tipo de feature).
    Incluye debug detallado.
    """
    tile = board[row][col]
    if tile is None:
        print(f"[DEBUG] ({row},{col}) sin loseta.")
        return False

    tile_info = TILE_INFO[tile.type]
    rotated_grid = GridRotator.rotate_grid(tile_info.grid, tile.rotation)
    start_feature = rotated_grid[pos - 1]

    if start_feature not in [FEATURE_CITY, FEATURE_ROAD]:
        print(f"[DEBUG] ({row},{col}) pos={pos}: feature={start_feature}, no es ciudad ni camino.")
        return False

    # Funciones auxiliares internas (sin dependencias externas)
    def neighbors_of_position(p):
        mapping = {
            1: [2, 4], 2: [1, 3, 5], 3: [2, 6],
            4: [1, 5, 7], 5: [2, 4, 6, 8], 6: [3, 5, 9],
            7: [4, 8], 8: [5, 7, 9], 9: [6, 8]
        }
        return mapping.get(p, [])

    def opposite(p):
        # Puntos cardinales del grid 3x3
        opposites = {1: 9, 2: 8, 3: 7, 4: 6, 6: 4, 7: 3, 8: 2, 9: 1, 5: 5}
        return opposites[p]

    def delta(p):
        # Traduce posición (1–9) a desplazamiento (dr, dc)
        mapping = {
            1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
            4: (0, -1), 5: (0, 0), 6: (0, 1),
            7: (1, -1), 8: (1, 0), 9: (1, 1),
        }
        return mapping[p]

    visited = set()
    queue = [(row, col, pos)]
    closed = True

    while queue:
        r, c, p = queue.pop(0)
        if (r, c, p) in visited:
            continue
        visited.add((r, c, p))

        t = board[r][c]
        if t is None:
            closed = False
            continue

        t_info = TILE_INFO[t.type]
        t_grid = GridRotator.rotate_grid(t_info.grid, t.rotation)
        feature = t_grid[p - 1]

        # Si el feature no coincide, saltar
        if feature != start_feature:
            continue

        # 1️⃣ Conexiones internas (dentro del mismo tile)
        for np in neighbors_of_position(p):
            if t_grid[np - 1] == feature and (r, c, np) not in visited:
                queue.append((r, c, np))

        # 2️⃣ Conexiones externas (solo en bordes cardinales)
        if p in [2, 4, 6, 8]:  # arriba, izquierda, derecha, abajo
            dr, dc = delta(p)
            nr, nc = r + dr, c + dc

            # Fuera del tablero = abierto
            if not (0 <= nr < len(board) and 0 <= nc < len(board)):
                print(f"[DEBUG] ({r},{c}) p={p} → borde sin loseta → ABIERTO ❌")
                closed = False
                continue

            neighbor_tile = board[nr][nc]
            if neighbor_tile is None:
                print(f"[DEBUG] ({r},{c}) p={p} → vecino vacío ({nr},{nc}) → ABIERTO ❌")
                closed = False
                continue

            neighbor_info = TILE_INFO[neighbor_tile.type]
            neighbor_grid = GridRotator.rotate_grid(neighbor_info.grid, neighbor_tile.rotation)
            opp = opposite(p)

            if neighbor_grid[opp - 1] != feature:
                print(f"[DEBUG] ({r},{c}) p={p} → vecino ({nr},{nc}) no coincide → ABIERTO ❌")
                closed = False
                continue

            queue.append((nr, nc, opp))

    if closed:
        print(f"[DEBUG] ({row},{col}) pos={pos} → ESTRUCTURA CERRADA ✅")
    else:
        print(f"[DEBUG] ({row},{col}) pos={pos} → estructura ABIERTA ❌")

    return closed


def get_valid_meeple_positions(board, row, col, tile_type, rotation):
    """
    Devuelve una lista de posiciones (1–9) válidas para colocar meeples en la loseta.
    Reglas:
    - Campos (F): válidos si tocan al menos un tile vacío o no están completamente rodeados
    - Monasterios (M): siempre válidos
    - Ciudades (C) y Caminos (R): solo si están abiertos (no cerrados)
    """
    valid_positions = []
    config = TILE_INFO[tile_type]
    rotated_grid = GridRotator.rotate_grid(config.grid, rotation)
    n = len(board)

    for pos in range(1, 10):
        feature = rotated_grid[pos - 1]

        # Monasterios siempre válidos
        if feature == FEATURE_MONASTERY:
            valid_positions.append(pos)
            continue

        # Campos válidos si no están completamente rodeados
        if feature == FEATURE_FIELD:
            open_to_air = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < n and 0 <= nc < n) or board[nr][nc] is None:
                    open_to_air = True
                    break
            if open_to_air:
                valid_positions.append(pos)
            continue

        # Ciudades y caminos: solo si no están cerrados
        if feature in [FEATURE_CITY, FEATURE_ROAD]:
            temp_board = [row_list.copy() for row_list in board]
            temp_board[row][col] = Tile(type=tile_type, rotation=rotation, meeple_info=None)
            if not is_structure_closed(temp_board, row, col, pos):
                valid_positions.append(pos)

    return valid_positions



# =============================================================================
# BOARD GENERATION
# =============================================================================

def get_neighbors_with_direction(board: List[List[Optional[Tile]]], row: int, col: int) -> List[Tuple[int, int, str]]:
    """Retorna vecinos existentes con su dirección relativa."""
    neighbors = []
    n = len(board)
    
    if row > 0 and board[row - 1][col] is not None:
        neighbors.append((row - 1, col, "TOP"))
    if row < n - 1 and board[row + 1][col] is not None:
        neighbors.append((row + 1, col, "BOTTOM"))
    if col > 0 and board[row][col - 1] is not None:
        neighbors.append((row, col - 1, "LEFT"))
    if col < n - 1 and board[row][col + 1] is not None:
        neighbors.append((row, col + 1, "RIGHT"))
    
    return neighbors

def find_valid_rotation(tile_type: str, board: List[List[Optional[Tile]]], row: int, col: int) -> Optional[int]:
    """Busca una rotación válida para el tile en la posición dada."""
    rotations = [0, 90, 180, 270]
    random.shuffle(rotations)
    
    for rotation in rotations:
        if can_place_tile(board, row, col, tile_type, rotation):
            return rotation
    
    return None

def generate_board(n: int = 5) -> List[List[Optional[Tile]]]:
    """
    Genera un tablero cuadrado n x n válido de Carcassonne (versión debug).
    
    Imprime paso a paso:
      - Tiles colocados
      - Meeples colocados
      - Si la estructura está cerrada o no
      - Resumen final
    """

    print("\n🔧 [DEBUG] Generando tablero...\n")

    # Inicializar tablero vacío n x n
    grid: List[List[Optional[Tile]]] = [[None for _ in range(n)] for _ in range(n)]
    center = n // 2
    
    # Inventario de tiles disponibles
    remaining_counts: Dict[str, int] = {tile_type: config.count for tile_type, config in TILE_INFO.items()}

    # ========================================================================
    # ETAPA A: GENERAR TABLERO COMPLETO (SIN MEEPLES)
    # ========================================================================
    print("📦 [DEBUG] Etapa A: Generando tablero sin meeples...\n")

    # Paso 1: Colocar tile inicial en el centro
    initial_type = random.choice([t for t, c in remaining_counts.items() if c > 0])
    initial_rotation = random.choice([0, 90, 180, 270])
    grid[center][center] = Tile(type=initial_type, rotation=initial_rotation, meeple_info=None)
    remaining_counts[initial_type] -= 1

    print(f"🧩 Tile inicial ({center},{center}) = {initial_type} rot={initial_rotation}")

    # Paso 2: Generar orden de llenado (anillos concéntricos desde el centro)
    positions: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == center and j == center:
                continue
            distance = max(abs(i - center), abs(j - center))
            positions.append((distance, i, j))
    positions.sort(key=lambda x: (x[0], random.random()))

    # Paso 3: Colocar tiles sin meeples
    for _, row, col in positions:
        vecinos = get_neighbors_with_direction(grid, row, col)
        if len(vecinos) == 0:
            continue

        available_types = [t for t, cnt in remaining_counts.items() if cnt > 0]
        if not available_types:
            break

        random.shuffle(available_types)
        
        for tile_type in available_types:
            valid_rotation = find_valid_rotation(tile_type, grid, row, col)
            if valid_rotation is not None:
                grid[row][col] = Tile(type=tile_type, rotation=valid_rotation, meeple_info=None)
                remaining_counts[tile_type] -= 1
                print(f"  ✔ Tile ({row},{col}) = {tile_type} rot={valid_rotation}")
                break

    # ========================================================================
    # ETAPA B: COLOCAR MEEPLES EN EL TABLERO FINAL
    # ========================================================================
    print("\n🎯 [DEBUG] Etapa B: Colocando meeples...\n")
    
    total_meeples: Dict[int, int] = {1: random.randint(3, 5), 2: random.randint(3, 5)}
    used_meeples: Dict[int, int] = {1: 0, 2: 0}
    all_valid_positions: Dict[int, List[Tuple[int, int, int]]] = {1: [], 2: []}
    
    for i in range(n):
        for j in range(n):
            tile = grid[i][j]
            if tile is None:
                continue
            
            valid_positions = get_valid_meeple_positions(grid, i, j, tile.type, tile.rotation)
            for pos in valid_positions:
                all_valid_positions[1].append((i, j, pos))
                all_valid_positions[2].append((i, j, pos))
    
    # Colocar meeples
    for player in [1, 2]:
        random.shuffle(all_valid_positions[player])
        placed = 0
        print(f"🎲 Jugador {player} (debe colocar {total_meeples[player]} meeples):")

        for i, j, pos in all_valid_positions[player]:
            if placed >= total_meeples[player]:
                break
            
            tile = grid[i][j]
            if tile.meeple_info is not None:
                continue

            # DEBUG: Verificar si la estructura está cerrada antes de poner el meeple
            cerrado = is_structure_closed(grid, i, j, pos)
            config = TILE_INFO[tile.type]
            rotated = GridRotator.rotate_grid(config.grid, tile.rotation)
            feature = rotated[pos - 1]

            estado = "❌ CERRADA" if cerrado else "✅ ABIERTA"
            print(f"   -> ({i},{j}) tile={tile.type}, rot={tile.rotation}, pos={pos}, "
                  f"feature={feature}, {estado}")

            if cerrado:
                continue  # No colocar si está cerrada

            grid[i][j] = Tile(type=tile.type, rotation=tile.rotation, meeple_info=(player, pos))
            placed += 1
            used_meeples[player] += 1

            # Eliminar esa posición del otro jugador
            other_player = 2 if player == 1 else 1
            all_valid_positions[other_player] = [
                (r, c, p) for r, c, p in all_valid_positions[other_player] 
                if not (r == i and c == j)
            ]

        print(f"   → Meeples colocados: {placed}/{total_meeples[player]}\n")

    # ========================================================================
    # ETAPA C: VERIFICAR MEEPLES MAL COLOCADOS
    # ========================================================================
    print("\n🔍 [DEBUG] Verificando meeples colocados en estructuras cerradas...\n")
    errores = 0
    for i in range(n):
        for j in range(n):
            tile = grid[i][j]
            if tile is None or tile.meeple_info is None:
                continue
            player, pos = tile.meeple_info
            if is_structure_closed(grid, i, j, pos):
                print(f"🚫 ERROR: Meeple jugador {player} en ({i},{j}) → estructura cerrada")
                errores += 1

    if errores == 0:
        print("✅ Todos los meeples están en estructuras abiertas.")
    else:
        print(f"⚠ Se detectaron {errores} meeples en estructuras cerradas.\n")

    # ========================================================================
    # ETAPA D: RESUMEN FINAL
    # ========================================================================
    print("\n📊 Resumen final:")
    for player in [1, 2]:
        print(f"  Jugador {player}: {used_meeples[player]} meeples (de {total_meeples[player]})")
    print("✅ Tablero generado correctamente.\n")

    return grid



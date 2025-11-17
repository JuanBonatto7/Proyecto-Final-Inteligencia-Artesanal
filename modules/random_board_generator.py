"""Generador de tableros aleatorios de Carcassonne para pruebas."""
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, deque
from origin_matrix import Board, Tile


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
    Determina si la estructura a la que pertenece `pos` está cerrada.
    - Para caminos y ciudades: usa BFS.
    - Para monasterios: está cerrada si todas las losetas alrededor están ocupadas.
    """
    tile = board[row][col]
    if tile is None:
        return False

    tile_info = TILE_INFO[tile.type]
    rotated_grid = GridRotator.rotate_grid(tile_info.grid, tile.rotation)
    start_feature = rotated_grid[pos - 1]

    #Monasterios
    if start_feature == FEATURE_MONASTERY or (pos == 5 and start_feature == FEATURE_FIELD):
        # El monasterio está cerrado si las 8 casillas alrededor están ocupadas
        n = len(board)
        deltas = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < n and 0 <= nc < n) or board[nr][nc] is None:
                return False
        return True

    #Ciudades y Caminos
    if start_feature not in [FEATURE_CITY, FEATURE_ROAD]:
        return False

    def neighbors_of_position(p):
        mapping = {
            1: [2, 4], 2: [1, 3, 5], 3: [2, 6],
            4: [1, 5, 7], 5: [2, 4, 6, 8], 6: [3, 5, 9],
            7: [4, 8], 8: [5, 7, 9], 9: [6, 8]
        }
        return mapping.get(p, [])

    def opposite(p):
        opposites = {1: 9, 2: 8, 3: 7, 4: 6, 6: 4, 7: 3, 8: 2, 9: 1, 5: 5}
        return opposites[p]

    def delta(p):
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

        if feature != start_feature:
            continue

        # Conexiones internas
        for np in neighbors_of_position(p):
            if t_grid[np - 1] == feature and (r, c, np) not in visited:
                queue.append((r, c, np))

        # Conexiones externas
        if p in [2, 4, 6, 8]:
            dr, dc = delta(p)
            nr, nc = r + dr, c + dc

            if not (0 <= nr < len(board) and 0 <= nc < len(board)):
                closed = False
                continue

            neighbor_tile = board[nr][nc]
            if neighbor_tile is None:
                closed = False
                continue

            neighbor_info = TILE_INFO[neighbor_tile.type]
            neighbor_grid = GridRotator.rotate_grid(neighbor_info.grid, neighbor_tile.rotation)
            opp = opposite(p)

            if neighbor_grid[opp - 1] != feature:
                closed = False
                continue

            queue.append((nr, nc, opp))

    return closed



def get_valid_meeple_positions(board, row, col, tile_type, rotation):
    """
    Devuelve una lista de posiciones (1–9) válidas para colocar meeples en la loseta.
    Reglas:
    - Campos (F): válidos si tocan al menos un tile vacío o no están completamente rodeados
    - Monasterios (M), Ciudades (C) y Caminos (R): solo si están abiertos (no cerrados)
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


#Generacion Random del tablero:

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

def generate_board(n) -> List[List[Optional[Tile]]]:
    """
    Genera un tablero cuadrado n x n válido de Carcassonne.
    """
    grid: List[List[Optional[Tile]]] = [[None for _ in range(n)] for _ in range(n)]
    center = n // 2

    remaining_counts: Dict[str, int] = {tile_type: config.count for tile_type, config in TILE_INFO.items()}

    initial_type = random.choice([t for t, c in remaining_counts.items() if c > 0])
    initial_rotation = random.choice([0, 90, 180, 270])
    grid[center][center] = Tile(type=initial_type, rotation=initial_rotation, meeple_info=None)
    remaining_counts[initial_type] -= 1

    positions: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == center and j == center:
                continue
            distance = max(abs(i - center), abs(j - center))
            positions.append((distance, i, j))
    positions.sort(key=lambda x: (x[0], random.random()))

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
                break

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

    for player in [1, 2]:
        random.shuffle(all_valid_positions[player])
        placed = 0

        for i, j, pos in all_valid_positions[player]:
            if placed >= total_meeples[player]:
                break

            tile = grid[i][j]
            if tile.meeple_info is not None:
                continue

            cerrado = is_structure_closed(grid, i, j, pos)
            if cerrado:
                continue

            grid[i][j] = Tile(type=tile.type, rotation=tile.rotation, meeple_info=(player, pos))
            placed += 1
            used_meeples[player] += 1

            other_player = 2 if player == 1 else 1
            all_valid_positions[other_player] = [
                (r, c, p) for r, c, p in all_valid_positions[other_player]
                if not (r == i and c == j)
            ]

    return grid



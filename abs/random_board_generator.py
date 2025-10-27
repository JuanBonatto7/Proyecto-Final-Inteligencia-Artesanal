import random
from typing import Tuple, Dict, List, Optional

from origin_matrix import Tile

__all__ = ["generate_board"]

# Set de losetas: definición de bordes y cantidad disponible
TILESET: Dict[str, Dict] = {
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

TOTAL_TILES = sum(info["count"] for info in TILESET.values())

def rotate_borders(borders: Dict[str, str], rotation_degrees: int) -> Dict[str, str]:
    """
    Rota los bordes de una loseta según su orientación en grados (0, 90, 180, 270).
    """
    order = ["N", "E", "S", "W"]
    steps = ((rotation_degrees % 360) // 90) % 4
    return {order[i]: borders[order[(i - steps) % 4]] for i in range(4)}

def are_compatible(a: str, b: str) -> bool:
    """Bordes compatibles si son del mismo tipo (field, road, city)."""
    return a == b

def get_neighbors_with_direction(
    grid: List[List[Optional[Tile]]],
    row: int,
    col: int
) -> List[Tuple[str, Tile, str]]:
    """Devuelve tuplas: (mi_direccion, loseta_vecina, direccion_del_vecino_hacia_mi)."""
    n = len(grid)
    neighbors: List[Tuple[str, Tile, str]] = []
    if row > 0 and grid[row - 1][col] is not None:
        neighbors.append(("N", grid[row - 1][col], "S"))
    if row < n - 1 and grid[row + 1][col] is not None:
        neighbors.append(("S", grid[row + 1][col], "N"))
    if col < n - 1 and grid[row][col + 1] is not None:
        neighbors.append(("E", grid[row][col + 1], "W"))
    if col > 0 and grid[row][col - 1] is not None:
        neighbors.append(("W", grid[row][col - 1], "E"))
    return neighbors

def fits_with_neighbors(
    tile_name: str,
    rotation_degrees: int,
    grid: List[List[Optional[Tile]]],
    row: int,
    col: int
) -> bool:
    """Verifica si una loseta con cierta orientación encaja con todos sus vecinos."""
    neighbors = get_neighbors_with_direction(grid, row, col)
    if len(neighbors) == 0:
        return True

    borders_proposal = rotate_borders(TILESET[tile_name]["borders"], rotation_degrees)

    for my_dir, neighbor_tile, neighbor_dir_to_me in neighbors:
        neighbor_borders = rotate_borders(
            TILESET[neighbor_tile.type]["borders"],
            neighbor_tile.rotation
        )
        if not are_compatible(borders_proposal[my_dir], neighbor_borders[neighbor_dir_to_me]):
            return False
    return True

def find_valid_rotation(
    tile_name: str,
    grid: List[List[Optional[Tile]]],
    row: int,
    col: int
) -> Optional[int]:
    """Devuelve rotación válida en grados (0, 90, 180, 270) o None."""
    for rot in (0, 90, 180, 270):
        if fits_with_neighbors(tile_name, rot, grid, row, col):
            return rot
    return None

def generate_board(n: int = 5) -> List[List[Optional[Tile]]]:
    """
    Genera un tablero cuadrado n x n válido respetando el inventario y las reglas del juego.
    Meeple: None si no hay, o (jugador, pos 1..9).
    """
    grid: List[List[Optional[Tile]]] = [[None for _ in range(n)] for _ in range(n)]
    center = n // 2

    remaining_counts: Dict[str, int] = {name: info["count"] for name, info in TILESET.items()}
    total_meeples: Dict[int, int] = {1: random.randint(3, 5), 2: random.randint(3, 5)}
    used_meeples: Dict[int, int] = {1: 0, 2: 0}

    # Loseta inicial
    initial_type = random.choice(list(remaining_counts.keys()))
    initial_rotation = random.choice([0, 90, 180, 270])
    grid[center][center] = Tile(initial_type, initial_rotation, None)
    remaining_counts[initial_type] -= 1

    # Orden de llenado: anillos concéntricos desde el centro
    positions: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == center and j == center:
                continue
            distance = max(abs(i - center), abs(j - center))
            positions.append((distance, i, j))
    positions.sort(key=lambda x: (x[0], random.random()))

    for _, row, col in positions:
        neighbors = get_neighbors_with_direction(grid, row, col)
        if len(neighbors) == 0:
            continue

        available_types = [t for t, cnt in remaining_counts.items() if cnt > 0]
        if not available_types:
            break
        random.shuffle(available_types)

        for tile_type in available_types:
            valid_rotation = find_valid_rotation(tile_type, grid, row, col)
            if valid_rotation is not None:
                # 30% probabilidad de meeple si quedan
                player = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 2])
                if player > 0 and used_meeples[player] < total_meeples[player]:
                    meeple_info = (player, random.randint(1, 9))  # 1..9
                    used_meeples[player] += 1
                else:
                    meeple_info = None

                grid[row][col] = Tile(tile_type, valid_rotation, meeple_info)
                remaining_counts[tile_type] -= 1
                break

    return grid



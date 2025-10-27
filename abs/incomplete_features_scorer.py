"""
Sistema de puntuacion final para Carcassonne.
Calcula puntos solo de características incompletas (ciudades, caminos, monasterios).
"""

from dataclasses import dataclass
from origin_matrix import Board, Tile
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, deque

__all__ = ["GameScorer", "set_debug"]

DEBUG = False
DEBUG_SHOW_TILE_POSITIONS = False

def set_debug(value: bool, show_positions: bool = False):
    """Activa/desactiva logs de depuración y si muestra posiciones de tiles."""
    global DEBUG, DEBUG_SHOW_TILE_POSITIONS
    DEBUG = bool(value)
    DEBUG_SHOW_TILE_POSITIONS = bool(show_positions)

def dbg(msg: str):
    if DEBUG:
        print(msg)

def dbg_block(lines: List[str]):
    if DEBUG:
        print("\n".join(lines))

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

SCOREABLE_FEATURES = {FEATURE_CITY, FEATURE_ROAD}

@dataclass
class TileConfig:
    grid: List[str]
    feature_connections: List[List[str]]
    has_pennant: bool

TILE_INFO: Dict[str, TileConfig] = {
    "A": TileConfig(grid=["F", "F", "F", "F", "M", "F", "F", "R", "F"], feature_connections=[], has_pennant=False),
    "B": TileConfig(grid=["F", "F", "F", "F", "M", "F", "F", "F", "F"], feature_connections=[], has_pennant=False),
    "C": TileConfig(grid=["C", "C", "C", "C", "C", "C", "C", "C", "C"], feature_connections=[["LEFT", "TOP", "RIGHT", "BOTTOM"]], has_pennant=True),
    "D": TileConfig(grid=["F", "R", "F", "F", "R", "C", "F", "R", "F"], feature_connections=[["TOP", "BOTTOM"]], has_pennant=False),
    "E": TileConfig(grid=["F", "C", "F", "F", "F", "F", "F", "F", "F"], feature_connections=[], has_pennant=False),
    "F": TileConfig(grid=["F", "F", "F", "C", "C", "C", "F", "F", "F"], feature_connections=[["LEFT", "RIGHT"]], has_pennant=True),
    "G": TileConfig(grid=["F", "C", "F", "F", "C", "F", "F", "C", "F"], feature_connections=[["TOP", "BOTTOM"]], has_pennant=False),
    "H": TileConfig(grid=["F", "F", "F", "C", "F", "C", "F", "F", "F"], feature_connections=[], has_pennant=False),
    "I": TileConfig(grid=["F", "F", "F", "F", "F", "C", "F", "C", "F"], feature_connections=[], has_pennant=False),
    "J": TileConfig(grid=["F", "C", "F", "F", "R", "R", "F", "R", "F"], feature_connections=[["RIGHT", "BOTTOM"]], has_pennant=False),
    "K": TileConfig(grid=["F", "R", "F", "R", "R", "C", "F", "F", "F"], feature_connections=[["LEFT", "TOP"]], has_pennant=False),
    "L": TileConfig(grid=["F", "R", "F", "R", "T", "C", "F", "R", "F"], feature_connections=[], has_pennant=False),
    "M": TileConfig(grid=["C", "C", "F", "C", "F", "F", "F", "F", "F"], feature_connections=[["LEFT", "TOP"]], has_pennant=True),
    "N": TileConfig(grid=["C", "C", "F", "C", "F", "F", "F", "F", "F"], feature_connections=[["LEFT", "TOP"]], has_pennant=False),
    "O": TileConfig(grid=["C", "C", "F", "C", "R", "R", "F", "R", "F"], feature_connections=[["LEFT", "TOP"], ["RIGHT", "BOTTOM"]], has_pennant=True),
    "P": TileConfig(grid=["C", "C", "F", "C", "R", "R", "F", "R", "F"], feature_connections=[["LEFT", "TOP"], ["RIGHT", "BOTTOM"]], has_pennant=False),
    "Q": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "F", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=True),
    "R": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "F", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=False),
    "S": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "R", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=True),
    "T": TileConfig(grid=["C", "C", "C", "C", "C", "C", "F", "R", "F"], feature_connections=[["LEFT", "TOP", "RIGHT"]], has_pennant=False),
    "U": TileConfig(grid=["F", "R", "F", "F", "R", "F", "F", "R", "F"], feature_connections=[["TOP", "BOTTOM"]], has_pennant=False),
    "V": TileConfig(grid=["F", "F", "F", "R", "R", "F", "F", "R", "F"], feature_connections=[["LEFT", "BOTTOM"]], has_pennant=False),
    "W": TileConfig(grid=["F", "F", "F", "R", "T", "R", "F", "R", "F"], feature_connections=[], has_pennant=False),
    "X": TileConfig(grid=["F", "R", "F", "R", "T", "R", "F", "R", "F"], feature_connections=[], has_pennant=False),
}

class GridRotator:
    @staticmethod
    def _rotate_grid_steps(flat_grid: List[str], steps: int) -> List[str]:
        """
        Rota una grilla 3x3 en pasos de 90° sentido horario.
        Mapeo de índices (0=0°, 1=90°, 2=180°, 3=270°):
            0 1 2       6 3 0       8 7 6       2 5 8
            3 4 5  -->  7 4 1  -->  5 4 3  -->  1 4 7
            6 7 8       8 5 2       2 1 0       0 3 6
        """
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
        steps = ((degrees % 360) // ROTATION_STEP) % 4
        return GridRotator._rotate_grid_steps(flat_grid, steps)

    @staticmethod
    def rotate_edge(edge: str, steps: int) -> str:
        index = EDGE_ORDER.index(edge)
        new_index = (index + (steps % 4)) % len(EDGE_ORDER)
        return EDGE_ORDER[new_index]

class CellInfo:
    def __init__(self, feature: str, tile_pos: Tuple[int, int],
                 local_pos: Tuple[int, int], letter: str):
        self.feature = feature
        self.tile_pos = tile_pos
        self.local_pos = local_pos
        self.letter = letter
        self.meeple: Optional[Dict] = None

class FeatureComponent:
    def __init__(self, feature: str, cells: Set[Tuple[int, int]]):
        self.feature = feature
        self.cells = cells

class GameScorer:
    def __init__(self, board: Board):
        self.board = board
        self.height = len(board.board)
        self.width = max(len(row) for row in board.board)
        self._global_cells: Dict[Tuple[int, int], CellInfo] = {}
        self._tile_has_pennant: Dict[Tuple[int, int], bool] = {}
        self._adjacency: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        self._cell_to_component: Dict[Tuple[int, int], Dict] = {}
        self._tile_edges: Dict[Tuple[int, int], Dict[str, str]] = {}
        self._build_global_grid()
        self._build_adjacency_graph()

    def _build_global_grid(self):
        for tile_row in range(self.height):
            for tile_col in range(len(self.board.board[tile_row])):
                tile = self.board.board[tile_row][tile_col]
                if tile:
                    self._process_tile(tile, tile_row, tile_col)

    def _process_tile(self, tile: Tile, tile_row: int, tile_col: int):
        config = TILE_INFO[tile.type]
        rotated_grid = GridRotator.rotate_grid(config.grid, tile.rotation)
        self._tile_edges[(tile_row, tile_col)] = self._derive_edges_from_grid(rotated_grid)

        for lr in range(GRID_SIZE):
            for lc in range(GRID_SIZE):
                self._create_global_cell(tile, tile_row, tile_col, lr, lc, rotated_grid)

        self._tile_has_pennant[(tile_row, tile_col)] = config.has_pennant

        # Meeple: NO rotar la posición; la posición 1..9 es relativa al tile ya rotado.
        if tile.meeple_info and tile.meeple_info[0] > 0:
            self._assign_meeple(tile, tile_row, tile_col)

    def _create_global_cell(self, tile: Tile, tile_row: int, tile_col: int,
                            local_row: int, local_col: int, rotated_grid: List[str]):
        idx = local_row * GRID_SIZE + local_col
        feature = rotated_grid[idx]
        g_row = tile_row * GRID_SIZE + local_row
        g_col = tile_col * GRID_SIZE + local_col
        self._global_cells[(g_row, g_col)] = CellInfo(
            feature=feature,
            tile_pos=(tile_row, tile_col),
            local_pos=(local_row, local_col),
            letter=tile.type,
        )

    def _assign_meeple(self, tile: Tile, tile_row: int, tile_col: int):
        """NO rotar la posición; la posición 1..9 es relativa al tile ya rotado."""
        player, position_1_to_9 = tile.meeple_info
        index0 = int(position_1_to_9) - 1  # 0..8
        lr, lc = divmod(index0, GRID_SIZE)
        g_row = tile_row * GRID_SIZE + lr
        g_col = tile_col * GRID_SIZE + lc
        cell = self._global_cells.get((g_row, g_col))
        if cell:
            cell.meeple = {
                "player": int(player),
                "original_position": position_1_to_9,
                "tile_position": (tile_row, tile_col),
            }

    def _build_adjacency_graph(self):
        self._add_intra_tile_adjacencies()
        self._add_internal_connections()
        self._add_border_connections()

    def _add_intra_tile_adjacencies(self):
        """Conecta celdas ortogonales vecinas dentro del mismo tile (solo C/R)."""
        for (row, col), cell in self._global_cells.items():
            if cell.feature not in SCOREABLE_FEATURES:
                continue
            tile_pos = cell.tile_pos
            for nrow, ncol in self._get_orthogonal_neighbors(row, col):
                ncell = self._global_cells.get((nrow, ncol))
                if not ncell or ncell.tile_pos != tile_pos:
                    continue
                if ncell.feature == cell.feature and ncell.feature in SCOREABLE_FEATURES:
                    self._adjacency[(row, col)].add((nrow, ncol))

    def _add_internal_connections(self):
        """Conexiones internas declaradas por tile (ej: unir bordes de una misma ciudad)."""
        for tr in range(self.height):
            for tc in range(len(self.board.board[tr])):
                tile = self.board.board[tr][tc]
                if tile:
                    self._process_tile_connections(tile, tr, tc)

    def _process_tile_connections(self, tile: Tile, tile_row: int, tile_col: int):
        config = TILE_INFO[tile.type]
        steps = (tile.rotation // ROTATION_STEP) % 4
        for edge_group in config.feature_connections:
            rotated_edges = [GridRotator.rotate_edge(edge, steps) for edge in edge_group]
            self._connect_edge_cells(tile_row, tile_col, rotated_edges)

    def _connect_edge_cells(self, tile_row: int, tile_col: int, edges: List[str]):
        cells: List[Tuple[int, int]] = []
        for edge in edges:
            lr, lc = EDGE_CENTERS[edge]
            cells.append((tile_row * GRID_SIZE + lr, tile_col * GRID_SIZE + lc))
        self._create_clique(cells)

    def _add_border_connections(self):
        """Conecta centros de borde entre tiles si ambos lados coinciden (C o R)."""
        for r in range(self.height):
            for c in range(len(self.board.board[r])):
                tile = self.board.board[r][c]
                if not tile:
                    continue
                if r - 1 >= 0 and c < len(self.board.board[r - 1]) and self.board.board[r - 1][c]:
                    self._maybe_connect_border((r, c), "TOP", (r - 1, c), "BOTTOM")
                if r + 1 < self.height and c < len(self.board.board[r + 1]) and self.board.board[r + 1][c]:
                    self._maybe_connect_border((r, c), "BOTTOM", (r + 1, c), "TOP")
                if c - 1 >= 0 and self.board.board[r][c - 1]:
                    self._maybe_connect_border((r, c), "LEFT", (r, c - 1), "RIGHT")
                if c + 1 < len(self.board.board[r]) and self.board.board[r][c + 1]:
                    self._maybe_connect_border((r, c), "RIGHT", (r, c + 1), "LEFT")

    def _maybe_connect_border(self, pos_a: Tuple[int, int], edge_a: str,
                              pos_b: Tuple[int, int], edge_b: str):
        feat_a = self._tile_edges.get(pos_a, {}).get(edge_a, FEATURE_FIELD)
        feat_b = self._tile_edges.get(pos_b, {}).get(edge_b, FEATURE_FIELD)
        if feat_a != feat_b or feat_a not in (FEATURE_CITY, FEATURE_ROAD):
            return
        lr_a, lc_a = EDGE_CENTERS[edge_a]
        lr_b, lc_b = EDGE_CENTERS[edge_b]
        ga = (pos_a[0] * GRID_SIZE + lr_a, pos_a[1] * GRID_SIZE + lc_a)
        gb = (pos_b[0] * GRID_SIZE + lr_b, pos_b[1] * GRID_SIZE + lc_b)
        if ga in self._global_cells and gb in self._global_cells:
            self._adjacency[ga].add(gb)
            self._adjacency[gb].add(ga)

    def _derive_edges_from_grid(self, rotated_grid: List[str]) -> Dict[str, str]:
        edges: Dict[str, str] = {}
        for edge, (lr, lc) in EDGE_CENTERS.items():
            val = rotated_grid[lr * GRID_SIZE + lc]
            if val == FEATURE_CITY:
                edges[edge] = FEATURE_CITY
            elif val == FEATURE_ROAD:
                edges[edge] = FEATURE_ROAD
            else:
                edges[edge] = FEATURE_FIELD
        return edges

    def _create_clique(self, cells: List[Tuple[int, int]]):
        n = len(cells)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = cells[i], cells[j]
                self._adjacency[a].add(b)
                self._adjacency[b].add(a)

    def _get_orthogonal_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        return [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

    def score(self) -> Dict[int, int]:
        scores = {1: 0, 2: 0}
        self._score_cities_and_roads(scores)
        self._score_monasteries(scores)
        return scores

    def _score_cities_and_roads(self, scores: Dict[int, int]):
        components = self._find_connected_components()

        for comp in components:
            tiles = self._get_tiles_from_cells(comp.cells)
            points = self._calculate_component_points(comp)
            winners = self._find_component_winners(comp.cells)

            # Solo mostrar debug por meeples que sumaron puntos
            if DEBUG and points > 0 and winners:
                tipo = "Ciudad" if comp.feature == FEATURE_CITY else "Camino"
                for pos in comp.cells:
                    m = self._global_cells[pos].meeple
                    if not m:
                        continue
                    player = m.get("player", 0)
                    if player in winners:
                        dbg(f"Jugador {player} +{points} puntos por {tipo} (tile {m['tile_position']}, pos {m['original_position']})")

            for p in winners:
                scores[p] += points

    def _score_monasteries(self, scores: Dict[int, int]):
        for tr in range(self.height):
            for tc in range(len(self.board.board[tr])):
                tile = self.board.board[tr][tc]
                if not tile:
                    continue
                center_global = (tr * GRID_SIZE + 1, tc * GRID_SIZE + 1)
                cell = self._global_cells.get(center_global)
                if not cell or cell.feature != FEATURE_MONASTERY:
                    continue
                m = cell.meeple
                if not m or m.get("player", 0) <= 0:
                    continue
                pts = self._count_surrounding_tiles(tr, tc)
                scores[m["player"]] += pts
                if DEBUG:
                    dbg(f"Jugador {m['player']} +{pts} puntos por Monasterio (tile {(tr, tc)}, pos 5)")

    def _find_connected_components(self) -> List["FeatureComponent"]:
        visited: Set[Tuple[int, int]] = set()
        comps: List[FeatureComponent] = []
        for pos, cell in self._global_cells.items():
            if pos in visited or cell.feature not in SCOREABLE_FEATURES:
                continue
            feature = cell.feature
            queue = deque([pos])
            visited.add(pos)
            cells = {pos}
            while queue:
                cur = queue.popleft()
                for nxt in self._adjacency.get(cur, ()):
                    if nxt in visited or self._global_cells[nxt].feature != feature:
                        continue
                    visited.add(nxt)
                    queue.append(nxt)
                    cells.add(nxt)
            comps.append(FeatureComponent(feature=feature, cells=cells))
        return comps

    def _get_tiles_from_cells(self, cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        return {self._global_cells[pos].tile_pos for pos in cells}

    def _calculate_component_points(self, comp: "FeatureComponent") -> int:
        tiles = self._get_tiles_from_cells(comp.cells)
        if comp.feature == FEATURE_ROAD:
            return len(tiles)
        if comp.feature == FEATURE_CITY:
            pennants = sum(1 for t in tiles if self._tile_has_pennant.get(t, False))
            return len(tiles) + pennants
        return 0

    def _find_component_winners(self, cells: Set[Tuple[int, int]]) -> List[int]:
        counts: Dict[int, int] = defaultdict(int)
        for pos in cells:
            m = self._global_cells[pos].meeple
            if m and m.get("player", 0) > 0:
                counts[m["player"]] += 1
        if not counts:
            return []
        mx = max(counts.values())
        return sorted([p for p, n in counts.items() if n == mx])

    def _is_component_complete(self, comp: "FeatureComponent", tiles: List[Tuple[int, int]]) -> bool:
        """
        Un componente está completo si cada borde de ese tipo tiene vecino
        con el mismo borde y su centro opuesto pertenece al mismo componente.
        """
        cells_set = comp.cells
        feat = comp.feature
        for tr, tc in tiles:
            edges = self._tile_edges.get((tr, tc), {})
            for edge, edge_feat in edges.items():
                if edge_feat != feat:
                    continue
                lr, lc = EDGE_CENTERS[edge]
                g_here = (tr * GRID_SIZE + lr, tc * GRID_SIZE + lc)
                if g_here not in cells_set:
                    continue
                nr, nc, opp = {
                    "TOP": (tr - 1, tc, "BOTTOM"),
                    "BOTTOM": (tr + 1, tc, "TOP"),
                    "LEFT": (tr, tc - 1, "RIGHT"),
                    "RIGHT": (tr, tc + 1, "LEFT"),
                }[edge]
                if not self._is_valid_tile_position(nr, nc):
                    return False
                if self._tile_edges.get((nr, nc), {}).get(opp) != feat:
                    return False
                lrn, lcn = EDGE_CENTERS[opp]
                g_nei = (nr * GRID_SIZE + lrn, nc * GRID_SIZE + lcn)
                if g_nei not in cells_set:
                    return False
        return True

    def _count_surrounding_tiles(self, tr: int, tc: int) -> int:
        count = 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if self._is_valid_tile_position(tr + dr, tc + dc):
                    count += 1
        return count

    def _is_valid_tile_position(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.height
            and 0 <= col < len(self.board.board[row])
            and self.board.board[row][col] is not None
        )
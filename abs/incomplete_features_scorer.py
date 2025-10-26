"""
Sistema de puntuacion final para Carcassonne.
Calcula puntos solo de características incompletas (ciudades, caminos, monasterios).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, deque

# -------------------------
# Constantes
# -------------------------
GRID_SIZE = 3
ROTATION_STEP = 90
MAX_MONASTERY_TILES = 9
EDGE_ORDER = ["TOP", "RIGHT", "BOTTOM", "LEFT"]

EDGE_CELLS = {
    "LEFT":   [(0, 0), (1, 0), (2, 0)],
    "TOP":    [(0, 0), (0, 1), (0, 2)],
    "RIGHT":  [(0, 2), (1, 2), (2, 2)],
    "BOTTOM": [(2, 0), (2, 1), (2, 2)],
}

# Características de tiles
FEATURE_CITY = "C"
FEATURE_ROAD = "R"
FEATURE_FIELD = "F"
FEATURE_MONASTERY = "M"
FEATURE_TOWN = "T"

SCOREABLE_FEATURES = {FEATURE_CITY, FEATURE_ROAD} #Los que pueden estar completos o incompletos (los monasterios no)

# -------------------------
# Modelos de datos
# -------------------------
@dataclass
class Tile:
    """Representa un tile del tablero"""
    name: str  # Nombre del tile (A-X)
    orientation: int  # Orientacio: 0-3 (rotaciones de 90 grados)
    meeple: Optional[Tuple[int, int]] = None  # (jugador, posición 0-8)


@dataclass
class Board:
    """Tablero de juego con matriz de tiles"""
    tiles: List[List[Optional[Tile]]]


@dataclass
class TileConfig:
    """Configuracion de un tipo de tile"""
    grid: List[str]  # Grilla 3x3 con caracteristicas
    feature_connections: List[List[str]]  # Conexiones internas entre bordes
    has_pennant: bool  # Si tiene estandarte


# -------------------------
# Configuracion de tiles
# -------------------------
TILE_INFO: Dict[str, TileConfig] = {
    "A": TileConfig(
        grid=["F", "F", "F", "F", "M", "F", "F", "R", "F"],
        feature_connections=[],
        has_pennant=False
    ),
    "B": TileConfig(
        grid=["F", "F", "F", "F", "M", "F", "F", "F", "F"],
        feature_connections=[],
        has_pennant=False
    ),
    "C": TileConfig(
        grid=["C", "C", "C", "C", "C", "C", "C", "C", "C"],
        feature_connections=[["LEFT", "TOP", "RIGHT", "BOTTOM"]],
        has_pennant=True
    ),
    "D": TileConfig(
        grid=["F", "R", "F", "F", "R", "C", "F", "R", "F"],
        feature_connections=[["TOP", "BOTTOM"]],
        has_pennant=False
    ),
    "E": TileConfig(
        grid=["C", "C", "C", "F", "F", "F", "F", "F", "F"],
        feature_connections=[],
        has_pennant=False
    ),
    "F": TileConfig(
        grid=["F", "F", "F", "C", "C", "C", "F", "F", "F"],
        feature_connections=[["LEFT", "RIGHT"]],
        has_pennant=True
    ),
    "G": TileConfig(
        grid=["F", "C", "F", "F", "C", "F", "F", "C", "F"],
        feature_connections=[["TOP", "BOTTOM"]],
        has_pennant=False
    ),
    "H": TileConfig(
        grid=["C", "F", "C", "C", "F", "C", "C", "F", "C"],
        feature_connections=[],
        has_pennant=False
    ),
    "I": TileConfig(
        grid=["F", "F", "F", "F", "F", "C", "F", "C", "C"],
        feature_connections=[],
        has_pennant=False
    ),
    "J": TileConfig(
        grid=["F", "C", "F", "F", "R", "R", "F", "R", "F"],
        feature_connections=[["RIGHT", "BOTTOM"]],
        has_pennant=False
    ),
    "K": TileConfig(
        grid=["F", "R", "C", "R", "R", "C", "F", "F", "C"],
        feature_connections=[["LEFT", "TOP"]],
        has_pennant=False
    ),
    "L": TileConfig(
        grid=["F", "R", "F", "R", "T", "C", "F", "R", "F"],
        feature_connections=[],
        has_pennant=False
    ),
    "M": TileConfig(
        grid=["C", "C", "F", "C", "F", "F", "F", "F", "F"],
        feature_connections=[["LEFT", "TOP"]],
        has_pennant=True
    ),
    "N": TileConfig(
        grid=["C", "C", "F", "C", "F", "F", "F", "F", "F"],
        feature_connections=[["LEFT", "TOP"]],
        has_pennant=False
    ),
    "O": TileConfig(
        grid=["C", "C", "F", "C", "R", "R", "F", "R", "F"],
        feature_connections=[["LEFT", "TOP"], ["RIGHT", "BOTTOM"]],
        has_pennant=True
    ),
    "P": TileConfig(
        grid=["C", "C", "F", "C", "R", "R", "F", "R", "F"],
        feature_connections=[["LEFT", "TOP"], ["RIGHT", "BOTTOM"]],
        has_pennant=False
    ),
    "Q": TileConfig(
        grid=["C", "C", "C", "C", "C", "C", "F", "F", "F"],
        feature_connections=[["LEFT", "TOP", "RIGHT"]],
        has_pennant=True
    ),
    "R": TileConfig(
        grid=["C", "C", "C", "C", "C", "C", "F", "F", "F"],
        feature_connections=[["LEFT", "TOP", "RIGHT"]],
        has_pennant=False
    ),
    "S": TileConfig(
        grid=["C", "C", "C", "C", "C", "C", "F", "R", "F"],
        feature_connections=[["LEFT", "TOP", "RIGHT"]],
        has_pennant=True
    ),
    "T": TileConfig(
        grid=["C", "C", "C", "C", "C", "C", "F", "R", "F"],
        feature_connections=[["LEFT", "TOP", "RIGHT"]],
        has_pennant=False
    ),
    "U": TileConfig(
        grid=["F", "R", "F", "F", "R", "F", "F", "R", "F"],
        feature_connections=[["TOP", "BOTTOM"]],
        has_pennant=False
    ),
    "V": TileConfig(
        grid=["F", "F", "F", "R", "R", "F", "F", "R", "F"],
        feature_connections=[["LEFT", "BOTTOM"]],
        has_pennant=False
    ),
    "W": TileConfig(
        grid=["F", "F", "F", "R", "T", "R", "F", "R", "F"],
        feature_connections=[],
        has_pennant=False
    ),
    "X": TileConfig(
        grid=["F", "R", "F", "R", "T", "R", "F", "R", "F"],
        feature_connections=[],
        has_pennant=False
    ),
}


# -------------------------
# Utilidades de geometría
# -------------------------
class GridRotator:
    """Maneja rotaciones de grillas 3x3"""
    
    @staticmethod
    def rotate_grid(flat_grid: List[str], degrees: int) -> List[str]:
        """Rota una grilla 3x3 en sentido horario"""
        # Sin rotación
        if degrees % 360 == 0:
            return flat_grid[:]
        
        # Convertir a matriz 3x3
        matrix = [
            flat_grid[0:3],
            flat_grid[3:6],
            flat_grid[6:9]
        ]
        
        # Rotar en pasos de 90 grados
        steps = (degrees % 360) // ROTATION_STEP
        for _ in range(steps):
            matrix = [[matrix[2 - j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
        
        # Aplanar de vuelta
        return [cell for row in matrix for cell in row]
    
    @staticmethod
    def rotate_position(row: int, col: int, steps: int) -> Tuple[int, int]:
        """Rota una posición (i, j) en una grilla 3x3"""
        for _ in range(steps):
            row, col = col, 2 - row
        return row, col
    
    @staticmethod
    def rotate_edge(edge: str, steps: int) -> str:
        """Rota el nombre de un borde (TOP->RIGHT->BOTTOM->LEFT)"""
        index = EDGE_ORDER.index(edge)
        new_index = (index + steps) % len(EDGE_ORDER)
        return EDGE_ORDER[new_index]


# -------------------------
# Sistema de puntuacion
# -------------------------
class CellInfo:
    """Información de una celda en la grilla global expandida"""
    def __init__(self, feature: str, tile_pos: Tuple[int, int], 
                 local_pos: Tuple[int, int], letter: str):
        self.feature = feature  # Tipo de característica (C/R/F/M/T)
        self.tile_pos = tile_pos  # Posicion del tile padre
        self.local_pos = local_pos  # Posicion local dentro del tile (0-2, 0-2)
        self.letter = letter  # Letra del tile (A-X)
        self.meeple: Optional[Dict] = None  # Info del meeple si existe


class FeatureComponent:
    """Representa un componente conexo de una característica"""
    def __init__(self, feature: str, cells: Set[Tuple[int, int]]):
        self.feature = feature  # Tipo (C o R)
        self.cells = cells  # Conjunto de celdas globales


class GameScorer:
    """Calculador de puntuacion final del juego"""
    
    def __init__(self, board: Board):
        self.board = board
        self.height = len(board.tiles)
        self.width = max(len(row) for row in board.tiles)
        
        # Grilla global expandida (cada tile = 3x3 celdas)
        self._global_cells: Dict[Tuple[int, int], CellInfo] = {}
        
        # Registro de tiles con estandarte
        self._tile_has_pennant: Dict[Tuple[int, int], bool] = {}
        
        # Grafo de adyacencia entre celdas de la misma caracteristica
        self._adjacency: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        
        # Construir estructuras
        self._build_global_grid()
        self._build_adjacency_graph()
    
    # -------------------------
    # Construccion del tablero
    # -------------------------
    def _build_global_grid(self):
        """Expande cada tile en una grilla 3x3 global"""
        for tile_row in range(self.height):
            for tile_col in range(len(self.board.tiles[tile_row])):
                tile = self.board.tiles[tile_row][tile_col]
                if not tile:
                    continue
                
                self._process_tile(tile, tile_row, tile_col)
    
    def _process_tile(self, tile: Tile, tile_row: int, tile_col: int):
        """Procesa un tile: rota su grilla, crea celdas y asigna meeple"""
        config = TILE_INFO[tile.name]
        rotated_grid = GridRotator.rotate_grid(
            config.grid,
            tile.orientation * ROTATION_STEP
        )
        
        # Crear las 9 celdas globales
        for local_row in range(GRID_SIZE):
            for local_col in range(GRID_SIZE):
                self._create_global_cell(
                    tile, tile_row, tile_col,
                    local_row, local_col, rotated_grid
                )
        
        # Registrar si tiene estandarte
        self._tile_has_pennant[(tile_row, tile_col)] = config.has_pennant
        
        # Asignar meeple si existe
        if tile.meeple:
            self._assign_meeple(tile, tile_row, tile_col)
    
    def _create_global_cell(self, tile: Tile, tile_row: int, tile_col: int,
                           local_row: int, local_col: int, rotated_grid: List[str]):
        """Crea una celda en la grilla global expandida"""
        # Coordenadas globales
        global_row = tile_row * GRID_SIZE + local_row
        global_col = tile_col * GRID_SIZE + local_col
        
        # Caracteristica de la celda
        feature = rotated_grid[local_row * GRID_SIZE + local_col]
        
        # Crear y guardar info de la celda
        self._global_cells[(global_row, global_col)] = CellInfo(
            feature=feature,
            tile_pos=(tile_row, tile_col),
            local_pos=(local_row, local_col),
            letter=tile.name
        )
    
    def _assign_meeple(self, tile: Tile, tile_row: int, tile_col: int):
        """Asigna un meeple a su celda global (teniendo en cuenta la rotacion)"""
        player, position = tile.meeple
        
        # Convertir posicion 0-8 a coordenadas locales
        local_row, local_col = divmod(position, GRID_SIZE)
        
        # Rotar posicion según orientación del tile
        rotated_row, rotated_col = GridRotator.rotate_position(
            local_row, local_col, tile.orientation
        )
        
        # Calcular coordenadas globales
        global_row = tile_row * GRID_SIZE + rotated_row
        global_col = tile_col * GRID_SIZE + rotated_col
        
        # Asignar meeple a la celda
        self._global_cells[(global_row, global_col)].meeple = {
            "player": int(player),
            "original_position": position,
            "tile_position": (tile_row, tile_col)
        }
    
    # -------------------------
    # Construccion del grafo
    # -------------------------
    def _build_adjacency_graph(self):
        """Construye el grafo de adyacencia entre celdas conectadas"""
        self._add_orthogonal_adjacencies()
        self._add_internal_connections()
    
    def _add_orthogonal_adjacencies(self):
        """Conecta celdas ortogonales vecinas con la misma característica"""
        for (row, col), cell_info in self._global_cells.items():
            for neighbor_pos in self._get_orthogonal_neighbors(row, col):
                if self._are_connected_orthogonally(cell_info, neighbor_pos):
                    self._adjacency[(row, col)].add(neighbor_pos)
    
    def _get_orthogonal_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Retorna las 4 posiciones ortogonales vecinas"""
        return [
            (row - 1, col),  # arriba
            (row + 1, col),  # abajo
            (row, col - 1),  # izquierda
            (row, col + 1)   # derecha
        ]
    
    def _are_connected_orthogonally(self, cell_info: CellInfo, 
                                    neighbor_pos: Tuple[int, int]) -> bool:
        """Verifica si dos celdas vecinas están conectadas (misma característica)"""
        if neighbor_pos not in self._global_cells:
            return False
        
        neighbor_info = self._global_cells[neighbor_pos]
        return neighbor_info.feature == cell_info.feature
    
    def _add_internal_connections(self):
        """Agrega conexiones internas dentro de cada tile (feature_connections)"""
        for tile_row in range(self.height):
            for tile_col in range(len(self.board.tiles[tile_row])):
                tile = self.board.tiles[tile_row][tile_col]
                if not tile:
                    continue
                
                self._process_tile_connections(tile, tile_row, tile_col)
    
    def _process_tile_connections(self, tile: Tile, tile_row: int, tile_col: int):
        """Procesa las conexiones internas definidas en feature_connections"""
        config = TILE_INFO[tile.name]
        
        # Rotar cada grupo de bordes conectados
        for edge_group in config.feature_connections:
            rotated_edges = [
                GridRotator.rotate_edge(edge, tile.orientation)
                for edge in edge_group
            ]
            self._connect_edge_cells(tile_row, tile_col, rotated_edges)
    
    def _connect_edge_cells(self, tile_row: int, tile_col: int, edges: List[str]):
        """Conecta las celdas de los bordes especificados (formando un clique)"""
        # Para cada celda dentro del borde (0, 1, 2)
        for cell_index in range(GRID_SIZE):
            cells_to_connect = []
            
            # Recoger las celdas de todos los bordes en esa posición
            for edge in edges:
                local_row, local_col = EDGE_CELLS[edge][cell_index]
                global_row = tile_row * GRID_SIZE + local_row
                global_col = tile_col * GRID_SIZE + local_col
                cells_to_connect.append((global_row, global_col))
            
            # Conectar todas entre si (clique)
            self._create_clique(cells_to_connect)
    
    def _create_clique(self, cells: List[Tuple[int, int]]):
        """Conecta todas las celdas entre sí (grafo completo)"""
        for cell_a in cells:
            for cell_b in cells:
                if cell_a != cell_b and self._can_connect(cell_a, cell_b):
                    self._adjacency[cell_a].add(cell_b)
    
    def _can_connect(self, cell_a: Tuple[int, int], cell_b: Tuple[int, int]) -> bool:
        """Verifica si dos celdas pueden conectarse (misma característica)"""
        if cell_a not in self._global_cells or cell_b not in self._global_cells:
            return False
        
        return (self._global_cells[cell_a].feature == 
                self._global_cells[cell_b].feature)
    
    # -------------------------
    # Analisis de componentes
    # -------------------------
    def _find_connected_components(self) -> List[FeatureComponent]:
        """Encuentra todos los componentes conexos de ciudades y caminos"""
        visited: Set[Tuple[int, int]] = set()
        components: List[FeatureComponent] = []
        
        for cell_pos, cell_info in self._global_cells.items():
            # Solo ciudades y caminos
            if cell_info.feature not in SCOREABLE_FEATURES:
                continue
            
            # Saltar si ya fue visitada
            if cell_pos in visited:
                continue
            
            # Explorar componente
            component = self._explore_component(cell_pos, visited)
            components.append(component)
        
        return components
    
    def _explore_component(self, start_pos: Tuple[int, int], 
                          visited: Set[Tuple[int, int]]) -> FeatureComponent:
        """Explora un componente conexo usando BFS"""
        queue = deque([start_pos])
        component_cells: Set[Tuple[int, int]] = set()
        
        while queue:
            current = queue.popleft()
            
            if current in component_cells:
                continue
            
            component_cells.add(current)
            
            # Agregar vecinos conectados
            for neighbor in self._adjacency[current]:
                if neighbor not in component_cells:
                    queue.append(neighbor)
        
        # Marcar como visitadas
        visited.update(component_cells)
        
        # Crear componente
        feature = self._global_cells[start_pos].feature
        return FeatureComponent(feature, component_cells)
    
    def _get_tiles_from_cells(self, cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Convierte un conjunto de celdas globales a posiciones de tiles"""
        return {(row // GRID_SIZE, col // GRID_SIZE) for row, col in cells}
    
    def _is_component_complete(self, cells: Set[Tuple[int, int]], feature: str) -> bool:
        """Verifica si un componente está cerrado (no tiene bordes abiertos)"""
        for row, col in cells:
            # Verificar cada vecino ortogonal
            for neighbor_pos in self._get_orthogonal_neighbors(row, col):
                if not self._is_valid_boundary(neighbor_pos, feature):
                    return False  # Hay un borde abierto
        return True
    
    def _is_valid_boundary(self, pos: Tuple[int, int], feature: str) -> bool:
        """Verifica si una posición es un borde válido (misma característica o borde del tablero)"""
        # Borde del tablero = válido
        if pos not in self._global_cells:
            return True
        
        # Debe tener la misma característica
        return self._global_cells[pos].feature == feature
    
    # -------------------------
    # Calculo de puntuación
    # -------------------------
    def score(self) -> Dict[int, int]:
        """Calcula la puntuación final de ambos jugadores"""
        scores = {1: 0, 2: 0}
        
        self._score_cities_and_roads(scores)
        self._score_monasteries(scores)
        
        return scores
    
    def _score_cities_and_roads(self, scores: Dict[int, int]):
        """Puntúa todas las ciudades y caminos incompletos"""
        components = self._find_connected_components()
        
        for component in components:
            # Ignorar si esta completo
            if self._is_component_complete(component.cells, component.feature):
                continue
            
            # Calcular puntos
            points = self._calculate_component_points(component)
            
            # Encontrar ganadores (mayoria de meeples)
            winners = self._find_component_winners(component.cells)
            
            # Asignar puntos
            for player in winners:
                scores[player] += points
    
    def _calculate_component_points(self, component: FeatureComponent) -> int:
        """Calcula los puntos de un componente"""
        # Obtener tiles unicos
        tiles = self._get_tiles_from_cells(component.cells)
        points = len(tiles)
        
        # Ciudades: sumar estandartes
        if component.feature == FEATURE_CITY:
            pennants = sum(1 for tile_pos in tiles 
                          if self._tile_has_pennant.get(tile_pos, False))
            points += pennants
        
        return points
    
    def _find_component_winners(self, cells: Set[Tuple[int, int]]) -> List[int]:
        """Encuentra los jugadores con mayoría de meeples en el componente"""
        meeple_counts = defaultdict(int)
        
        # Contar meeples por jugador
        for cell_pos in cells:
            meeple = self._global_cells[cell_pos].meeple
            if meeple:
                meeple_counts[meeple["player"]] += 1
        
        # Sin meeples = sin ganador
        if not meeple_counts:
            return []
        
        # Encontrar maximo y retornar todos los empatados
        max_count = max(meeple_counts.values())
        return [player for player, count in meeple_counts.items() 
                if count == max_count]
    
    def _score_monasteries(self, scores: Dict[int, int]):
        """Puntua todos los monasterios incompletos"""
        for tile_row in range(self.height):
            for tile_col in range(len(self.board.tiles[tile_row])):
                tile = self.board.tiles[tile_row][tile_col]
                if not tile:
                    continue
                
                if self._is_monastery(tile):
                    self._score_monastery(tile, tile_row, tile_col, scores)
    
    def _is_monastery(self, tile: Tile) -> bool:
        """Verifica si un tile tiene un monasterio en el centro"""
        config = TILE_INFO[tile.name]
        rotated_grid = GridRotator.rotate_grid(
            config.grid,
            tile.orientation * ROTATION_STEP
        )
        center_index = (GRID_SIZE * GRID_SIZE) // 2  # posicion 4
        return rotated_grid[center_index] == FEATURE_MONASTERY
    
    def _score_monastery(self, tile: Tile, tile_row: int, tile_col: int, 
                        scores: Dict[int, int]):
        """Puntua un monasterio individual (solo si esta incompleto)"""
        # Contar tiles alrededor
        surrounding_count = self._count_surrounding_tiles(tile_row, tile_col)
        
        # Completo = 9 tiles (no puntua)
        if surrounding_count == MAX_MONASTERY_TILES:
            return
        
        # Buscar meeple en el centro
        center_row = tile_row * GRID_SIZE + 1
        center_col = tile_col * GRID_SIZE + 1
        meeple = self._global_cells.get((center_row, center_col), CellInfo("", (0,0), (0,0), "")).meeple
        
        # Asignar puntos al dueño del meeple
        if meeple:
            scores[meeple["player"]] += surrounding_count
    
    def _count_surrounding_tiles(self, tile_row: int, tile_col: int) -> int:
        """Cuenta tiles alrededor de una posicion (incluido el propio = max 9)"""
        count = 1  # el tile del monasterio mismo
        
        # Verificar los 8 vecinos
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                
                neighbor_row = tile_row + delta_row
                neighbor_col = tile_col + delta_col
                
                if self._is_valid_tile_position(neighbor_row, neighbor_col):
                    count += 1
        
        return count
    
    def _is_valid_tile_position(self, row: int, col: int) -> bool:
        """Verifica si una posicion contiene un tile valido"""
        if not (0 <= row < self.height):
            return False
        
        if not (0 <= col < len(self.board.tiles[row])):
            return False
        
        return self.board.tiles[row][col] is not None


# -------------------------
# Ejemplo de uso
# -------------------------
def create_example_board() -> Board:
    """Crea un tablero de ejemplo para demostracion"""
    origin_matrix = [
        # FILA 0
        [None, None, None, None, Tile("W", 0, None), Tile("D", 1, (2, 4)), 
         None, None, None, None, None, None],
        
        # FILA 1
        [None, Tile("C", 0, None), None, None, Tile("L", 0, None), Tile("N", 0, None),
         Tile("N", 2, None), Tile("T", 3, None), Tile("S", 1, (1, 4)), 
         Tile("H", 0, None), Tile("I", 2, None), None],
        
        # FILA 2
        [None, Tile("S", 3, None), Tile("V", 0, None), None, Tile("W", 2, None),
         Tile("V", 0, None), None, Tile("P", 0, None), None, 
         Tile("E", 2, None), None, None],
        
        # FILA 3
        [None, None, Tile("W", 3, None), Tile("U", 1, None), Tile("U", 1, None),
         Tile("P", 0, None), Tile("M", 3, None), None, Tile("U", 0, None),
         Tile("H", 1, None), None, None],
        
        # FILA 4
        [None, None, Tile("U", 0, None), Tile("V", 3, None), Tile("V", 0, None),
         Tile("B", 0, None), Tile("E", 0, None), Tile("V", 3, None),
         Tile("K", 0, None), Tile("R", 0, (1, 4)), Tile("P", 0, None), Tile("J", 2, None)],
        
        # FILA 5
        [Tile("A", 0, (2, 4)), None, Tile("U", 0, None), Tile("V", 2, None),
         Tile("K", 0, None), Tile("F", 0, None), Tile("M", 3, None),
         Tile("D", 0, None), Tile("G", 1, None), Tile("H", 0, None), None, None],
        
        # FILA 6
        [Tile("V", 2, None), Tile("U", 1, None), Tile("W", 2, None), Tile("D", 1, None),
         None, None, Tile("I", 3, None), Tile("M", 3, None), Tile("A", 1, None),
         Tile("V", 3, None), Tile("V", 0, None), Tile("U", 0, None)],
        
        # FILA 7
        [None, Tile("B", 0, None), Tile("E", 1, None), Tile("R", 3, None),
         Tile("E", 0, None), Tile("Q", 3, None), Tile("N", 3, None),
         Tile("R", 0, None), None, Tile("D", 2, None), None, None],
        
        # FILA 8
        [None, None, Tile("B", 0, None), Tile("E", 0, None), Tile("B", 0, None),
         None, Tile("F", 1, None), None, None, Tile("V", 0, None),
         Tile("X", 1, None), None],
        
        # FILA 9
        [None, None, None, None, None, None, Tile("L", 3, None),
         Tile("P", 3, None), Tile("O", 3, None), None, None, None],
        
        # FILA 10
        [None, None, None, None, None, None, Tile("U", 0, None),
         None, Tile("L", 3, None), None, None, None]
    ]
    
    return Board(origin_matrix)


def print_scores(scores: Dict[int, int]):
    """Imprime los resultados de forma legible"""
    separator = "=" * 50
    
    print(f"\n{separator}")
    print("PUNTUACION FINAL")
    print(separator)
    print(f"Jugador 1: {scores[1]} puntos")
    print(f"Jugador 2: {scores[2]} puntos")
    print(separator)


def main():
    """Funcion principal"""
    board = create_example_board()
    scorer = GameScorer(board)
    scores = scorer.score()
    print_scores(scores)


if __name__ == "__main__":
    main()

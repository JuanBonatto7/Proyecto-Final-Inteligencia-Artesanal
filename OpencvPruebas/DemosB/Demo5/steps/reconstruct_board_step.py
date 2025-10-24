"""
Paso 10: Reconstruir representación del tablero.
Crea una matriz y visualización del estado del juego.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pipeline.pipeline_step import PipelineStep
from config import Config


class Tile:
    """Representa una loseta en el tablero"""
    
    def __init__(
        self,
        tile_type: str,
        rotation: int,
        position: Tuple[int, int],
        confidence: float = 0.0
    ):
        self.tile_type = tile_type
        self.rotation = rotation
        self.position = position  # (row, col) o (i, j)
        self.confidence = confidence
    
    def __str__(self):
        return f"{self.tile_type}R{self.rotation}"
    
    def __repr__(self):
        return f"Tile({self.tile_type}, rot={self.rotation}, pos={self.position}, conf={self.confidence:.2f})"


class Board:
    """Representa el estado completo del tablero"""
    
    def __init__(self):
        self.tiles: List[Tile] = []
        self.rows: int = 0
        self.cols: int = 0
    
    def add_tile(self, tile: Tile):
        """Añade una loseta al tablero"""
        self.tiles.append(tile)
        self._update_dimensions()
    
    def _update_dimensions(self):
        """Actualiza las dimensiones del tablero basándose en las posiciones"""
        if not self.tiles:
            self.rows = 0
            self.cols = 0
            return
        
        positions = [t.position for t in self.tiles]
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        
        self.rows = max(rows) - min(rows) + 1
        self.cols = max(cols) - min(cols) + 1
    
    def get_matrix_representation(self) -> List[List[Optional[Tile]]]:
        """
        Retorna matriz 2D del tablero.
        
        Returns:
            Matriz donde cada celda es un Tile o None
        """
        if not self.tiles:
            return []
        
        # Encontrar límites
        positions = [t.position for t in self.tiles]
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        # Crear matriz
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        matrix = [[None for _ in range(width)] for _ in range(height)]
        
        # Llenar matriz
        for tile in self.tiles:
            i, j = tile.position
            row = i - min_row
            col = j - min_col
            matrix[row][col] = tile
        
        return matrix


class ReconstructBoardStep(PipelineStep):
    """
    Reconstruye el tablero a partir de los tiles clasificados.
    
    Crea:
    - Objeto Board con todas las losetas
    - Matriz 2D del tablero
    - Visualización del tablero completo
    """
    
    def __init__(self):
        super().__init__("Reconstruct Board")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruye el tablero"""
        
        classified_tiles = inputs.get('classified_tiles', [])
        img = inputs.get('img')
        
        if not classified_tiles:
            print("⚠ No hay tiles clasificados para reconstruir")
            board = Board()
            inputs['board'] = board
            inputs['board_matrix'] = []
            inputs['debug_image'] = img.copy() if img is not None else np.zeros((600, 600, 3), dtype=np.uint8)
            return inputs
        
        print(f"  Reconstruyendo tablero con {len(classified_tiles)} tiles...")
        
        # Crear objeto Board
        board = Board()
        
        for tile_data in classified_tiles:
            tile = Tile(
                tile_type=tile_data['tile_type'],
                rotation=tile_data['rotation'],
                position=tile_data['position'],
                confidence=tile_data.get('confidence', 0.0)
            )
            board.add_tile(tile)
        
        # Obtener matriz
        matrix = board.get_matrix_representation()
        
        print(f"  ✅ Tablero reconstruido: {board.rows}x{board.cols}")
        
        # Guardar resultados
        inputs['board'] = board
        inputs['board_matrix'] = matrix
        
        # Crear visualización
        debug_image = self._create_visualization(img, board, matrix)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _create_visualization(
        self,
        img: np.ndarray,
        board: Board,
        matrix: List[List[Optional[Tile]]]
    ) -> np.ndarray:
        """
        Crea visualización del tablero reconstruido.
        
        Args:
            img: Imagen original
            board: Objeto Board
            matrix: Matriz del tablero
            
        Returns:
            Imagen con visualización
        """
        # Usar imagen original o crear canvas
        if img is not None:
            if len(img.shape) == 2:
                result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                result = img.copy()
        else:
            result = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Añadir panel de información
        info_panel = self._create_info_panel(board, matrix)
        
        # Redimensionar imagen si es necesario para añadir panel
        h, w = result.shape[:2]
        panel_height = 200
        
        # Crear imagen combinada
        combined = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
        combined[:h, :w] = result
        combined[h:, :] = info_panel[:panel_height, :w] if info_panel.shape[1] >= w else np.pad(
            info_panel[:panel_height],
            ((0, 0), (0, w - info_panel.shape[1]), (0, 0)),
            mode='constant'
        )
        
        return combined
    
    def _create_info_panel(
        self,
        board: Board,
        matrix: List[List[Optional[Tile]]]
    ) -> np.ndarray:
        """Crea panel de información del tablero"""
        
        panel_width = 800
        panel_height = 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Título
        cv2.putText(
            panel, "TABLERO RECONSTRUIDO",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )
        
        # Estadísticas
        y_offset = 60
        stats = [
            f"Dimensiones: {board.rows}x{board.cols}",
            f"Total tiles: {len(board.tiles)}",
            f"Confianza promedio: {np.mean([t.confidence for t in board.tiles]):.1%}" if board.tiles else "N/A"
        ]
        
        for stat in stats:
            cv2.putText(
                panel, stat,
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            y_offset += 25
        
        # Matriz (versión simplificada)
        y_offset += 10
        cv2.putText(
            panel, "Matriz del tablero:",
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
        y_offset += 20
        
        # Mostrar primeras filas de la matriz
        max_rows_to_show = 3
        for i, row in enumerate(matrix[:max_rows_to_show]):
            row_str = " | ".join([str(tile) if tile else "----" for tile in row])
            cv2.putText(
                panel, row_str[:80],  # Limitar longitud
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )
            y_offset += 18
        
        if len(matrix) > max_rows_to_show:
            cv2.putText(
                panel, f"... ({len(matrix) - max_rows_to_show} filas más)",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1
            )
        
        return panel
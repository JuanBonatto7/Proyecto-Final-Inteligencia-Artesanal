"""
Paso 10: Reconstrucción del tablero.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from pipeline.pipeline_step import PipelineStep
from config import Config
from models.tile import Tile
from models.board import Board
from utils.visualization_utils import create_board_visualization


class ReconstructBoardStep(PipelineStep):
    """
    Reconstruye la representación completa del tablero.
    
    Crea una matriz con todas las fichas y sus rotaciones.
    """
    
    def __init__(
        self,
        grid_rows: int = Config.TILE_GRID_ROWS,
        grid_cols: int = Config.TILE_GRID_COLS
    ):
        """
        Inicializa el paso de reconstrucción.
        
        Args:
            grid_rows: Número de filas del tablero
            grid_cols: Número de columnas del tablero
        """
        super().__init__("Reconstruct Board")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruye el tablero a partir de las fichas clasificadas.
        
        Args:
            inputs: Debe contener 'classified_tiles'
            
        Returns:
            Inputs actualizado con:
                - 'board': Objeto Board con el estado completo
                - 'board_matrix': Matriz numpy con identificadores
                - 'debug_image': Visualización del tablero
        """
        classified_tiles: List[Tile] = inputs.get('classified_tiles', [])
        
        # Determinar dimensiones del tablero
        rows, cols = self._determine_board_size(classified_tiles)
        
        # Crear tablero
        board = Board(rows, cols)
        
        # Agregar fichas al tablero
        for tile in classified_tiles:
            try:
                board.add_tile(tile)
            except ValueError as e:
                print(f"⚠ Error agregando ficha {tile}: {e}")
        
        # Obtener matriz
        board_matrix = board.get_matrix_representation()
        
        inputs['board'] = board
        inputs['board_matrix'] = board_matrix
        
        # Crear visualización
        debug_image = self._create_visualization(board, classified_tiles)
        inputs['debug_image'] = debug_image
        
        # Imprimir resumen
        self._print_summary(board)
        
        return inputs
    
    def _determine_board_size(self, tiles: List[Tile]) -> tuple:
        """
        Determina el tamaño del tablero basado en las fichas.
        
        Args:
            tiles: Lista de fichas
            
        Returns:
            Tupla (rows, cols)
        """
        if not tiles:
            return self.grid_rows, self.grid_cols
        
        max_row = max(tile.position[0] for tile in tiles)
        max_col = max(tile.position[1] for tile in tiles)
        
        rows = max(max_row + 1, self.grid_rows)
        cols = max(max_col + 1, self.grid_cols)
        
        return rows, cols
    
    def _create_visualization(
        self,
        board: Board,
        tiles: List[Tile]
    ) -> np.ndarray:
        """Crea visualización del tablero reconstruido."""
        # Crear visualización de matriz
        board_vis = create_board_visualization(board)
        
        # Agregar estadísticas
        stats = board.get_statistics()
        
        # Panel de información
        info_height = 150
        info_panel = np.ones((info_height, board_vis.shape[1], 3), dtype=np.uint8) * 240
        
        # Agregar texto de estadísticas
        y_offset = 30
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        color = (0, 0, 0)
        
        info_lines = [
            f"Tablero: {board.rows}x{board.cols}",
            f"Fichas detectadas: {stats['occupied_cells']} / {stats['total_cells']}",
            f"Ocupacion: {stats['occupation_rate']*100:.1f}%",
            f"Confianza promedio: {stats['average_confidence']:.2f}",
            f"Tipos de fichas: {len(stats['tile_types_count'])}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(
                info_panel,
                line,
                (20, y_offset + i * line_height),
                font,
                font_scale,
                color,
                font_thickness
            )
        
        # Combinar visualización de tablero con panel de info
        result = np.vstack([info_panel, board_vis])
        
        return result
    
    def _print_summary(self, board: Board) -> None:
        """Imprime resumen del tablero en consola."""
        stats = board.get_statistics()
        
        print("\n" + "="*60)
        print("  RESUMEN DEL TABLERO")
        print("="*60)
        print(f"  Dimensiones: {board.rows}x{board.cols}")
        print(f"  Fichas detectadas: {stats['occupied_cells']}")
        print(f"  Celdas vacías: {stats['empty_cells']}")
        print(f"  Tasa de ocupación: {stats['occupation_rate']*100:.1f}%")
        print(f"  Confianza promedio: {stats['average_confidence']:.2f}")
        print(f"\n  Distribución de fichas:")
        
        for tile_type, count in sorted(stats['tile_types_count'].items()):
            print(f"    {tile_type}: {count}")
        
        print("="*60 + "\n")
"""
Modelo de datos para el tablero de Carcassonne.
"""

from typing import List, Optional, Tuple
import numpy as np
from .tile import Tile


class Board:
    """
    Representa el tablero completo del juego Carcassonne.
    
    Attributes:
        rows: Número de filas del tablero
        cols: Número de columnas del tablero
        tiles: Lista de fichas detectadas en el tablero
        matrix: Matriz numpy con identificadores de fichas
    """
    
    def __init__(self, rows: int, cols: int):
        """
        Inicializa el tablero.
        
        Args:
            rows: Número de filas
            cols: Número de columnas
        """
        self.rows = rows
        self.cols = cols
        self.tiles: List[Tile] = []
        self.matrix = np.full((rows, cols), None, dtype=object)
    
    def add_tile(self, tile: Tile) -> None:
        """
        Agrega una ficha al tablero.
        
        Args:
            tile: Ficha a agregar
            
        Raises:
            ValueError: Si la posición está fuera de límites o ya ocupada
        """
        row, col = tile.position
        
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Posición ({row}, {col}) fuera de límites")
        
        if self.matrix[row, col] is not None:
            print(f"Advertencia: Sobreescribiendo ficha en posición ({row}, {col})")
        
        self.tiles.append(tile)
        self.matrix[row, col] = tile.get_identifier()
    
    def get_tile_at(self, row: int, col: int) -> Optional[Tile]:
        """
        Obtiene la ficha en una posición específica.
        
        Args:
            row: Fila
            col: Columna
            
        Returns:
            Tile si existe, None si está vacía
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        
        tile_id = self.matrix[row, col]
        if tile_id is None:
            return None
        
        for tile in self.tiles:
            if tile.position == (row, col):
                return tile
        
        return None
    
    def get_matrix_representation(self) -> np.ndarray:
        """
        Retorna matriz con identificadores de fichas.
        
        Returns:
            Array numpy (rows x cols) con strings "TIPO_ROTACION"
        """
        return self.matrix.copy()
    
    def get_statistics(self) -> dict:
        """
        Retorna estadísticas del tablero.
        
        Returns:
            Diccionario con estadísticas
        """
        total_cells = self.rows * self.cols
        occupied_cells = len(self.tiles)
        empty_cells = total_cells - occupied_cells
        
        tile_types = {}
        for tile in self.tiles:
            tile_types[tile.tile_type] = tile_types.get(tile.tile_type, 0) + 1
        
        avg_confidence = (
            sum(tile.confidence for tile in self.tiles) / len(self.tiles)
            if self.tiles else 0.0
        )
        
        return {
            'total_cells': total_cells,
            'occupied_cells': occupied_cells,
            'empty_cells': empty_cells,
            'occupation_rate': occupied_cells / total_cells if total_cells > 0 else 0,
            'tile_types_count': tile_types,
            'average_confidence': avg_confidence
        }
    
    def __str__(self) -> str:
        """Representación en string del tablero."""
        return f"Board({self.rows}x{self.cols}, {len(self.tiles)} fichas)"
    
    def __repr__(self) -> str:
        """Representación detallada del tablero."""
        return (f"Board(rows={self.rows}, cols={self.cols}, "
                f"tiles={len(self.tiles)})")
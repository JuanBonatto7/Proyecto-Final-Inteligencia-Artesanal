#!/usr/bin/env python3
"""
Origin Matrix Converter
Convierte una imagen de tablero de Carcassonne en una Origin Matrix (Board)
"""

import cv2
import os
import sys
from typing import Optional, List
from tile_cutter import CarcassonneTileDetector
from meeple_detector import MeepleDetector
from tile_detector import CarcassonneTileDetector as TileTypeDetector
from rotation_detector import CarcassonneRotationDetector
from origin_matrix import Tile, Board


class OriginMatrixConverter:
    """Convierte imagen de tablero a Origin Matrix"""
    
    def __init__(self):
        self.tile_detector = CarcassonneTileDetector()
        self.meeple_detector = MeepleDetector()
        self.type_detector = TileTypeDetector()
        self.rotation_detector = CarcassonneRotationDetector()
        self.results = []
        self.min_row = 0
        self.min_col = 0
        self.max_row = 0
        self.max_col = 0
        self.rows = 0
        self.cols = 0
    
    def convert(self, image_path: str, num_reference_points: int = 8) -> Optional[Board]:
        """
        Convierte imagen de tablero a Origin Matrix
        
        Args:
            image_path: Ruta a la imagen del tablero
            num_reference_points: Número de puntos de referencia para detección
            
        Returns:
            Board con la Origin Matrix o None si falla
        """
        # Cargar imagen
        if not self.tile_detector.load_image(image_path):
            return None
        
        # Seleccionar puntos de referencia
        if not self.tile_detector.select_reference_tiles(num_points=num_reference_points):
            return None
        
        # Detectar todas las losetas
        self.tile_detector.assign_grid_positions()
        tiles = self.tile_detector.detect_tiles_interpolated()
        
        if not tiles:
            return None
        
        # Calcular dimensiones
        self._calculate_dimensions(tiles)
        
        # Analizar cada loseta
        self._process_tiles(tiles)
        
        # Crear y retornar Board
        return self._create_board()
    
    def _calculate_dimensions(self, tiles):
        """Calcula dimensiones del tablero"""
        self.min_row = min(tile.grid_row for tile in tiles)
        self.max_row = max(tile.grid_row for tile in tiles)
        self.min_col = min(tile.grid_col for tile in tiles)
        self.max_col = max(tile.grid_col for tile in tiles)
        
        self.rows = self.max_row - self.min_row + 1
        self.cols = self.max_col - self.min_col + 1
    
    def _process_tiles(self, tiles):
        """Procesa todas las losetas detectando tipo, rotación y meeples"""
        self.results = []
        
        for i, tile in enumerate(tiles):
            temp_path = f"temp_tile_{i}.png"
            cv2.imwrite(temp_path, tile.image)
            
            # Detectar tipo
            try:
                tile_type = self.type_detector.detect_tile(temp_path)
            except:
                tile_type = "?"
            
            # Detectar rotación
            try:
                rotation = self.rotation_detector.detect_rotation(tile_type, temp_path)
            except:
                rotation = 0
            
            # Detectar meeple
            try:
                meeple_result = self.meeple_detector.detect_meeple(temp_path)
            except:
                meeple_result = {
                    'has_meeple': False,
                    'color': None,
                    'position': None
                }
            
            # Limpiar temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Guardar resultado
            self.results.append({
                'grid_position': (tile.grid_row, tile.grid_col),
                'tile_type': tile_type,
                'rotation': rotation,
                'has_meeple': meeple_result['has_meeple'],
                'meeple_color': meeple_result.get('color'),
                'meeple_position': meeple_result.get('position')
            })
    
    def _create_board(self) -> Board:
        """Crea la Origin Matrix (Board)"""
        # Inicializar matriz vacía
        matrix: List[List[Optional[Tile]]] = [[None] * self.cols for _ in range(self.rows)]
        
        # Llenar matriz
        for result in self.results:
            grid_row, grid_col = result['grid_position']
            
            # Convertir a índices de matriz
            matrix_row = grid_row - self.min_row
            matrix_col = grid_col - self.min_col
            
            # Verificar límites
            if not (0 <= matrix_row < self.rows and 0 <= matrix_col < self.cols):
                continue
            
            tile_type = result['tile_type']
            
            # Filtrar BLANCO y desconocidos
            if tile_type in ('BLANCO', '?'):
                continue
            
            # Procesar meeple
            meeple_info = None
            if result['has_meeple'] and result['meeple_position'] is not None:
                if result['meeple_color'] == 'blue':
                    player = 1
                elif result['meeple_color'] == 'black':
                    player = 2
                else:
                    player = 0
                
                meeple_pos = result['meeple_position'] + 1
                meeple_info = (player, meeple_pos)
            
            # Crear Tile
            tile = Tile(
                type=tile_type,
                rotation=result['rotation'],
                meeple_info=meeple_info
            )
            
            # Asignar a matriz
            matrix[matrix_row][matrix_col] = tile
        
        return Board(board=matrix)
    
    def get_dimensions(self):
        """Retorna dimensiones y offsets del tablero"""
        return {
            'rows': self.rows,
            'cols': self.cols,
            'min_row': self.min_row,
            'max_row': self.max_row,
            'min_col': self.min_col,
            'max_col': self.max_col
        }


def main():
    """Ejemplo de uso"""
    if len(sys.argv) < 2:
        print("Uso: python origin_matrix_converter.py <imagen_tablero>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: No se encontro la imagen {image_path}")
        return
    
    # Crear converter
    converter = OriginMatrixConverter()
    
    # Convertir a Origin Matrix
    board = converter.convert(image_path, num_reference_points=8)
    
    if board is None:
        print("Error: No se pudo generar la Origin Matrix")
        return
    
    # Mostrar información
    dims = converter.get_dimensions()
    print(f"Origin Matrix generada:")
    print(f"  Dimensiones: {dims['rows']}x{dims['cols']}")
    print(f"  Rango: filas [{dims['min_row']}, {dims['max_row']}], cols [{dims['min_col']}, {dims['max_col']}]")
    
    # Contar tiles
    total_tiles = sum(1 for row in board.board for tile in row if tile is not None)
    print(f"  Total de losetas: {total_tiles}")


if __name__ == "__main__":
    main()
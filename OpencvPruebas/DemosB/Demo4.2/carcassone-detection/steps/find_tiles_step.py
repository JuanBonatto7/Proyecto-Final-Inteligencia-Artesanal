"""
Paso 8: Detección y extracción de fichas del tablero.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pipeline.pipeline_step import PipelineStep
from config import Config
from utils.image_utils import crop_rectangle, order_points


class FindTilesStep(PipelineStep):
    """
    Localiza y extrae las fichas individuales del tablero.
    
    Utiliza los puntos de grilla para identificar cada celda
    y extraer la imagen de cada ficha.
    """
    
    def __init__(
        self,
        min_tile_size: int = Config.MIN_TILE_SIZE,
        max_tile_size: int = Config.MAX_TILE_SIZE
    ):
        """
        Inicializa el detector de fichas.
        
        Args:
            min_tile_size: Tamaño mínimo de ficha en píxeles
            max_tile_size: Tamaño máximo de ficha en píxeles
        """
        super().__init__("Find Tiles")
        self.min_tile_size = min_tile_size
        self.max_tile_size = max_tile_size
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encuentra y extrae las fichas del tablero.
        
        Args:
            inputs: Debe contener 'img_warped' o 'img' y 'warped_points' o 'grid_points'
            
        Returns:
            Inputs actualizado con:
                - 'tile_regions': Lista de regiones de fichas [(corners, image), ...]
                - 'tile_images': Lista de imágenes de fichas
                - 'tile_positions': Lista de posiciones en grilla [(row, col), ...]
                - 'debug_image': Visualización
        """
        # Obtener imagen y puntos
        image = inputs.get('img_warped', inputs.get('img'))
        points = inputs.get('warped_points', inputs.get('grid_points', []))
        
        if len(points) < 4:
            print("⚠ Advertencia: No hay suficientes puntos para detectar fichas")
            inputs['tile_regions'] = []
            inputs['tile_images'] = []
            inputs['tile_positions'] = []
            inputs['debug_image'] = image.copy()
            return inputs
        
        # Organizar puntos en grilla
        grid = self._organize_points_into_grid(points)
        
        # Extraer fichas
        tile_regions = []
        tile_images = []
        tile_positions = []
        
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        for row in range(rows - 1):
            for col in range(cols - 1):
                # Obtener las 4 esquinas de la celda
                corners = self._get_cell_corners(grid, row, col)
                
                if corners is None:
                    continue
                
                # Verificar si la celda tiene tamaño válido
                if not self._is_valid_tile_size(corners):
                    continue
                
                # Extraer imagen de la ficha
                tile_img = crop_rectangle(image, corners)
                
                if tile_img is not None and tile_img.size > 0:
                    tile_regions.append((corners, tile_img))
                    tile_images.append(tile_img)
                    tile_positions.append((row, col))
        
        inputs['tile_regions'] = tile_regions
        inputs['tile_images'] = tile_images
        inputs['tile_positions'] = tile_positions
        
        # Crear visualización
        debug_image = self._create_visualization(image, tile_regions)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _organize_points_into_grid(
        self,
        points: List[Tuple[float, float]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Organiza puntos en una estructura de grilla 2D.
        
        Args:
            points: Lista de puntos desordenados
            
        Returns:
            Lista de listas representando la grilla
        """
        if not points:
            return []
        
        points = np.array(points)
        
        # Ordenar por Y primero
        sorted_by_y = points[np.argsort(points[:, 1])]
        
        # Agrupar por filas (puntos con Y similar)
        rows = []
        current_row = [sorted_by_y[0]]
        y_threshold = 20  # Umbral para considerar misma fila
        
        for point in sorted_by_y[1:]:
            if abs(point[1] - current_row[0][1]) < y_threshold:
                current_row.append(point)
            else:
                # Ordenar la fila actual por X
                current_row.sort(key=lambda p: p[0])
                rows.append(current_row)
                current_row = [point]
        
        # No olvidar la última fila
        if current_row:
            current_row.sort(key=lambda p: p[0])
            rows.append(current_row)
        
        # Normalizar: asegurar que todas las filas tengan la misma cantidad de columnas
        max_cols = max(len(row) for row in rows) if rows else 0
        normalized_grid = []
        
        for row in rows:
            if len(row) < max_cols:
                # Interpolar puntos faltantes (simplificado)
                row = row + [row[-1]] * (max_cols - len(row))
            normalized_grid.append([tuple(p) for p in row])
        
        return normalized_grid
    
    def _get_cell_corners(
        self,
        grid: List[List[Tuple[float, float]]],
        row: int,
        col: int
    ) -> Optional[np.ndarray]:
        """
        Obtiene las 4 esquinas de una celda en la grilla.
        
        Args:
            grid: Grilla de puntos
            row: Índice de fila
            col: Índice de columna
            
        Returns:
            Array (4, 2) con las esquinas o None si no se puede obtener
        """
        try:
            top_left = grid[row][col]
            top_right = grid[row][col + 1]
            bottom_right = grid[row + 1][col + 1]
            bottom_left = grid[row + 1][col]
            
            corners = np.array([
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ], dtype=np.float32)
            
            return corners
            
        except (IndexError, KeyError):
            return None
    
    def _is_valid_tile_size(self, corners: np.ndarray) -> bool:
        """
        Verifica si una celda tiene un tamaño válido para ser una ficha.
        
        Args:
            corners: Esquinas de la celda
            
        Returns:
            True si el tamaño es válido
        """
        # Calcular ancho y alto aproximados
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        
        # Verificar rangos
        if width < self.min_tile_size or width > self.max_tile_size:
            return False
        if height < self.min_tile_size or height > self.max_tile_size:
            return False
        
        # Verificar que sea aproximadamente cuadrada (fichas de Carcassonne son cuadradas)
        aspect_ratio = width / height if height > 0 else 0
        if not (0.7 <= aspect_ratio <= 1.3):
            return False
        
        return True
    
    def _create_visualization(
        self,
        image: np.ndarray,
        tile_regions: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Crea visualización con las fichas detectadas."""
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        # Dibujar cada región de ficha
        for idx, (corners, tile_img) in enumerate(tile_regions):
            # Dibujar contorno
            points = corners.astype(np.int32)
            cv2.polylines(result, [points], True, (0, 255, 0), 2)
            
            # Dibujar número de ficha en el centro
            center = tuple(np.mean(points, axis=0).astype(int))
            cv2.putText(
                result,
                str(idx + 1),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
        
        # Información
        cv2.putText(
            result,
            f"Tiles detected: {len(tile_regions)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result
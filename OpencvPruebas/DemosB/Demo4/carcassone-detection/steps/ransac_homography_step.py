"""
Paso 7: Corrección de perspectiva con RANSAC.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config
from utils.geometry_utils import calculate_homography_ransac


class RANSACHomographyStep(PipelineStep):
    """
    Calcula y aplica corrección de perspectiva usando RANSAC.
    
    Esto endereza el tablero si fue fotografiado en ángulo.
    """
    
    def __init__(
        self,
        threshold: float = Config.RANSAC_THRESHOLD,
        max_iterations: int = Config.RANSAC_MAX_ITERATIONS
    ):
        """
        Inicializa el paso de homografía RANSAC.
        
        Args:
            threshold: Umbral de error para inliers
            max_iterations: Máximo número de iteraciones
        """
        super().__init__("RANSAC Homography")
        self.threshold = threshold
        self.max_iterations = max_iterations
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula homografía y corrige perspectiva.
        
        Args:
            inputs: Debe contener 'grid_points' y 'img'
            
        Returns:
            Inputs actualizado con:
                - 'homography_matrix': Matriz de homografía 3x3
                - 'img_warped': Imagen con perspectiva corregida
                - 'warped_points': Puntos transformados
                - 'debug_image': Visualización
        """
        grid_points = inputs.get('grid_points', [])
        image = inputs['img']
        
        if len(grid_points) < 4:
            print("⚠ Advertencia: No hay suficientes puntos para homografía")
            inputs['homography_matrix'] = None
            inputs['img_warped'] = image.copy()
            inputs['warped_points'] = []
            inputs['debug_image'] = image.copy()
            return inputs
        
        # Ordenar puntos en grilla
        grid_points = self._sort_grid_points(grid_points)
        
        # Calcular dimensiones del tablero corregido
        board_width, board_height = self._estimate_board_size(grid_points)
        
        # Crear puntos destino (grilla perfecta)
        dst_points = self._create_destination_grid(
            len(grid_points),
            board_width,
            board_height
        )
        
        # Calcular homografía
        src_points = np.array(grid_points, dtype=np.float32)
        dst_points = np.array(dst_points[:len(src_points)], dtype=np.float32)
        
        H = calculate_homography_ransac(
            src_points,
            dst_points,
            self.threshold,
            self.max_iterations
        )
        
        if H is None:
            print("⚠ Advertencia: No se pudo calcular homografía")
            inputs['homography_matrix'] = None
            inputs['img_warped'] = image.copy()
            inputs['warped_points'] = grid_points
            inputs['debug_image'] = image.copy()
            return inputs
        
        # Aplicar transformación
        warped = cv2.warpPerspective(
            image,
            H,
            (board_width, board_height)
        )
        
        # Transformar puntos
        warped_points = cv2.perspectiveTransform(
            src_points.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2).tolist()
        
        inputs['homography_matrix'] = H
        inputs['img_warped'] = warped
        inputs['warped_points'] = warped_points
        
        # Crear visualización
        debug_image = self._create_visualization(image, warped, grid_points)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _sort_grid_points(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Ordena puntos de grilla de arriba-abajo, izquierda-derecha."""
        points = np.array(points)
        
        # Ordenar por Y primero, luego por X
        sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
        
        return points[sorted_indices].tolist()
    
    def _estimate_board_size(
        self,
        points: List[Tuple[float, float]]
    ) -> Tuple[int, int]:
        """Estima el tamaño del tablero basado en los puntos."""
        if not points:
            return 800, 800
        
        points = np.array(points)
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        width = int(np.max(x_coords) - np.min(x_coords)) + 100
        height = int(np.max(y_coords) - np.min(y_coords)) + 100
        
        return max(width, 400), max(height, 400)
    
    def _create_destination_grid(
        self,
        num_points: int,
        width: int,
        height: int
    ) -> List[Tuple[float, float]]:
        """Crea grilla de puntos destino perfectamente alineados."""
        # Estimar número de filas y columnas
        cols = int(np.sqrt(num_points))
        rows = (num_points + cols - 1) // cols
        
        margin = 50
        cell_width = (width - 2 * margin) / (cols - 1) if cols > 1 else width
        cell_height = (height - 2 * margin) / (rows - 1) if rows > 1 else height
        
        points = []
        for row in range(rows):
            for col in range(cols):
                if len(points) >= num_points:
                    break
                x = margin + col * cell_width
                y = margin + row * cell_height
                points.append((x, y))
        
        return points
    
    def _create_visualization(
        self,
        original: np.ndarray,
        warped: np.ndarray,
        points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Crea visualización comparativa."""
        # Dibujar puntos en original
        vis_original = original.copy()
        if len(vis_original.shape) == 2:
            vis_original = cv2.cvtColor(vis_original, cv2.COLOR_GRAY2BGR)
        
        for point in points:
            x, y = map(int, point)
            cv2.circle(vis_original, (x, y), 5, (0, 255, 0), -1)
        
        # Redimensionar warped para comparación
        if warped.shape != original.shape:
            h, w = original.shape[:2]
            warped_resized = cv2.resize(warped, (w, h))
        else:
            warped_resized = warped
        
        if len(warped_resized.shape) == 2:
            warped_resized = cv2.cvtColor(warped_resized, cv2.COLOR_GRAY2BGR)
        
        # Concatenar
        comparison = np.hstack([vis_original, warped_resized])
        
        # Etiquetas
        cv2.putText(
            comparison,
            "Original with Grid",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        mid = comparison.shape[1] // 2
        cv2.putText(
            comparison,
            "Warped (Corrected)",
            (mid + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return comparison
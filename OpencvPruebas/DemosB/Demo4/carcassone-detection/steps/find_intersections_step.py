"""
Paso 6: Encontrar intersecciones de líneas.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pipeline.pipeline_step import PipelineStep
from config import Config
from utils.geometry_utils import (
    find_line_intersection,
    calculate_angle,
    cluster_points
)


class FindIntersectionsStep(PipelineStep):
    """
    Encuentra puntos de intersección entre líneas detectadas.
    
    Estos puntos representan las esquinas de las fichas.
    """
    
    def __init__(
        self,
        min_distance: float = Config.INTERSECTION_MIN_DISTANCE,
        angle_threshold: float = Config.INTERSECTION_ANGLE_THRESHOLD
    ):
        """
        Inicializa el buscador de intersecciones.
        
        Args:
            min_distance: Distancia mínima para agrupar puntos
            angle_threshold: Umbral de ángulo para considerar intersección válida
        """
        super().__init__("Find Intersections")
        self.min_distance = min_distance
        self.angle_threshold = angle_threshold
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encuentra intersecciones entre líneas.
        
        Args:
            inputs: Debe contener 'horizontal_lines' y 'vertical_lines'
            
        Returns:
            Inputs actualizado con:
                - 'intersections': Lista de puntos de intersección
                - 'grid_points': Puntos agrupados formando grilla
                - 'debug_image': Visualización
        """
        horizontal_lines = inputs.get('horizontal_lines', [])
        vertical_lines = inputs.get('vertical_lines', [])
        
        # Encontrar todas las intersecciones
        intersections = self._find_all_intersections(
            horizontal_lines,
            vertical_lines
        )
        
        # Agrupar puntos cercanos
        if intersections:
            grid_points = cluster_points(intersections, self.min_distance)
        else:
            grid_points = []
        
        inputs['intersections'] = intersections
        inputs['grid_points'] = grid_points
        
        # Crear visualización
        original_img = inputs.get('img', inputs.get('original_img'))
        debug_image = self._create_visualization(
            original_img,
            horizontal_lines,
            vertical_lines,
            grid_points
        )
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _find_all_intersections(
        self,
        horizontal_lines: List[List[float]],
        vertical_lines: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """
        Encuentra todas las intersecciones entre líneas H y V.
        
        Args:
            horizontal_lines: Lista de líneas horizontales
            vertical_lines: Lista de líneas verticales
            
        Returns:
            Lista de puntos de intersección
        """
        intersections = []
        
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = find_line_intersection(h_line, v_line)
                
                if intersection is not None:
                    # Verificar que la intersección esté dentro de los segmentos
                    if self._is_point_on_segments(intersection, h_line, v_line):
                        intersections.append(intersection)
        
        return intersections
    
    def _is_point_on_segments(
        self,
        point: Tuple[float, float],
        line1: List[float],
        line2: List[float],
        tolerance: float = 10.0
    ) -> bool:
        """
        Verifica si un punto está cerca de ambos segmentos de línea.
        
        Args:
            point: Punto a verificar
            line1: Primera línea
            line2: Segunda línea
            tolerance: Tolerancia en píxeles
            
        Returns:
            True si el punto está sobre ambos segmentos
        """
        x, y = point
        
        # Verificar line1
        x1, y1, x2, y2 = line1
        if not (min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and
                min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance):
            return False
        
        # Verificar line2
        x1, y1, x2, y2 = line2
        if not (min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and
                min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance):
            return False
        
        return True
    
    def _create_visualization(
        self,
        image: np.ndarray,
        horizontal_lines: List[List[float]],
        vertical_lines: List[List[float]],
        grid_points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Crea visualización con líneas e intersecciones."""
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        # Dibujar líneas horizontales en azul
        for line in horizontal_lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Dibujar líneas verticales en verde
        for line in vertical_lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Dibujar intersecciones en rojo
        for point in grid_points:
            x, y = map(int, point)
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
        
        # Información
        cv2.putText(
            result,
            f"Intersections: {len(grid_points)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result
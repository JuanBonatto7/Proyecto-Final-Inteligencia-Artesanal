"""
Paso 5: Transformada de Hough para detección de líneas (MEJORADO).
"""

import cv2
import numpy as np
from typing import Dict, Any
from pipeline.pipeline_step import PipelineStep
from config import Config
from utils.geometry_utils import filter_lines_by_length
from utils.visualization_utils import draw_2point_line_segments

class HoughLineTransformStep(PipelineStep):
    """
    Detecta líneas usando la Transformada Probabilística de Hough.
    Retorna tanto líneas polares como segmentos de línea.
    """
    
    def __init__(
        self,
        rho: float = Config.HOUGH_RHO,
        theta: float = Config.HOUGH_THETA,
        threshold: int = Config.HOUGH_THRESHOLD,
        min_line_length: int = Config.HOUGH_MIN_LINE_LENGTH,
        max_line_gap: int = Config.HOUGH_MAX_LINE_GAP
    ):
        super().__init__("Hough Line Transform")
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta líneas usando HoughLinesP (probabilística).
        
        Args:
            inputs: Debe contener 'img_dilated' o 'img_edges'
            
        Returns:
            Inputs actualizado con:
                - 'line_segments': Segmentos de línea (x1, y1, x2, y2)
                - 'debug_image': Visualización
        """
        edges = inputs.get('img_dilated', inputs['img_edges'])
        
        # Detectar líneas con HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is not None:
            lines = np.squeeze(lines, axis=1)
            
            # Filtrar líneas muy cortas
            lines = filter_lines_by_length(lines, min_length=30)
        else:
            lines = np.array([])
        
        inputs['line_segments'] = lines
        
        # Crear visualización
        original_img = inputs.get('img', inputs.get('original_img'))
        debug_image = self._create_visualization(original_img, lines)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _create_visualization(
        self,
        image: np.ndarray,
        line_segments: np.ndarray
    ) -> np.ndarray:
        """Crea visualización con líneas detectadas."""
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        # Dibujar segmentos de línea
        if line_segments is not None and len(line_segments) > 0:
            draw_2point_line_segments(result, line_segments, color=(0, 255, 0), thickness=2)
        
        # Información
        num_lines = len(line_segments) if line_segments is not None else 0
        cv2.putText(
            result,
            f"Line segments detected: {num_lines}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result
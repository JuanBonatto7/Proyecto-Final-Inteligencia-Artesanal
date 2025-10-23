"""
Paso 5: Transformada de Hough para detección de líneas.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from pipeline.pipeline_step import PipelineStep
from config import Config


class HoughLineTransformStep(PipelineStep):
    """
    Detecta líneas usando la Transformada Probabilística de Hough.
    
    Estas líneas representan los bordes de las fichas en el tablero.
    """
    
    def __init__(
        self,
        rho: float = Config.HOUGH_RHO,
        theta: float = Config.HOUGH_THETA,
        threshold: int = Config.HOUGH_THRESHOLD,
        min_line_length: int = Config.HOUGH_MIN_LINE_LENGTH,
        max_line_gap: int = Config.HOUGH_MAX_LINE_GAP
    ):
        """
        Inicializa el detector de líneas Hough.
        
        Args:
            rho: Resolución de distancia en píxeles
            theta: Resolución angular en radianes
            threshold: Umbral de acumulador
            min_line_length: Longitud mínima de línea
            max_line_gap: Máximo gap permitido entre segmentos
        """
        super().__init__("Hough Line Transform")
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta líneas en la imagen de bordes.
        
        Args:
            inputs: Debe contener 'img_dilated' o 'img_edges'
            
        Returns:
            Inputs actualizado con:
                - 'lines': Lista de líneas detectadas [(x1,y1,x2,y2), ...]
                - 'horizontal_lines': Líneas horizontales
                - 'vertical_lines': Líneas verticales
                - 'debug_image': Visualización
        """
        edges = inputs.get('img_dilated', inputs['img_edges'])
        
        # Detectar líneas
        lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            lines = []
        else:
            lines = lines.reshape(-1, 4).tolist()
        
        # Separar líneas horizontales y verticales
        horizontal_lines, vertical_lines = self._separate_lines(lines)
        
        inputs['lines'] = lines
        inputs['horizontal_lines'] = horizontal_lines
        inputs['vertical_lines'] = vertical_lines
        
        # Crear visualización
        original_img = inputs.get('img', inputs.get('original_img'))
        debug_image = self._create_visualization(original_img, lines)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _separate_lines(
        self,
        lines: List[List[float]],
        angle_threshold: float = 15.0
    ) -> tuple:
        """
        Separa líneas en horizontales y verticales.
        
        Args:
            lines: Lista de líneas
            angle_threshold: Umbral de ángulo en grados
            
        Returns:
            Tupla (horizontal_lines, vertical_lines)
        """
        horizontal = []
        vertical = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calcular ángulo
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
            
            # Clasificar
            if angle < angle_threshold:
                horizontal.append(line)
            elif angle > (90 - angle_threshold):
                vertical.append(line)
        
        return horizontal, vertical
    
    def _create_visualization(
        self,
        image: np.ndarray,
        lines: List[List[float]]
    ) -> np.ndarray:
        """Crea visualización con líneas detectadas."""
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        # Dibujar todas las líneas
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Información
        cv2.putText(
            result,
            f"Lines detected: {len(lines)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result
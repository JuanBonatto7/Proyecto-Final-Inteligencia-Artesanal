"""
Paso 4: Dilatación de bordes.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config


class DilateStep(PipelineStep):
    """
    Aplica dilatación morfológica a los bordes detectados.
    
    Esto conecta bordes fragmentados y fortalece las líneas.
    """
    
    def __init__(
        self,
        kernel_size: Tuple[int, int] = Config.DILATE_KERNEL_SIZE,
        iterations: int = Config.DILATE_ITERATIONS
    ):
        """
        Inicializa el paso de dilatación.
        
        Args:
            kernel_size: Tamaño del kernel estructurante
            iterations: Número de veces que se aplica la dilatación
        """
        super().__init__("Dilate Edges")
        self.kernel_size = kernel_size
        self.iterations = iterations
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica dilatación a los bordes.
        
        Args:
            inputs: Debe contener 'img_edges'
            
        Returns:
            Inputs actualizado con:
                - 'img_dilated': Bordes dilatados
                - 'debug_image': Visualización
        """
        edges = inputs['img_edges']
        
        # Crear kernel estructurante
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.kernel_size
        )
        
        # Aplicar dilatación
        dilated = cv2.dilate(
            edges,
            kernel,
            iterations=self.iterations
        )
        
        inputs['img_dilated'] = dilated
        
        # Crear visualización
        debug_image = self._create_comparison(edges, dilated)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _create_comparison(
        self,
        edges: np.ndarray,
        dilated: np.ndarray
    ) -> np.ndarray:
        """Crea comparación antes/después de dilatación."""
        # Convertir a BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        
        # Colorear: edges en rojo, dilated en verde
        edges_colored = edges_bgr.copy()
        edges_colored[edges > 0] = [0, 0, 255]
        
        dilated_colored = dilated_bgr.copy()
        dilated_colored[dilated > 0] = [0, 255, 0]
        
        # Concatenar
        comparison = np.hstack([edges_colored, dilated_colored])
        
        # Etiquetas
        cv2.putText(
            comparison,
            "Original Edges",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        mid_point = comparison.shape[1] // 2
        cv2.putText(
            comparison,
            "Dilated Edges",
            (mid_point + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return comparison
    
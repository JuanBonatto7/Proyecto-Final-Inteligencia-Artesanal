"""
Paso 3: Detección de bordes con Canny.
"""

import cv2
import numpy as np
from typing import Dict, Any
from pipeline.pipeline_step import PipelineStep
from config import Config


class CannyEdgeDetectorStep(PipelineStep):
    """
    Detecta bordes en la imagen usando el algoritmo Canny.
    
    Este paso es crucial para identificar los límites de las fichas.
    """
    
    def __init__(
        self,
        threshold1: int = Config.CANNY_THRESHOLD_1,
        threshold2: int = Config.CANNY_THRESHOLD_2,
        aperture_size: int = Config.CANNY_APERTURE_SIZE
    ):
        """
        Inicializa el detector de bordes Canny.
        
        Args:
            threshold1: Umbral inferior para histéresis
            threshold2: Umbral superior para histéresis
            aperture_size: Tamaño de apertura para operador Sobel
        """
        super().__init__("Canny Edge Detection")
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta bordes en la imagen.
        
        Args:
            inputs: Debe contener 'img_blurred' o 'img'
            
        Returns:
            Inputs actualizado con:
                - 'img_edges': Imagen de bordes
                - 'img_gray': Imagen en escala de grises
                - 'debug_image': Visualización
        """
        # Usar imagen desenfocada si existe, sino la original
        image = inputs.get('img_blurred', inputs['img'])
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detectar bordes
        edges = cv2.Canny(
            gray,
            self.threshold1,
            self.threshold2,
            apertureSize=self.aperture_size
        )
        
        inputs['img_edges'] = edges
        inputs['img_gray'] = gray
        
        # Crear visualización
        debug_image = self._create_visualization(image, edges)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _create_visualization(
        self,
        original: np.ndarray,
        edges: np.ndarray
    ) -> np.ndarray:
        """Crea visualización de bordes detectados."""
        # Convertir edges a BGR para visualización
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Asegurar que original sea BGR
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Superponer bordes en verde sobre la imagen original
        overlay = original.copy()
        overlay[edges > 0] = [0, 255, 0]
        
        # Combinar original y overlay con transparencia
        result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        
        # Añadir información
        edge_count = np.count_nonzero(edges)
        cv2.putText(
            result,
            f"Edges detected: {edge_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result
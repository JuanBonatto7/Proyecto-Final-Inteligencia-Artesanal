"""
Paso 1: Redimensionamiento de imagen.
"""

import cv2
import numpy as np
from typing import Dict, Any
from pipeline.pipeline_step import PipelineStep
from config import Config


class ResizeStep(PipelineStep):
    """
    Redimensiona la imagen de entrada manteniendo el aspect ratio.
    
    Esto optimiza el procesamiento y estandariza el tamaño de entrada.
    """
    
    def __init__(self, max_width: int = Config.MAX_IMAGE_WIDTH):
        """
        Inicializa el paso de redimensionamiento.
        
        Args:
            max_width: Ancho máximo de la imagen resultante
        """
        super().__init__("Resize Image")
        self.max_width = max_width
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redimensiona la imagen de entrada.
        
        Args:
            inputs: Debe contener 'img' con la imagen original
            
        Returns:
            Inputs actualizado con:
                - 'img': Imagen redimensionada
                - 'original_img': Imagen original
                - 'scale_factor': Factor de escala aplicado
                - 'original_size': Tamaño original (height, width)
                - 'debug_image': Imagen para visualización
        """
        image = inputs['img']
        original_size = image.shape[:2]
        
        height, width = original_size
        
        if width <= self.max_width:
            scale_factor = 1.0
            resized_image = image.copy()
        else:
            scale_factor = self.max_width / width
            new_width = self.max_width
            new_height = int(height * scale_factor)
            
            resized_image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
        
        # Actualizar inputs
        inputs['original_img'] = image
        inputs['img'] = resized_image
        inputs['scale_factor'] = scale_factor
        inputs['original_size'] = original_size
        
        # Crear imagen de debug
        debug_image = resized_image.copy()
        self._add_info_text(
            debug_image,
            f"Original: {width}x{height}",
            f"Resized: {resized_image.shape[1]}x{resized_image.shape[0]}",
            f"Scale: {scale_factor:.2f}"
        )
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _add_info_text(self, image: np.ndarray, *texts: str) -> None:
        """Agrega texto informativo a la imagen."""
        y_offset = 30
        for i, text in enumerate(texts):
            cv2.putText(
                image,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
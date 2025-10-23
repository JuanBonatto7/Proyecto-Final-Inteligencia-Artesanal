"""
Paso 2: Desenfoque Gaussiano.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config


class BlurStep(PipelineStep):
    """
    Aplica desenfoque gaussiano para reducir ruido.
    
    Esto ayuda a mejorar la detección de bordes eliminando
    ruido de alta frecuencia.
    """
    
    def __init__(
        self,
        kernel_size: Tuple[int, int] = Config.BLUR_KERNEL_SIZE,
        sigma: float = Config.BLUR_SIGMA
    ):
        """
        Inicializa el paso de desenfoque.
        
        Args:
            kernel_size: Tamaño del kernel (debe ser impar)
            sigma: Desviación estándar del kernel gaussiano
        """
        super().__init__("Gaussian Blur")
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Validar que kernel_size sea impar
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("kernel_size debe tener valores impares")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica desenfoque gaussiano a la imagen.
        
        Args:
            inputs: Debe contener 'img'
            
        Returns:
            Inputs actualizado con:
                - 'img_blurred': Imagen desenfocada
                - 'debug_image': Visualización
        """
        image = inputs['img']
        
        # Aplicar desenfoque gaussiano
        blurred = cv2.GaussianBlur(
            image,
            self.kernel_size,
            self.sigma
        )
        
        inputs['img_blurred'] = blurred
        
        # Crear visualización comparativa
        debug_image = self._create_comparison(image, blurred)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _create_comparison(
        self,
        original: np.ndarray,
        blurred: np.ndarray
    ) -> np.ndarray:
        """Crea imagen comparativa lado a lado."""
        # Asegurar que ambas imágenes tengan el mismo número de canales
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(blurred.shape) == 2:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        
        # Concatenar horizontalmente
        comparison = np.hstack([original, blurred])
        
        # Agregar etiquetas
        height = comparison.shape[0]
        mid_point = comparison.shape[1] // 2
        
        cv2.putText(
            comparison,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            comparison,
            "Blurred",
            (mid_point + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return comparison
"""
Paso 2: Desenfoque Gaussiano.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config


class BlurStep(PipelineStep):
    
    def __init__(
        self,
        kernel_size: Tuple[int, int] = Config.BLUR_KERNEL_SIZE,
        sigma: float = Config.BLUR_SIGMA
    ):
        super().__init__("Gaussian Blur")
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image = inputs['img']
        
        blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        
        inputs['img_blurred'] = blurred
        inputs['debug_image'] = blurred.copy()
        
        return inputs
"""
Paso 1: Redimensionamiento de imagen.
"""

import cv2
import numpy as np
from typing import Dict, Any
from pipeline.pipeline_step import PipelineStep
from config import Config


class ResizeStep(PipelineStep):
    
    def __init__(self, max_width: int = Config.MAX_IMAGE_WIDTH):
        super().__init__("Resize Image")
        self.max_width = max_width
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
        
        inputs['original_img'] = image
        inputs['img'] = resized_image
        inputs['scale_factor'] = scale_factor
        inputs['original_size'] = original_size
        inputs['debug_image'] = resized_image.copy()
        
        return inputs
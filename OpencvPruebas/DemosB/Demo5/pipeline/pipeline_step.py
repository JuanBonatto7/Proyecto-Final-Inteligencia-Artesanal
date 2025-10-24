"""
Clase base abstracta para los pasos del pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class PipelineStep(ABC):
    """
    Clase base abstracta para todos los pasos del pipeline.
    
    Cada paso debe implementar el método process() que recibe
    un diccionario de entradas y retorna un diccionario actualizado.
    """
    
    def __init__(self, name: str):
        """
        Inicializa el paso del pipeline.
        
        Args:
            name: Nombre descriptivo del paso
        """
        self.name = name
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa los datos de entrada.
        
        Args:
            inputs: Diccionario con los datos de entrada
            
        Returns:
            Diccionario con los datos de entrada más los resultados del paso
            
        Raises:
            NotImplementedError: Si no se implementa en la clase hija
        """
        raise NotImplementedError("Debe implementar el método process()")
    
    def get_visualization_image(self, inputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Genera una imagen para visualización del paso.
        
        Args:
            inputs: Diccionario con los datos procesados
            
        Returns:
            Imagen para visualización o None si no aplica
        """
        return inputs.get('debug_imag' \
        'e', None)
    
    def __str__(self) -> str:
        """Retorna el nombre del paso."""
        return self.name
    
    def __repr__(self) -> str:
        """Retorna representación del paso."""
        return f"{self.__class__.__name__}(name='{self.name}')"
"""
Modelo de datos para las fichas de Carcassonne.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class Tile:
    """
    Representa una ficha del juego Carcassonne.
    
    Attributes:
        tile_type: Letra identificadora de la ficha (A-Y)
        rotation: Rotación de la ficha (0, 1, 2, 3)
        position: Posición (row, col) en la matriz del tablero
        corners: Coordenadas de las 4 esquinas de la ficha
        confidence: Nivel de confianza de la clasificación (0-1)
        image: Imagen recortada de la ficha
    """
    
    tile_type: str
    rotation: int
    position: Tuple[int, int]
    corners: Optional[np.ndarray] = None
    confidence: float = 0.0
    image: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validación de datos después de inicialización."""
        if not ('A' <= self.tile_type <= 'Y'):
            raise ValueError(f"tile_type debe estar entre A-Y, recibido: {self.tile_type}")
        
        if self.rotation not in [0, 1, 2, 3]:
            raise ValueError(f"rotation debe ser 0, 1, 2 o 3, recibido: {self.rotation}")
        
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError(f"position debe ser tupla (row, col)")
    
    def get_identifier(self) -> str:
        """
        Retorna identificador único de la ficha.
        
        Returns:
            String en formato "TIPO_ROTACION" (ej: "A_2")
        """
        return f"{self.tile_type}_{self.rotation}"
    
    def rotate_right(self) -> 'Tile':
        """
        Retorna una nueva ficha rotada 90° a la derecha.
        
        Returns:
            Nueva instancia de Tile con rotación incrementada
        """
        new_rotation = (self.rotation + 1) % 4
        return Tile(
            tile_type=self.tile_type,
            rotation=new_rotation,
            position=self.position,
            corners=self.corners,
            confidence=self.confidence,
            image=self.image
        )
    
    def __str__(self) -> str:
        """Representación en string de la ficha."""
        return f"Tile({self.tile_type}, rot={self.rotation}, pos={self.position})"
    
    def __repr__(self) -> str:
        """Representación detallada de la ficha."""
        return (f"Tile(type={self.tile_type}, rotation={self.rotation}, "
                f"position={self.position}, confidence={self.confidence:.2f})")
    
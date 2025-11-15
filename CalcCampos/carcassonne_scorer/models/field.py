# ============================================================================
# models/field.py
# ============================================================================

from dataclasses import dataclass
from typing import Set, Optional
import numpy as np

from config.settings import Config

@dataclass
class Field:
    """Representa un campo en el tablero."""
    
    id: int
    mask: np.ndarray
    area: int
    meeples_p1: int = 0
    meeples_p2: int = 0
    closed_castles_touching: int = 0  # Castillos CERRADOS que tocan este campo
    is_valid: bool = True
    
    @property
    def owner(self) -> Optional[int]:
        """Determina el dueño del campo."""
        if self.meeples_p1 > self.meeples_p2:
            return 1
        elif self.meeples_p2 > self.meeples_p1:
            return 2
        elif self.meeples_p1 > 0 and self.meeples_p1 == self.meeples_p2:
            return 0  # Empate
        return None
    
    @property
    def points(self) -> int:
        """Calcula los puntos del campo."""
        return self.closed_castles_touching * Config.POINTS_PER_CLOSED_CASTLE
    
    def __repr__(self):
        owner_str = f"P{self.owner}" if self.owner else "None"
        if self.owner == 0:
            owner_str = "TIE"
        return (f"Field(id={self.id}, area={self.area}, "
                f"meeples=({self.meeples_p1},{self.meeples_p2}), "
                f"owner={owner_str}, closed_castles={self.closed_castles_touching}, "
                f"points={self.points})")
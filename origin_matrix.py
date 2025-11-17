"""
Módulo que define las estructuras de datos para el tablero de Carcassonne.
Contiene las clases Tile y Board para representar losetas y el tablero completo.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional

__all__ = ["Tile", "Board"]

@dataclass
class Tile:
    """Representa una loseta del juego Carcassonne"""
    type: str              # Letra de la loseta (A-X)
    rotation: int          # Rotación en grados: 0, 90, 180, 270
    meeple_info: Optional[Tuple[int, int]]  # None o (jugador, posición 1..9)

@dataclass
class Board:
    """Representa el tablero completo del juego"""
    board: List[List[Optional[Tile]]]  # Matriz de tiles (None = casilla vacía)

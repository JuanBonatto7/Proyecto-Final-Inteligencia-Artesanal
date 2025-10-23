import random
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class Tile:
    """Representa una loseta del tablero de Carcassonne"""
    nombre: str
    orientacion: int  # 0, 1, 2, 3 (rotación en 90°)
    meeple: Tuple[int, int]  # (jugador, posición)


# --- Diccionario con las losetas base ---
TILESET = {
    "A": {"borders": {"N": "field", "E": "field", "S": "road", "W": "field"}, "count": 2},
    "B": {"borders": {"N": "field", "E": "field", "S": "field", "W": "field"}, "count": 4},
    "C": {"borders": {"N": "city", "E": "city", "S": "city", "W": "city"}, "count": 1},
    "D": {"borders": {"N": "city", "E": "city", "S": "field", "W": "city"}, "count": 4},
    "E": {"borders": {"N": "city", "E": "field", "S": "field", "W": "city"}, "count": 5},
    "F": {"borders": {"N": "city", "E": "city", "S": "road", "W": "city"}, "count": 2},
    "G": {"borders": {"N": "city", "E": "field", "S": "field", "W": "field"}, "count": 1},
    "H": {"borders": {"N": "city", "E": "city", "S": "field", "W": "field"}, "count": 3},
    "I": {"borders": {"N": "city", "E": "field", "S": "field", "W": "city"}, "count": 2},
    "J": {"borders": {"N": "city", "E": "field", "S": "road", "W": "field"}, "count": 3},
    "K": {"borders": {"N": "city", "E": "road", "S": "field", "W": "field"}, "count": 3},
    "M": {"borders": {"N": "city", "E": "road", "S": "road", "W": "field"}, "count": 3},
    "N": {"borders": {"N": "city", "E": "city", "S": "field", "W": "road"}, "count": 3},
    "O": {"borders": {"N": "city", "E": "road", "S": "field", "W": "city"}, "count": 2},
    "P": {"borders": {"N": "city", "E": "field", "S": "field", "W": "road"}, "count": 2},
    "Q": {"borders": {"N": "city", "E": "road", "S": "city", "W": "road"}, "count": 3},
    "R": {"borders": {"N": "city", "E": "city", "S": "city", "W": "city"}, "count": 1},
    "S": {"borders": {"N": "city", "E": "field", "S": "city", "W": "field"}, "count": 3},
    "T": {"borders": {"N": "city", "E": "road", "S": "city", "W": "field"}, "count": 2},
    "U": {"borders": {"N": "city", "E": "road", "S": "road", "W": "road"}, "count": 1},
    "V": {"borders": {"N": "field", "E": "road", "S": "field", "W": "road"}, "count": 8},
    "W": {"borders": {"N": "field", "E": "road", "S": "road", "W": "field"}, "count": 9},
    "X": {"borders": {"N": "road", "E": "road", "S": "road", "W": "road"}, "count": 4},
    "Y": {"borders": {"N": "road", "E": "road", "S": "road", "W": "road"}, "count": 1},
}


# --- Funciones auxiliares ---

def rotar_bordes(bordes: Dict[str, str], rotacion: int) -> Dict[str, str]:
    """Rota los bordes según la orientación (0–3)."""
    lados = ["N", "E", "S", "W"]
    rotados = {}
    for i, lado in enumerate(lados):
        rotados[lados[(i + rotacion) % 4]] = bordes[lado]
    return rotados


def son_compatibles(tile1_bordes, tile2_bordes, lado1, lado2):
    """Verifica si dos losetas encajan por los bordes indicados."""
    return tile1_bordes[lado1] == tile2_bordes[lado2]


import random

def generar_tablero(n=5):
    """Genera un tablero cuadrado n x n válido respetando el inventario,
    las reglas del juego y los límites de meeples por jugador."""
    tablero = [[None for _ in range(n)] for _ in range(n)]
    centro = n // 2

    # Inventario de losetas disponibles
    disponibles = {nombre: data["count"] for nombre, data in TILESET.items()}

    # Asigna entre 3 y 5 meeples por jugador
    meeples_totales = {1: random.randint(3, 5), 2: random.randint(3, 5)}
    meeples_usados = {1: 0, 2: 0}

    # Colocar una loseta inicial en el centro (sin meeple)
    nombre_inicial = random.choice(list(disponibles.keys()))
    orientacion_inicial = random.choice([0, 1, 2, 3])
    tablero[centro][centro] = Tile(nombre_inicial, orientacion_inicial, (0, 0))
    disponibles[nombre_inicial] -= 1

    direcciones = {
        "N": (-1, 0, "S"),
        "S": (1, 0, "N"),
        "E": (0, 1, "W"),
        "W": (0, -1, "E")
    }

    for i in range(n):
        for j in range(n):
            if tablero[i][j] is not None:
                continue

            # Buscar vecinos ya colocados
            vecinos = []
            for lado, (dx, dy, opuesto) in direcciones.items():
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < n and tablero[ni][nj] is not None:
                    vecinos.append((lado, tablero[ni][nj], opuesto))

            if not vecinos:
                continue  # sin vecinos, no colocamos nada todavía

            # Intentar colocar una loseta compatible
            tipos_disponibles = [k for k, v in disponibles.items() if v > 0]
            random.shuffle(tipos_disponibles)

            colocada = False
            for nombre in tipos_disponibles:
                # Rotación aleatoria entre 0-3
                rotaciones = [0, 1, 2, 3]
                random.shuffle(rotaciones)

                for rot in rotaciones:
                    bordes_rot = rotar_bordes(TILESET[nombre]["borders"], rot)
                    if all(
                        son_compatibles(
                            bordes_rot,
                            rotar_bordes(TILESET[v[1].nombre]["borders"], v[1].orientacion),
                            v[0], v[2]
                        )
                        for v in vecinos
                    ):
                        # Elegir jugador y verificar meeples disponibles
                        jugador = random.choice([1, 2])
                        if meeples_usados[jugador] < meeples_totales[jugador]:
                            meeple = (jugador, random.randint(0, 8))
                            meeples_usados[jugador] += 1
                        else:
                            meeple = (0, 0)  # sin meeple

                        tablero[i][j] = Tile(nombre, rot, meeple)
                        disponibles[nombre] -= 1
                        colocada = True
                        break
                if colocada:
                    break

    return tablero, disponibles, meeples_usados, meeples_totales



def mostrar_tablero(tablero):
    print("\n=== TABLERO GENERADO ===")
    for fila in tablero:
        linea = []
        for tile in fila:
            if tile is None:
                linea.append("[   ]")
            else:
                j, pos = tile.meeple
                texto = f"{tile.nombre}{tile.orientacion}"
                if j != 0:
                    texto += f"(J{j})"
                linea.append(f"[{texto:^6}]")
        print(" ".join(linea))

# ==========================================================
# EJEMPLO DE USO
# ==========================================================
if __name__ == "__main__":
    tablero, restantes, usados, totales = generar_tablero(5)

    mostrar_tablero(tablero)

    print("\n=== INFORME FINAL ===")
    print("Meeples totales por jugador:", totales)
    print("Meeples usados:", usados)
    print("\nInventario restante:")
    for k, v in restantes.items():
        print(f"  {k}: {v} losetas restantes")
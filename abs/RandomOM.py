import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

@dataclass
class Tile:
    """Representa una loseta del tablero de Carcassonne"""
    nombre: str
    orientacion: int  # 0, 1, 2, 3 (rotación en múltiplos de 90°)
    meeple: Tuple[int, int]  # (jugador, posición)


# =============================================================================
# TILESET CORREGIDO - 60 LOSETAS TOTALES
# =============================================================================

TILESET = {
    "A": {"borders": {"N": "field", "E": "field", "S": "road", "W": "field"}, "count": 2},
    "B": {"borders": {"N": "field", "E": "field", "S": "field", "W": "field"}, "count": 4},
    "C": {"borders": {"N": "city", "E": "city", "S": "city", "W": "city"}, "count": 1},
    "D": {"borders": {"N": "road", "E": "city", "S": "road", "W": "field"}, "count": 4},
    "E": {"borders": {"N": "city", "E": "field", "S": "field", "W": "field"}, "count": 5},
    "F": {"borders": {"N": "field", "E": "city", "S": "field", "W": "city"}, "count": 2},
    "G": {"borders": {"N": "city", "E": "field", "S": "city", "W": "field"}, "count": 1},
    "H": {"borders": {"N": "field", "E": "city", "S": "field", "W": "city"}, "count": 3},
    "I": {"borders": {"N": "field", "E": "city", "S": "city", "W": "field"}, "count": 2},
    "J": {"borders": {"N": "city", "E": "road", "S": "road", "W": "field"}, "count": 3},
    "K": {"borders": {"N": "road", "E": "city", "S": "field", "W": "road"}, "count": 3},
    "L": {"borders": {"N": "road", "E": "city", "S": "road", "W": "road"}, "count": 3},
    "M": {"borders": {"N": "city", "E": "field", "S": "field", "W": "city"}, "count": 3},
    "N": {"borders": {"N": "city", "E": "field", "S": "field", "W": "city"}, "count": 3},
    "O": {"borders": {"N": "city", "E": "road", "S": "road", "W": "city"}, "count": 2},
    "P": {"borders": {"N": "city", "E": "road", "S": "road", "W": "city"}, "count": 3},
    "Q": {"borders": {"N": "city", "E": "city", "S": "field", "W": "city"}, "count": 1},
    "R": {"borders": {"N": "city", "E": "city", "S": "field", "W": "city"}, "count": 3},
    "S": {"borders": {"N": "city", "E": "city", "S": "road", "W": "city"}, "count": 2},
    "T": {"borders": {"N": "city", "E": "city", "S": "road", "W": "city"}, "count": 1},
    "U": {"borders": {"N": "road", "E": "field", "S": "road", "W": "field"}, "count": 8},
    "V": {"borders": {"N": "field", "E": "field", "S": "road", "W": "road"}, "count": 9},
    "W": {"borders": {"N": "field", "E": "road", "S": "road", "W": "road"}, "count": 4},
    "X": {"borders": {"N": "road", "E": "road", "S": "road", "W": "road"}, "count": 1},
}

TOTAL_LOSETAS = sum(data["count"] for data in TILESET.values())


# =============================================================================
# FUNCIONES DE ROTACIÓN
# =============================================================================

def rotar_bordes(bordes: Dict[str, str], rotacion: int) -> Dict[str, str]:
    """
    Rota los bordes de una loseta.
    
    Rotación ANTIHORARIA:
    - 0: sin rotación
    - 1: 90° antihorario (N→W, E→N, S→E, W→S)
    - 2: 180° (N→S, E→W, S→N, W→E)
    - 3: 270° antihorario (N→E, E→S, S→W, W→N)
    """
    if rotacion == 0:
        return bordes.copy()
    elif rotacion == 1:
        return {"N": bordes["E"], "E": bordes["S"], "S": bordes["W"], "W": bordes["N"]}
    elif rotacion == 2:
        return {"N": bordes["S"], "E": bordes["W"], "S": bordes["N"], "W": bordes["E"]}
    elif rotacion == 3:
        return {"N": bordes["W"], "E": bordes["N"], "S": bordes["E"], "W": bordes["S"]}
    return bordes.copy()


# =============================================================================
# FUNCIONES DE COMPATIBILIDAD
# =============================================================================

def obtener_vecinos_con_direccion(tablero: List[List[Optional[Tile]]], 
                                   fila: int, col: int) -> List[Tuple[str, Tile, str]]:
    """
    Obtiene los vecinos de una posición con información direccional.
    
    Returns:
        Lista de tuplas: (mi_direccion, loseta_vecina, direccion_del_vecino_hacia_mi)
        
    Ejemplo: Si hay un vecino al Norte de (5,5):
        - mi_direccion = "N" (desde mi posición, el vecino está al Norte)
        - direccion_del_vecino_hacia_mi = "S" (desde el vecino, yo estoy al Sur)
    """
    n = len(tablero)
    vecinos = []
    
    # Norte: fila-1
    if fila > 0 and tablero[fila-1][col] is not None:
        vecinos.append(("N", tablero[fila-1][col], "S"))
    
    # Sur: fila+1
    if fila < n-1 and tablero[fila+1][col] is not None:
        vecinos.append(("S", tablero[fila+1][col], "N"))
    
    # Este: col+1
    if col < n-1 and tablero[fila][col+1] is not None:
        vecinos.append(("E", tablero[fila][col+1], "W"))
    
    # Oeste: col-1
    if col > 0 and tablero[fila][col-1] is not None:
        vecinos.append(("W", tablero[fila][col-1], "E"))
    
    return vecinos


def verificar_compatibilidad(nombre_loseta: str, 
                             orientacion: int,
                             tablero: List[List[Optional[Tile]]], 
                             fila: int, 
                             col: int) -> bool:
    """
    Verifica si una loseta con cierta orientación puede colocarse en una posición.
    
    REGLA DE CARCASSONNE:
    Cuando dos losetas se tocan, los bordes que comparten deben ser del mismo tipo.
    
    Ejemplo:
        Loseta A en (5,5) tiene al Norte "city"
        Loseta B en (4,5) tiene al Sur "city"
        → ✅ Compatible
        
        Loseta A en (5,5) tiene al Norte "city"
        Loseta B en (4,5) tiene al Sur "road"
        → ❌ NO compatible
    
    Args:
        nombre_loseta: Tipo de loseta a colocar
        orientacion: Rotación de la loseta (0-3)
        tablero: Tablero actual
        fila, col: Posición donde se quiere colocar
    
    Returns:
        True si la loseta encaja con TODOS sus vecinos
    """
    # Obtener vecinos
    vecinos = obtener_vecinos_con_direccion(tablero, fila, col)
    
    # Si no hay vecinos, no podemos colocar (excepto la primera loseta)
    if len(vecinos) == 0:
        return False
    
    # Obtener los bordes de la loseta propuesta (con rotación aplicada)
    bordes_propuesta = rotar_bordes(TILESET[nombre_loseta]["borders"], orientacion)
    
    # Verificar cada vecino
    for mi_direccion, loseta_vecina, dir_vecino_hacia_mi in vecinos:
        # Obtener bordes del vecino (con su rotación aplicada)
        bordes_vecino = rotar_bordes(
            TILESET[loseta_vecina.nombre]["borders"],
            loseta_vecina.orientacion
        )
        
        # Mi borde que toca al vecino
        mi_borde = bordes_propuesta[mi_direccion]
        
        # Borde del vecino que me toca
        borde_vecino = bordes_vecino[dir_vecino_hacia_mi]
        
        # VERIFICACIÓN CRÍTICA: deben ser iguales
        if mi_borde != borde_vecino:
            return False
    
    return True


# =============================================================================
# GENERACIÓN DEL TABLERO
# =============================================================================

def generar_tablero(n: int = 25) -> List[List[Optional[Tile]]]:
    """
    Genera un tablero n×n de Carcassonne con colocación válida de losetas.
    
    Algoritmo:
    1. Coloca una loseta inicial en el centro
    2. Ordena el resto de posiciones por distancia al centro
    3. Para cada posición que tenga al menos un vecino:
       a. Prueba cada tipo de loseta disponible
       b. Prueba cada rotación posible (0, 1, 2, 3)
       c. Si encuentra una combinación válida, la coloca
       d. Si no encuentra ninguna, deja la posición vacía
    
    Args:
        n: Tamaño del tablero (25×25 recomendado para 60 losetas)
    
    Returns:
        Tablero n×n con losetas colocadas
    """
    print(f"📊 Generando tablero {n}×{n} con {TOTAL_LOSETAS} losetas disponibles\n")
    
    # Inicializar tablero vacío
    tablero = [[None for _ in range(n)] for _ in range(n)]
    centro = n // 2
    
    # Inventario de losetas
    disponibles = {nombre: cantidad["count"] for nombre, cantidad in TILESET.items()}
    
    # Sistema de meeples
    meeples_totales = {1: random.randint(8, 12), 2: random.randint(8, 12)}
    meeples_usados = {1: 0, 2: 0}
    
    # =========================================================================
    # PASO 1: Colocar loseta inicial en el centro
    # =========================================================================
    nombre_inicial = random.choice(list(disponibles.keys()))
    orientacion_inicial = random.choice([0, 1, 2, 3])
    tablero[centro][centro] = Tile(nombre_inicial, orientacion_inicial, (0, 0))
    disponibles[nombre_inicial] -= 1
    
    print(f"🎯 Loseta inicial: {nombre_inicial} (rot {orientacion_inicial}) en ({centro},{centro})")
    
    # =========================================================================
    # PASO 2: Crear lista de posiciones ordenadas por distancia
    # =========================================================================
    posiciones = []
    for i in range(n):
        for j in range(n):
            if i == centro and j == centro:
                continue
            # Distancia de Chebyshev (cuadrados concéntricos)
            dist = max(abs(i - centro), abs(j - centro))
            posiciones.append((dist, i, j))
    
    # Ordenar por distancia y luego aleatorizar dentro de cada capa
    posiciones.sort(key=lambda x: (x[0], random.random()))
    
    # =========================================================================
    # PASO 3: Intentar colocar losetas
    # =========================================================================
    colocadas = 1
    intentos_fallidos = 0
    
    for _, fila, col in posiciones:
        # Solo intentar si tiene vecinos
        vecinos = obtener_vecinos_con_direccion(tablero, fila, col)
        if len(vecinos) == 0:
            continue
        
        # Obtener tipos de losetas disponibles
        tipos_disponibles = [t for t, cant in disponibles.items() if cant > 0]
        
        if not tipos_disponibles:
            break
        
        # Barajar para aleatoriedad
        random.shuffle(tipos_disponibles)
        
        # Intentar colocar una loseta
        colocada = False
        
        for tipo in tipos_disponibles:
            # Probar cada rotación
            rotaciones = [0, 1, 2, 3]
            random.shuffle(rotaciones)
            
            for rot in rotaciones:
                # ⭐ VERIFICACIÓN DE COMPATIBILIDAD
                if verificar_compatibilidad(tipo, rot, tablero, fila, col):
                    # Decidir si colocar meeple (30% de probabilidad)
                    jugador = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 2, 2])
                    
                    if jugador > 0 and meeples_usados[jugador] < meeples_totales[jugador]:
                        meeple = (jugador, random.randint(0, 8))
                        meeples_usados[jugador] += 1
                    else:
                        meeple = (0, 0)
                    
                    # ✅ Colocar loseta
                    tablero[fila][col] = Tile(tipo, rot, meeple)
                    disponibles[tipo] -= 1
                    colocadas += 1
                    colocada = True
                    break
            
            if colocada:
                break
        
        if not colocada:
            intentos_fallidos += 1
    
    # =========================================================================
    # ESTADÍSTICAS
    # =========================================================================
    losetas_usadas = TOTAL_LOSETAS - sum(disponibles.values())
    cobertura = (colocadas / (n * n)) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 ESTADÍSTICAS DEL TABLERO GENERADO")
    print(f"{'='*60}")
    print(f"  Tamaño del tablero:      {n}×{n} ({n*n} posiciones)")
    print(f"  Losetas colocadas:       {colocadas}")
    print(f"  Cobertura:               {cobertura:.1f}%")
    print(f"  Losetas usadas del set:  {losetas_usadas}/{TOTAL_LOSETAS}")
    print(f"  Posiciones sin llenar:   {intentos_fallidos}")
    print(f"  Meeples rojos:           {meeples_usados[1]}/{meeples_totales[1]}")
    print(f"  Meeples azules:          {meeples_usados[2]}/{meeples_totales[2]}")
    print(f"{'='*60}\n")
    
    return tablero


# =============================================================================
# VERIFICACIÓN DE INTEGRIDAD
# =============================================================================

def verificar_integridad(tablero: List[List[Optional[Tile]]]) -> Tuple[bool, int]:
    """
    Verifica que todas las losetas del tablero encajen correctamente.
    
    Recorre cada loseta y verifica que cada uno de sus bordes coincida
    con el borde correspondiente de sus vecinos.
    
    Returns:
        (es_valido, cantidad_de_errores)
    """
    n = len(tablero)
    errores = []
    
    for i in range(n):
        for j in range(n):
            if tablero[i][j] is None:
                continue
            
            loseta = tablero[i][j]
            bordes_loseta = rotar_bordes(
                TILESET[loseta.nombre]["borders"],
                loseta.orientacion
            )
            
            # Verificar cada vecino
            vecinos = obtener_vecinos_con_direccion(tablero, i, j)
            
            for mi_dir, vecino, dir_vecino in vecinos:
                bordes_vecino = rotar_bordes(
                    TILESET[vecino.nombre]["borders"],
                    vecino.orientacion
                )
                
                mi_borde = bordes_loseta[mi_dir]
                borde_vecino = bordes_vecino[dir_vecino]
                
                if mi_borde != borde_vecino:
                    errores.append({
                        'pos': (i, j),
                        'loseta': f"{loseta.nombre}:rot{loseta.orientacion}",
                        'dir': mi_dir,
                        'borde': mi_borde,
                        'vecino': f"{vecino.nombre}:rot{vecino.orientacion}",
                        'dir_vecino': dir_vecino,
                        'borde_vecino': borde_vecino
                    })
    
    # Mostrar resultados
    if len(errores) == 0:
        print("✅ VERIFICACIÓN EXITOSA: Todas las losetas encajan correctamente")
        return True, 0
    else:
        print(f"❌ VERIFICACIÓN FALLIDA: Se encontraron {len(errores)} errores\n")
        print("=" * 80)
        
        # Mostrar primeros 10 errores
        for idx, e in enumerate(errores[:10], 1):
            print(f"Error #{idx}:")
            print(f"  Posición {e['pos']}: {e['loseta']}")
            print(f"  Su lado {e['dir']} = '{e['borde']}'")
            print(f"  Vecino: {e['vecino']}")
            print(f"  Su lado {e['dir_vecino']} = '{e['borde_vecino']}'")
            print(f"  ⚠️  '{e['borde']}' ≠ '{e['borde_vecino']}'")
            print()
        
        if len(errores) > 10:
            print(f"... y {len(errores) - 10} errores más\n")
        
        print("=" * 80)
        return False, len(errores)


# =============================================================================
# PRUEBAS Y DEBUGGING
# =============================================================================

def test_rotacion():
    """Prueba la función de rotación con ejemplos"""
    print("\n" + "=" * 80)
    print("🧪 TEST DE ROTACIÓN")
    print("=" * 80 + "\n")
    
    for nombre in ["D", "L", "U"]:
        print(f"Loseta {nombre}:")
        base = TILESET[nombre]["borders"]
        print(f"  Base: {base}")
        
        for rot in range(4):
            rotado = rotar_bordes(base, rot)
            print(f"  Rot {rot} ({rot*90}°): N={rotado['N']}, E={rotado['E']}, S={rotado['S']}, W={rotado['W']}")
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🎲 GENERADOR DE TABLERO DE CARCASSONNE")
    print("=" * 80 + "\n")
    
    # Test de rotación
    test_rotacion()
    
    # Generar tablero
    print("=" * 80)
    print("Generando tablero...\n")
    tablero = generar_tablero(n=25)
    
    # Verificar integridad
    print("Verificando integridad del tablero...")
    verificar_integridad(tablero)
    
    print("\n✅ Proceso completado\n")
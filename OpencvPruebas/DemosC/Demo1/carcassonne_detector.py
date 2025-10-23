import cv2
import numpy as np
import os
from itertools import product # Lo usaremos como en find_tiles.py

# --- (Las funciones de clasificación y carga de referencias pueden seguir igual) ---
# def cargar_locetas_referencia(ruta_carpeta): ...
# def clasificar_loceta(loceta_img, referencias): ...
# ... (copia aquí las funciones de clasificación de nuestro script anterior) ...

# --- PARÁMETROS CONFIGURABLES ---
MIN_AREA_LOCETA = 3000
MAX_AREA_LOCETA = 10000
THRESHOLD_MATCHES = 15

# --- (Las funciones de clasificación y carga de referencias van aquí) ---
def cargar_locetas_referencia(ruta_carpeta):
    """
    Carga las imágenes de referencia y pre-calcula sus características ORB.
    """
    referencias = {}
    orb = cv2.ORB_create(nfeatures=1000)

    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith((".jpg", ".png")):
            nombre_loceta = os.path.splitext(archivo)[0]
            ruta_completa = os.path.join(ruta_carpeta, archivo)
            imagen = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
            if imagen is not None:
                kp, des = orb.detectAndCompute(imagen, None)
                referencias[nombre_loceta] = {"kp": kp, "des": des, "img": imagen}
    return referencias

def clasificar_loceta(loceta_img, referencias):
    """
    Compara una loceta con la base de datos de referencia para identificarla.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Asegurarnos de que la loceta a clasificar sea de 8 bits (escala de grises)
    if len(loceta_img.shape) > 2:
        loceta_img = cv2.cvtColor(loceta_img, cv2.COLOR_BGR2GRAY)

    kp_loceta, des_loceta = orb.detectAndCompute(loceta_img, None)

    if des_loceta is None:
        return "Desconocida", 0

    mejor_match = {"nombre": "Desconocida", "matches": 0}
    for nombre, data in referencias.items():
        if data["des"] is not None:
            matches = bf.match(des_loceta, data["des"])
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > mejor_match["matches"]:
                mejor_match["nombre"] = nombre
                mejor_match["matches"] = len(matches)

    if mejor_match["matches"] < THRESHOLD_MATCHES:
        return "Desconocida", 0

    nombre_identificado = mejor_match["nombre"]
    des_referencia = referencias[nombre_identificado]["des"]
    mejor_rotacion = {"rot": 0, "matches": mejor_match["matches"]}

    for rotacion in range(1, 4):
        loceta_rotada = cv2.rotate(loceta_img, rotacion - 1)
        kp_rot, des_rot = orb.detectAndCompute(loceta_rotada, None)
        if des_rot is not None:
            matches = bf.match(des_rot, des_referencia)
            if len(matches) > mejor_rotacion["matches"]:
                mejor_rotacion["matches"] = len(matches)
                mejor_rotacion["rot"] = rotacion
    return nombre_identificado, mejor_rotacion["rot"]


# --- FASE 1: ENCONTRAR LA CUADRÍCULA Y LA HOMOGRAFÍA ---

def encontrar_homografia(imagen):
    """
    Encuentra la transformación de perspectiva (Homografía) del tablero.
    Esta es una versión SIMPLIFICADA del enfoque del paper.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # 1. Detectar Bordes (Como en el paper)
    # cv2.Canny: Detecta bordes (cambios abruptos de intensidad).
    bordes = cv2.Canny(gris, 50, 150, apertureSize=3)
    
    # 2. Detectar Líneas (Como en el paper)
    # cv2.HoughLinesP: Detecta segmentos de línea en la imagen de bordes.
    # Nos devuelve una lista de puntos (x1, y1, x2, y2) por cada línea.
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lineas is None:
        print("No se detectaron líneas. No se puede calcular la homografía.")
        return None, None

    # (Aquí iría la lógica compleja de clustering de líneas que el paper describe)
    # --- SIMPLIFICACIÓN ---
    # Para nuestra demo, asumiremos que las líneas más largas son los bordes del tablero
    # y que podemos encontrar 4 esquinas.
    
    # Vamos a encontrar los puntos extremos de la imagen para simular las 4 esquinas
    # NOTA: Esta es una gran simplificación. El método real requiere agrupar
    # líneas por ángulo (vertical/horizontal) y encontrar sus intersecciones.
    
    puntos_imagen = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        puntos_imagen.append([x1, y1])
        puntos_imagen.append([x2, y2])
        
    puntos_imagen = np.array(puntos_imagen)
    
    # cv2.minAreaRect: Encuentra el rectángulo de área mínima que rodea los puntos.
    # Esto nos da un buen estimado de la orientación del tablero.
    rect = cv2.minAreaRect(puntos_imagen)
    # cv2.boxPoints: Obtiene las 4 esquinas de ese rectángulo.
    esquinas_imagen = cv2.boxPoints(rect)
    esquinas_imagen = np.int0(esquinas_imagen)

    # 3. Calcular Homografía (Como en el paper y board_img_translator.py)
    # Ordenamos las esquinas para que coincidan
    esquinas_imagen = sorted(esquinas_imagen, key=lambda p: p[0] + p[1]) # Simplificación de orden
    
    # Puntos de destino (nuestro tablero ideal)
    # Asumimos que el tablero detectado es de 4x4 locetas (64*4 = 256)
    # Esto debe ajustarse al tamaño real de tu tablero.
    LADO = 256 
    puntos_tablero = np.array([
        [0, 0],
        [LADO, 0],
        [0, LADO],
        [LADO, LADO]
    ], dtype="float32")
    
    # Re-ordenamos las esquinas de la imagen para que coincidan con los puntos del tablero
    esquinas_imagen_float = np.array([
        esquinas_imagen[0],  # Arriba-izquierda
        esquinas_imagen[2],  # Arriba-derecha (simplificación)
        esquinas_imagen[1],  # Abajo-izquierda (simplificación)
        esquinas_imagen[3]   # Abajo-derecha
    ], dtype="float32")

    # cv2.getPerspectiveTransform: Calcula la matriz de homografía.
    # ¡Esta es la matriz mágica!
    H = cv2.getPerspectiveTransform(esquinas_imagen_float, puntos_tablero)
    
    # También calculamos la inversa (como en board_img_translator.py)
    H_inv, _ = cv2.findHomography(puntos_tablero, esquinas_imagen_float) #

    return H, H_inv, esquinas_imagen

# --- FASE 2: EXTRAER Y CLASIFICAR ---

def extraer_y_clasificar(imagen, H_inv, referencias):
    """
    Usa la homografía para extraer y clasificar cada loceta.
    Inspirado en find_tiles.py
    """
    
    # El tamaño de una loceta en el "mundo ideal" (como en el paper y el código)
    TAMAÑO_LOCETA = 64 
    matriz_resultado = {}
    imagen_con_contornos = imagen.copy()
    
    # Iteramos sobre una cuadrícula teórica, como en find_tiles.py
    # Asumimos un tablero de 4x4 para esta demo
    for y, x in product(range(4), range(4)):
        
        # 1. Definir los puntos de la loceta en el "mundo ideal"
        puntos_tablero = np.array([
            [x * TAMAÑO_LOCETA, y * TAMAÑO_LOCETA],
            [(x + 1) * TAMAÑO_LOCETA, y * TAMAÑO_LOCETA],
            [(x + 1) * TAMAÑO_LOCETA, (y + 1) * TAMAÑO_LOCETA],
            [0 * TAMAÑO_LOCETA, (y + 1) * TAMAÑO_LOCETA]
        ], dtype="float32")
        
        # 2. Proyectar esos puntos en la imagen real usando la Homografía Inversa
        # cv2.perspectiveTransform: Aplica la matriz de homografía a un conjunto de puntos.
        puntos_imagen = cv2.perspectiveTransform(np.array([puntos_tablero]), H_inv)[0]
        
        # 3. Extraer la loceta (Como en el paper y find_tiles.py)
        # Puntos de destino para la extracción (un cuadrado perfecto)
        puntos_destino = np.array([
            [0, 0],
            [TAMAÑO_LOCETA, 0],
            [TAMAÑO_LOCETA, TAMAÑO_LOCETA],
            [0, TAMAÑO_LOCETA]
        ], dtype="float32")
        
        # cv2.getPerspectiveTransform: Calcula la matriz para "des-distorsionar" esta loceta.
        M = cv2.getPerspectiveTransform(puntos_imagen, puntos_destino)
        # cv2.warpPerspective: ¡Extrae la loceta y la aplana!
        loceta_plana = cv2.warpPerspective(imagen, M, (TAMAÑO_LOCETA, TAMAÑO_LOCETA))
        
        # 4. Clasificar la loceta (usando nuestra función anterior)
        # El paper también usa un Canny para ver si es una loceta o la mesa
        
        # (Añadimos un chequeo simple como el del paper)
        media_bordes = np.mean(cv2.Canny(loceta_plana, 50, 100))
        
        if media_bordes > 10: # Si hay suficientes bordes, es una loceta (ajustar umbral)
            nombre, rotacion = clasificar_loceta(loceta_plana, referencias)
            
            # Dibujar en la imagen de debug
            puntos_imagen_int = np.int32(puntos_imagen)
            cv2.polylines(imagen_con_contornos, [puntos_imagen_int], True, (0, 255, 0), 2)
            texto = f"{nombre} R{rotacion}"
            cv2.putText(imagen_con_contornos, texto, (puntos_imagen_int[0][0], puntos_imagen_int[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            matriz_resultado[(x, y)] = (nombre, rotacion)
        else:
            matriz_resultado[(x, y)] = None # Espacio vacío

    return matriz_resultado, imagen_con_contornos

# --- Flujo Principal del Programa ---
if __name__ == "__main__":
    imagen_tablero = cv2.imread("tablero.jpg")
    referencias = cargar_locetas_referencia("locetas_referencia")

    if imagen_tablero is None:
        print("Error: No se pudo cargar 'tablero.jpg'.")
    elif not referencias:
        print("Error: No se encontraron imágenes en 'locetas_referencia'.")
    else:
        # 1. Encontrar la homografía
        H, H_inv, esquinas = encontrar_homografia(imagen_tablero)
        
        if H is not None:
            print("Homografía calculada.")
            
            # (Debug) Dibujar las esquinas detectadas
            cv2.drawContours(imagen_tablero, [esquinas], 0, (0, 0, 255), 3)
            cv2.imwrite("debug_homografia.jpg", imagen_tablero)
            print("Se guardó 'debug_homografia.jpg' con los bordes detectados.")

            # 2. Extraer y clasificar
            matriz, imagen_resultado = extraer_y_clasificar(imagen_tablero, H_inv, referencias)
            
            print("\n--- MATRIZ RESULTANTE ---")
            for (y, x), valor in matriz.items():
                if valor:
                    print(f"Posición ({x}, {y}): {valor[0]} (Rot: {valor[1]})")
            
            cv2.imwrite("resultado.jpg", imagen_resultado)
            print("\nAnálisis completo. Se ha guardado la imagen 'resultado.jpg'.")
        
        else:
            print("Fallo al detectar el tablero. Intenta con una imagen más clara o con bordes mejor definidos.")
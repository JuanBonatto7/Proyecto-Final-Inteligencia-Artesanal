import cv2
import numpy as np

# --- 1. Variables de Marcador de Posición ---
# El usuario debe ajustar estas rutas
ruta_imagen_tablero = '20250905_195402.jpg'
# Esta sería la ruta a tu imagen de referencia del tipo 'A' (sin rotar)
ruta_plantilla_A = 'plantilla_A_0deg.png' 


# --- 2. Funciones Clave de Visión Artificial ---

def clasificar_loseta(loseta_recortada, diccionario_plantillas):
    """
    Compara una loseta recortada con todas las plantillas base (A, B, C...)
    y detecta su tipo y rotación con Template Matching.
    """
    mejor_match_valor = -1
    mejor_tipo = None
    mejor_rotacion = 0
    
    # Supongamos que tienes un diccionario con { 'A': img_A, 'B': img_B, ... }
    for tipo, plantilla_base in diccionario_plantillas.items():
        for rotacion in range(4): # 0, 90, 180, 270 grados
            # Generar la versión rotada de la plantilla para la comparación
            if rotacion == 1:
                plantilla_rotada = cv2.rotate(plantilla_base, cv2.ROTATE_90_CLOCKWISE)
            elif rotacion == 2:
                plantilla_rotada = cv2.rotate(plantilla_base, cv2.ROTATE_180)
            elif rotacion == 3:
                plantilla_rotada = cv2.rotate(plantilla_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                plantilla_rotada = plantilla_base
            
            # Asegurar que el tamaño de la loseta a clasificar y la plantilla coincidan para TM
            # Esto es CRÍTICO, o podrías tener que usar otros métodos como SIFT/ORB
            h, w, _ = plantilla_rotada.shape
            loseta_resize = cv2.resize(loseta_recortada, (w, h))

            # Realizar el Template Matching (comparación)
            # cv2.TM_CCOEFF_NORMED es uno de los métodos más robustos
            resultado = cv2.matchTemplate(loseta_resize, plantilla_rotada, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(resultado)
            
            # Actualizar el mejor resultado
            if max_val > mejor_match_valor:
                mejor_match_valor = max_val
                mejor_tipo = tipo
                mejor_rotacion = rotacion * 90 # Almacenar la rotación en grados o 0-3

    return mejor_tipo, mejor_rotacion, mejor_match_valor

# --- 3. Uso de Ejemplo (Solo para entender Template Matching) ---

# Cargar la imagen del tablero completo
# img_tablero = cv2.imread(ruta_imagen_tablero)
# Aquí se asumiría que ya has recortado una loseta específica (loseta_muestra)
# loseta_muestra = img_tablero[y:y+h, x:x+w] 

# Ejemplo de carga de una plantilla (DEBES REEMPLAZAR ESTO CON TU LÓGICA DE CARGA REAL)
# plantilla_A = cv2.imread(ruta_plantilla_A)
# diccionario_plantillas_ejemplo = {'A': plantilla_A, 'B': img_B_etc}
# ...

# Si tuvieras una loseta recortada de verdad:
# tipo, rotacion, confianza = clasificar_loseta(loseta_muestra, diccionario_plantillas_ejemplo)

# print(f"Loseta clasificada como Tipo: {tipo} con Rotación: {rotacion} grados (Confianza: {confianza:.2f})")
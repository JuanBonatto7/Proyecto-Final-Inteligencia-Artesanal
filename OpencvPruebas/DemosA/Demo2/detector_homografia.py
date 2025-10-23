import cv2
import numpy as np
import os

# --- PARÁMETROS CONFIGURABLES (NUEVOS VALORES) ---
MIN_AREA_LOCETA = 1000      # Bajamos el mínimo para capturar locetas en perspectiva o sombra.
MAX_AREA_LOCETA = 15000     # Un rango que incluye el área más grande que vimos (8616.5) con margen.
FACTOR_DESENFOQUE = 15      # Este se queda como está, funcionó bien.
THRESHOLD_MATCHES = 15


# --- FUNCIONES AUXILIARES ---

def ordenar_puntos(puntos):
    """ Ordena los 4 puntos de un contorno en el orden:
    arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda.
    """
    puntos = puntos.reshape((4, 2))
    puntos_ordenados = np.zeros((4, 2), dtype="float32")
    suma = puntos.sum(axis=1)
    puntos_ordenados[0] = puntos[np.argmin(suma)]
    puntos_ordenados[2] = puntos[np.argmax(suma)]
    diferencia = np.diff(puntos, axis=1)
    puntos_ordenados[1] = puntos[np.argmin(diferencia)]
    puntos_ordenados[3] = puntos[np.argmax(diferencia)]
    return puntos_ordenados


def corregir_perspectiva(imagen, puntos_esquinas):
    """ Aplica una transformación de perspectiva para obtener una vista cenital de la loceta. """
    puntos_ordenados = ordenar_puntos(puntos_esquinas)
    (tl, tr, br, bl) = puntos_ordenados

    ancho_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    ancho_max = max(int(ancho_a), int(ancho_b))

    alto_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    alto_max = max(int(alto_a), int(alto_b))

    puntos_destino = np.array([
        [0, 0],
        [ancho_max - 1, 0],
        [ancho_max - 1, alto_max - 1],
        [0, alto_max - 1]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(puntos_ordenados, puntos_destino)
    imagen_transformada = cv2.warpPerspective(imagen, matriz, (ancho_max, alto_max))
    return cv2.resize(imagen_transformada, (200, 200))


def encontrar_locetas(imagen_tablero):
    """ Paso 1: Detecta y aísla todas las locetas cuadradas del tablero. """
    locetas_encontradas = []
    imagen_original_para_dibujar = imagen_tablero.copy()

    gris = cv2.cvtColor(imagen_tablero, cv2.COLOR_BGR2GRAY)
    desenfocada = cv2.GaussianBlur(gris, (FACTOR_DESENFOQUE, FACTOR_DESENFOQUE), 0)
    binaria = cv2.adaptiveThreshold(
        desenfocada, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((5, 5), np.uint8)
    binaria_erosionada = cv2.erode(binaria, kernel, iterations=1)
    cv2.imwrite("debug_binaria_erosionada.jpg", binaria_erosionada)
    print("Se ha guardado la imagen 'debug_binaria_erosionada.jpg' para análisis.")

    contornos, _ = cv2.findContours(binaria_erosionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("\n--- ÁREAS DE CONTORNOS DETECTADOS ---")
    for c in contornos:
        area = cv2.contourArea(c)
        print(f"Contorno con área: {area}")
        if MIN_AREA_LOCETA < area < MAX_AREA_LOCETA:
            perimetro = cv2.arcLength(c, True)
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                print(f" -> ¡Candidato a loceta encontrado con área {area}!")
                loceta_corregida = corregir_perspectiva(imagen_original_para_dibujar, aprox)
                locetas_encontradas.append((loceta_corregida, aprox))
                cv2.drawContours(imagen_original_para_dibujar, [aprox], -1, (0, 255, 0), 3)

    return locetas_encontradas, imagen_original_para_dibujar


def cargar_locetas_referencia(ruta_carpeta):
    """ Carga las imágenes de referencia y pre-calcula sus características ORB. """
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
    """ Paso 2: Compara una loceta con la base de datos de referencia para identificarla. """
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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


# --- FLUJO PRINCIPAL ---

if __name__ == "__main__":
    imagen_tablero = cv2.imread("tablero.jpg")
    referencias = cargar_locetas_referencia("locetas_referencia")

    if imagen_tablero is None:
        print("Error: No se pudo cargar la imagen 'tablero.jpg'. Verifica que el archivo exista.")
    elif not referencias:
        print("Error: No se encontraron imágenes en la carpeta 'locetas_referencia'.")
    else:
        locetas_aisladas, imagen_con_contornos = encontrar_locetas(imagen_tablero)
        print(f"\nProceso de detección terminado. Se encontraron {len(locetas_aisladas)} locetas válidas.")

        matriz_resultado = []
        for loceta_img, contorno in locetas_aisladas:
            nombre, rotacion = clasificar_loceta(cv2.cvtColor(loceta_img, cv2.COLOR_BGR2GRAY), referencias)
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                texto = f"{nombre} R{rotacion}"
                cv2.putText(imagen_con_contornos, texto, (cX - 40, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                matriz_resultado.append({"nombre": nombre, "rotacion": rotacion, "x": cX, "y": cY})

        print("\n--- RESULTADOS FINALES ---")
        if not matriz_resultado:
            print("No se clasificó ninguna loceta.")
        else:
            for item in matriz_resultado:
                print(f"Loceta: {item['nombre']}, Rotación: {item['rotacion']} en ({item['x']}, {item['y']})")

        cv2.imwrite("resultado.jpg", imagen_con_contornos)
        print("\nAnálisis completo. Se ha guardado la imagen 'resultado.jpg'.")

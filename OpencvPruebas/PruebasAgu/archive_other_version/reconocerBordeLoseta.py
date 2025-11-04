import cv2
import numpy as np
import ctypes

def ajustar_imagen_a_pantalla(img, margen=80):
    """
    Escala la imagen para que quepa en la pantalla del usuario (margen en píxeles).
    Usa GetSystemMetrics en Windows para obtener la resolución.
    """
    try:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:
        # Valor por defecto si no se puede obtener la resolución
        screen_w, screen_h = 1366, 768

    max_w = max(100, screen_w - margen)
    max_h = max(100, screen_h - margen)

    h, w = img.shape[:2]
    if w > max_w or h > max_h:
        escala = min(max_w / w, max_h / h)
        new_w = int(w * escala)
        new_h = int(h * escala)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def individualizar_losetas_mejorado(ruta_imagen):
    """
    Intenta detectar y contornear todas las losetas cuadradas en una imagen de Carcassonne
    mejorando la separación del fondo y los filtros de forma. Muestra la imagen ajustada
    para que se vea completa en pantalla.
    """
    # 1. Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return

    imagen_contornos = imagen.copy()
    
    # 2. Preprocesamiento: Escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 3. Umbral Adaptativo para aislar las losetas del fondo
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    # 4. Operaciones Morfológicas (Dilatación y Erosión)
    kernel = np.ones((3,3), np.uint8)
    dilatado = cv2.dilate(umbral, kernel, iterations=1) 
    erosion = cv2.erode(dilatado, kernel, iterations=1)

    # 5. Detección de Contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Filtrar y Dibujar Cuadrados Potenciales (Losetas)
    contador_losetas = 0
    
    # Parámetros para filtrar (ajustar según el tamaño de la loseta en la foto)
    # Para tu imagen, las losetas son relativamente grandes.
    AREA_MINIMA = 1500 # Aumentado de 2000 para ser más flexible
    AREA_MAXIMA = 30000 # Aumentado para no excluir todo el tablero
    RATIO_TOLERANCIA = 0.20 # La relación de aspecto debe estar entre 0.8 y 1.2 (1 +/- 0.20)

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        
        # Aproximación: Usamos un factor menor (0.02) para ser más preciso con las esquinas
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        
        # Condición: Solo formas con 4 esquinas (cuadrados o rectángulos)
        if len(aprox) == 4:
            area = cv2.contourArea(aprox)
            
            # Filtro por área
            if AREA_MINIMA < area < AREA_MAXIMA:
                
                (x, y, w, h) = cv2.boundingRect(aprox)
                aspect_ratio = float(w) / h
                
                # Filtro por relación de aspecto (cercano a 1)
                if abs(aspect_ratio - 1.0) <= RATIO_TOLERANCIA:
                    # Dibujar el contorno final encontrado
                    cv2.drawContours(imagen_contornos, [aprox], -1, (0, 255, 0), 3) # Línea verde más gruesa
                    contador_losetas += 1

    print(f"Losetas identificadas: {contador_losetas}")

    # 7. Mostrar el resultado y ajustarlo a la pantalla
    imagen_mostrable = ajustar_imagen_a_pantalla(imagen_contornos)
    window_name = "Losetas Contorneadas Mejoradas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, imagen_mostrable)
    # Ajustar el tamaño de la ventana al de la imagen escalada
    h_show, w_show = imagen_mostrable.shape[:2]
    cv2.resizeWindow(window_name, w_show, h_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Opcional: Guardar la imagen de salida
    # cv2.imwrite("losetas_contorneadas_final.jpg", imagen_contornos)


# --- Ejecución del Programa ---
# Usa la ruta de la imagen del tablero completo
ruta_de_tu_imagen = '20250905_195402.jpg' 
individualizar_losetas_mejorado(ruta_de_tu_imagen)
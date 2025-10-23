import cv2
import numpy as np
import os
from collections import deque

# ============= CONFIGURACIÓN =============
# Tamaño al que se normalizarán las losetas para la clasificación.
TILE_CLASSIFICATION_SIZE = 64
# Tamaño al que se normalizarán las losetas tras la corrección de perspectiva.
TILE_UNWARPED_SIZE = 100
# Umbral de confianza para la clasificación.
MIN_MATCH_SCORE = 0.50

class CarcassonneBoardRecognizer:
    
    def __init__(self, referencias_path):
        self.referencias = self._cargar_referencias(referencias_path)
        self.avg_tile_size = (0, 0)

    def _cargar_referencias(self, path):
        """Carga y pre-procesa las imágenes de referencia de las losetas."""
        refs = {}
        if not os.path.exists(path):
            print(f"⚠️ Directorio de referencias no encontrado en '{path}'")
            return refs
            
        print(f"📚 Cargando referencias desde '{path}'...")
        for archivo in os.listdir(path):
            if not archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            nombre = os.path.splitext(archivo)[0]
            img = cv2.imread(os.path.join(path, archivo))
            
            if img is not None:
                # Normaliza para clasificación
                img_norm = cv2.resize(img, (TILE_CLASSIFICATION_SIZE, TILE_CLASSIFICATION_SIZE))
                gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
                
                # Normalización Z-score para robustez a la iluminación
                gray = gray.astype(np.float32)
                mean, std = gray.mean(), gray.std()
                gray_norm = (gray - mean) / (std + 1e-6) # Añadir epsilon para evitar división por cero
                
                refs[nombre] = gray_norm
        
        print(f"  ✅ {len(refs)} referencias cargadas.")
        return refs

    def find_tile_candidates(self, imagen):
        """Detecta candidatos a loseta basándose en contornos cuadrados."""
        print("\n🔍 Paso 1: Buscando candidatos a loseta...")
        
        # Preprocesamiento
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Usar Canny es a menudo más robusto que un umbral simple
        edges = cv2.Canny(blurred, 50, 150)
        cv2.imwrite("debug_01_edges.jpg", edges)
        
        # Detección de contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        h, w = imagen.shape[:2]
        min_area = (w * h) * 0.001 # Una loseta debe ocupar al menos el 0.1% de la imagen
        max_area = (w * h) * 0.1   # Y no más del 10%
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue

            # Aproximar contorno a un polígono
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Un cuadrado tiene 4 vértices
            if len(approx) == 4:
                candidates.append(approx)
        
        print(f"  📊 {len(contours)} contornos encontrados, {len(candidates)} son candidatos cuadrados.")
        return candidates

    def unwarp_and_classify_tile(self, imagen, contour, tile_id):
        """Corrige la perspectiva de un candidato y lo clasifica."""
        # Extraer la loseta con corrección de perspectiva
        pts_src = np.array(contour, dtype=np.float32).reshape(4, 2)
        
        # Puntos de destino para una imagen plana
        pts_dst = np.array([
            [0, 0],
            [TILE_UNWARPED_SIZE - 1, 0],
            [TILE_UNWARPED_SIZE - 1, TILE_UNWARPED_SIZE - 1],
            [0, TILE_UNWARPED_SIZE - 1]
        ], dtype=np.float32)

        # Reordenar puntos del contorno para que coincidan con el destino
        # El punto superior-izquierdo debe ser el primero, luego superior-derecho, etc.
        s = pts_src.sum(axis=1)
        ordered_pts = np.zeros((4, 2), dtype=np.float32)
        ordered_pts[0] = pts_src[np.argmin(s)]
        ordered_pts[2] = pts_src[np.argmax(s)]
        diff = np.diff(pts_src, axis=1)
        ordered_pts[1] = pts_src[np.argmin(diff)]
        ordered_pts[3] = pts_src[np.argmax(diff)]

        # Aplicar transformación de perspectiva
        matrix = cv2.getPerspectiveTransform(ordered_pts, pts_dst)
        unwarped = cv2.warpPerspective(imagen, matrix, (TILE_UNWARPED_SIZE, TILE_UNWARPED_SIZE))
        cv2.imwrite(f"debug_tile_{tile_id:03d}.jpg", unwarped)

        # Clasificar la loseta (tu lógica de clasificación es buena)
        pieza = cv2.resize(unwarped, (TILE_CLASSIFICATION_SIZE, TILE_CLASSIFICATION_SIZE))
        gray = cv2.cvtColor(pieza, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean, std = gray.mean(), gray.std()
        gray_norm = (gray - mean) / (std + 1e-6)

        mejor = {'nombre': 'desconocido', 'rotacion': 0, 'score': -np.inf}
        
        for nombre, ref_gray_norm in self.referencias.items():
            for rot in range(4):
                pieza_rot = np.rot90(gray_norm, k=rot)
                score = np.sum(pieza_rot * ref_gray_norm)
                
                if score > mejor['score']:
                    mejor = {'nombre': nombre, 'rotacion': rot, 'score': score}
        
        # Normalizar el score a una "confianza" entre 0 y 1 (aproximado)
        confianza = 1 / (1 + np.exp(-mejor['score'] / (TILE_CLASSIFICATION_SIZE)))
        
        if confianza < MIN_MATCH_SCORE:
            return None, None, None

        return mejor['nombre'], mejor['rotacion'], confianza

    def reconstruct_grid(self, detected_tiles):
        """Reconstruye la matriz del tablero a partir de las posiciones de las losetas."""
        if not detected_tiles:
            return []
        
        print("\n🔍 Paso 3: Reconstruyendo la grilla...")
        
        # Calcular tamaño promedio de loseta
        avg_w = np.mean([t['w'] for t in detected_tiles])
        avg_h = np.mean([t['h'] for t in detected_tiles])
        self.avg_tile_size = (avg_w, avg_h)
        print(f"  📏 Tamaño promedio de loseta: {avg_w:.1f}x{avg_h:.1f} px")

        # Mapear centroides a losetas
        centroids = { (t['cx'], t['cy']): t for t in detected_tiles }
        
        # Encontrar la loseta de inicio (arriba a la izquierda)
        start_centroid = min(centroids.keys(), key=lambda p: (p[1], p[0]))
        
        # BFS para construir la grilla
        queue = deque([(start_centroid, (0, 0))]) # (centroid, (fila, col))
        visited = {start_centroid}
        grid_map = { (0, 0): centroids[start_centroid] }
        
        while queue:
            current_centroid, (r, c) = queue.popleft()
            cx, cy = current_centroid

            # Buscar vecinos en las 4 direcciones
            for dr, dc, direction in [(0, 1, "derecha"), (0, -1, "izquierda"), (1, 0, "abajo"), (-1, 0, "arriba")]:
                # Estimar la posición del centroide del vecino
                next_cx_est = cx + dc * avg_w
                next_cy_est = cy + dr * avg_h
                
                # Encontrar el centroide real más cercano al estimado
                mejor_vecino = None
                min_dist = 0.5 * avg_w # Un vecino no puede estar más lejos que la mitad de una loseta
                
                for other_centroid in centroids:
                    if other_centroid in visited:
                        continue
                    dist = np.sqrt((other_centroid[0] - next_cx_est)**2 + (other_centroid[1] - next_cy_est)**2)
                    if dist < min_dist:
                        min_dist = dist
                        mejor_vecino = other_centroid
                
                if mejor_vecino:
                    new_coord = (r + dr, c + dc)
                    if new_coord not in grid_map:
                        visited.add(mejor_vecino)
                        queue.append((mejor_vecino, new_coord))
                        grid_map[new_coord] = centroids[mejor_vecino]
        
        # Crear la matriz final
        min_r = min(r for r,c in grid_map.keys())
        max_r = max(r for r,c in grid_map.keys())
        min_c = min(c for r,c in grid_map.keys())
        max_c = max(c for r,c in grid_map.keys())
        
        matriz = [[None] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
        
        for (r, c), tile_info in grid_map.items():
            matriz[r - min_r][c - min_c] = (tile_info['nombre'], tile_info['rotacion'])

        print(f"  ✅ Grilla reconstruida de {len(matriz)}x{len(matriz[0])}")
        return matriz


    def procesar(self, imagen_path):
        """Pipeline completo de detección y clasificación."""
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            raise ValueError(f"Error cargando {imagen_path}")

        # Redimensionar si es muy grande para acelerar el proceso
        h, w = imagen.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            imagen = cv2.resize(imagen, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # PASO 1: Encontrar candidatos a loseta
        candidates = self.find_tile_candidates(imagen)
        
        # Dibujar candidatos en una imagen de debug
        debug_candidates_img = imagen.copy()
        cv2.drawContours(debug_candidates_img, candidates, -1, (0, 255, 0), 2)
        cv2.imwrite("debug_02_candidates.jpg", debug_candidates_img)

        # PASO 2: Clasificar cada candidato
        print("\n🔍 Paso 2: Clasificando losetas...")
        detected_tiles = []
        imagen_final = imagen.copy()
        
        for i, contour in enumerate(candidates):
            nombre, rotacion, confianza = self.unwarp_and_classify_tile(imagen, contour, i)
            
            if nombre:
                # Obtener bounding box y centroide para la reconstrucción
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w//2, y + h//2
                
                detected_tiles.append({
                    'nombre': nombre,
                    'rotacion': rotacion,
                    'confianza': confianza,
                    'contour': contour,
                    'cx': cx, 'cy': cy, 'w': w, 'h': h
                })
                
                # Dibujar resultado en la imagen final
                texto = f"{nombre[:5]}R{rotacion}"
                cv2.drawContours(imagen_final, [contour], -1, (0, 255, 0), 2)
                cv2.putText(imagen_final, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"  ✅ {len(detected_tiles)} losetas clasificadas con éxito.")
        cv2.imwrite("resultado_final.jpg", imagen_final)
        
        # PASO 3: Reconstruir la grilla
        matriz = self.reconstruct_grid(detected_tiles)
        
        return matriz, detected_tiles


if __name__ == "__main__":
    try:
        # Asegúrate de que el path a tus referencias sea correcto
        detector = CarcassonneBoardRecognizer("locetas_referencia")
        
        # Asegúrate de que el path a tu imagen de tablero sea correcto
        matriz, _ = detector.procesar("tablero.jpg")
        
        print("\n" + "="*70)
        print("📊 MATRIZ DEL TABLERO RESULTANTE")
        print("="*70)
        if matriz:
            for fila in matriz:
                linea = ""
                for celda in fila:
                    if celda:
                        linea += f"[{celda[0]:>8s}, R{celda[1]}] "
                    else:
                        linea += "[-- V A C I O --] "
                print(linea)
        else:
            print("No se pudo construir la matriz del tablero.")

    except Exception as e:
        print(f"\n💥 ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
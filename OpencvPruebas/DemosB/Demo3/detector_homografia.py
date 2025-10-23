import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

# ============= CONFIGURACIÓN =============
TILE_SIZE = 64  # Tamaño estándar del proyecto de referencia
MIN_MATCH_SCORE = 0.50

class CarcassonneDetector:
    """
    Implementación del pipeline del proyecto de referencia:
    1. Resize → Blur → Canny → Dilate → HoughLinesP
    2. FindIntersections con sistema de votación
    3. RANSACHomography
    4. FindTiles usando homografía
    5. TileClassifier con NCC
    """
    
    def __init__(self, referencias_path):
        self.referencias = self._cargar_referencias(referencias_path)
        self.H = None  # Homografía
        self.H_inv = None
        
    def _cargar_referencias(self, path):
        """Carga templates de piezas"""
        refs = {}
        if not os.path.exists(path):
            return refs
            
        for archivo in os.listdir(path):
            if not archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            nombre = os.path.splitext(archivo)[0]
            img = cv2.imread(os.path.join(path, archivo))
            
            if img is not None:
                # Recortar bordes (como en el proyecto)
                img = img[4:-4, 4:-4]
                img_norm = cv2.resize(img, (TILE_SIZE, TILE_SIZE))
                gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
                
                refs[nombre] = {'img': img_norm, 'gray': gray}
        
        print(f"📚 Referencias: {len(refs)}")
        return refs
    
    def paso1_preprocesar(self, imagen):
        """PASO 1: Resize + Blur"""
        print("\n[1/6] Preprocesamiento...")
        
        # Resize a máximo 800px de ancho
        h, w = imagen.shape[:2]
        max_width = 800
        if w > max_width:
            scale = max_width / w
            imagen = cv2.resize(imagen, None, fx=scale, fy=scale)
            print(f"  Redimensionado a {imagen.shape[1]}x{imagen.shape[0]}")
        
        # Blur Gaussiano (9x9, std=0.75)
        blurred = cv2.GaussianBlur(imagen, (9, 9), 0.75)
        
        cv2.imwrite("debug_01_blur.jpg", blurred)
        return imagen, blurred
    
    def paso2_detectar_bordes_lineas(self, blurred, imagen):
        """PASO 2: Canny + Dilate + HoughLinesP"""
        print("\n[2/6] Detección de bordes y líneas...")
        
        # Canny
        edges = cv2.Canny(blurred, 100, 200)
        cv2.imwrite("debug_02_edges.jpg", edges)
        
        # Dilate para engrosar líneas
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        cv2.imwrite("debug_03_edges_dilated.jpg", edges_dilated)
        
        # Hough Lines Probabilistic
        line_segments = cv2.HoughLinesP(
            edges_dilated,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=30
        )
        
        # ===== DEBUG: Visualizar líneas detectadas =====
        debug_lineas = imagen.copy()
        if line_segments is not None:
            for x1, y1, x2, y2 in line_segments[:, 0]:
                cv2.line(debug_lineas, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite("debug_lineas.jpg", debug_lineas)
        print("Guardado debug_lineas.jpg con las líneas detectadas")
        # ===============================================

        if line_segments is None:
            print("  ⚠️ No se detectaron líneas")
            return None
        
        line_segments = line_segments.reshape(-1, 4)
        print(f"  Segmentos detectados: {len(line_segments)}")
        
        return line_segments

    def paso3_encontrar_intersecciones(self, imagen, line_segments):
        """
        PASO 3: FindIntersections con sistema de votación
        Basado en find_intersections.py del proyecto
        """
        print("\n[3/6] Encontrando intersecciones con votación...")
        
        if line_segments is None or len(line_segments) < 2:
            return None, None
        
        # Calcular ángulos
        angles = []
        for x1, y1, x2, y2 in line_segments:
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        
        angles = np.array(angles).reshape(-1, 1)
        angles_deg = np.rad2deg(angles)
        
        # KMeans para agrupar en 2 direcciones (H y V)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(angles_deg)
        labels = kmeans.labels_
        
        # --- MODIFICACIÓN 1: Lógica de asignación CORREGIDA ---
        # Queremos que label 0 = Vertical (cerca de +/- 90)
        # Queremos que label 1 = Horizontal (cerca de 0 / +/- 180)
        centers = kmeans.cluster_centers_.flatten()
        
        # Si el centro 0 está MÁS CERCA de 0 (horizontal) que el centro 1...
        if np.abs(centers[0]) < np.abs(centers[1]):
            labels = 1 - labels # ...invertimos los labels.
        
        lines_v_ruidosas = line_segments[labels == 0]  # Verticales (label 0)
        lines_h_ruidosas = line_segments[labels == 1]  # Horizontales (label 1)
        
        # --- MODIFICACIÓN 2: Usar la nueva función de FILTRADO ---
        print("   Filtrando líneas 'verticales' por ángulo...")
        lines1 = self._filtrar_lineas_por_angulo(lines_v_ruidosas, 10)
        
        print("   Filtrando líneas 'horizontales' por ángulo...")
        lines2 = self._filtrar_lineas_por_angulo(lines_h_ruidosas, 10)
        
        if len(lines1) == 0 or len(lines2) == 0:
            print(" ⚠️ No quedaron líneas después del filtrado de ángulo.")
            return None, None
        
        # --- MODIFICACIÓN 3: Comentarios y variables CORREGIDOS ---
        print(f"   Líneas V limpias: {len(lines1)}, H limpias: {len(lines2)}")
        
        # Sistema de votación para intersecciones
        h, w = imagen.shape[:2]
        votes = np.zeros((h, w), dtype=np.float32)
        intersection_labels = {}
        
        # Agrupar líneas paralelas (ahora con líneas limpias)
        groups1 = self._agrupar_lineas_por_posicion(lines1, es_vertical=True)
        groups2 = self._agrupar_lineas_por_posicion(lines2, es_vertical=False)
        
        print(f"   Grupos V: {len(groups1)}, H: {len(groups2)}")
        
        # Debug: dibujar grupos
        debug_groups = imagen.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, group in enumerate(groups1[:10]):  # Primeros 10 grupos
            color = colors[i % len(colors)]
            for line in group:
                cv2.line(debug_groups, (line[0], line[1]), (line[2], line[3]), color, 2)
        for i, group in enumerate(groups2[:10]):
            color = colors[i % len(colors)]
            for line in group:
                cv2.line(debug_groups, (line[0], line[1]), (line[2], line[3]), color, 2)
        cv2.imwrite("debug_03b_groups.jpg", debug_groups)
        print("💾 debug_03b_groups.jpg - Primeros 10 grupos de cada tipo")
        
        # Calcular intersecciones con votación
        # 'i' será el índice Vertical (grupo 1), 'j' el Horizontal (grupo 2)
        for i, group1 in enumerate(groups1):
            for j, group2 in enumerate(groups2):
                # Tomar línea representativa de cada grupo
                l1 = group1[0]
                l2 = group2[0]
                
                # Encontrar intersección
                punto = self._interseccion_lineas(l1, l2)
                
                if punto is None:
                    continue
                
                x, y = punto
                if not (0 <= x < w and 0 <= y < h):
                    continue
                
                # Votar basado en longitud de líneas
                len1 = np.sqrt((l1[2]-l1[0])**2 + (l1[3]-l1[1])**2)
                len2 = np.sqrt((l2[2]-l2[0])**2 + (l2[3]-l2[1])**2)
                vote = len1 + len2
                
                votes[y, x] += vote
                intersection_labels[(x, y)] = (i, j)
        
        # Non-Maximum Suppression en votes
        window_size = 25
        intersecciones = []
        
        for y in range(0, h, window_size):
            for x in range(0, w, window_size):
                window = votes[y:min(y+window_size, h), x:min(x+window_size, w)]
                
                if window.max() == 0:
                    continue
                
                # Encontrar máximo local
                max_idx = np.unravel_index(window.argmax(), window.shape)
                max_y = y + max_idx[0]
                max_x = x + max_idx[1]
                
                if (max_x, max_y) in intersection_labels:
                    i_label, j_label = intersection_labels[(max_x, max_y)]
                    intersecciones.append({
                        'punto': (max_x, max_y),
                        'label': (i_label, j_label) # (Vertical, Horizontal)
                    })
        
        print(f"   Intersecciones encontradas: {len(intersecciones)}")
        
        # Debug
        debug_int = imagen.copy()
        for inter in intersecciones:
            x, y = inter['punto']
            cv2.circle(debug_int, (x, y), 5, (0, 0, 255), -1)
            i, j = inter['label']
            cv2.putText(debug_int, f"({i},{j})", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imwrite("debug_04_intersections.jpg", debug_int)
        
        return intersecciones, intersection_labels

    def _agrupar_lineas_por_posicion(self, lines, es_vertical=True):
        """
        Agrupa líneas paralelas por su posición usando BINS.
        Evita el problema de agrupamiento secuencial.
        """
        if len(lines) == 0:
            return []
        
        # Extraer posición característica
        if es_vertical:
            positions = np.array([(l[0] + l[2]) / 2 for l in lines])
        else:
            positions = np.array([(l[1] + l[3]) / 2 for l in lines])
        
        # Estrategia: dividir en bins de tamaño fijo
        BIN_SIZE = 40  # Cada bin es de 40 píxeles
        
        min_pos = positions.min()
        max_pos = positions.max()
        
        # Crear bins
        num_bins = int((max_pos - min_pos) / BIN_SIZE) + 1
        bins = [[] for _ in range(num_bins)]
        
        # Asignar líneas a bins
        for i, pos in enumerate(positions):
            bin_idx = int((pos - min_pos) / BIN_SIZE)
            if 0 <= bin_idx < num_bins:
                bins[bin_idx].append(lines[i])
        
        # Filtrar bins vacíos y tomar la línea más larga de cada bin
        groups = []
        for bin_lines in bins:
            if len(bin_lines) > 0:
                # Tomar la línea más larga del bin
                lengths = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in bin_lines]
                best_line = bin_lines[np.argmax(lengths)]
                groups.append([best_line])
        
        return groups
    
    def _interseccion_lineas(self, l1, l2):
        """Calcula intersección de dos líneas"""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        
        return (int(x), int(y))
    
    def _filtrar_lineas_por_angulo(self, lines, angle_threshold_deg=10):
        """
        Filtra líneas que se desvían mucho del ángulo MEDIANO.
        Esto elimina ruido y líneas diagonales.
        """
        if len(lines) < 2:
            return lines
        
        angles_deg = []
        for x1, y1, x2, y2 in lines:
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angles_deg.append(np.rad2deg(angle_rad))
        
        angles_deg = np.array(angles_deg)
        
        # Usar mediana es más robusto a outliers que la media
        median_angle = np.median(angles_deg)
        
        # Calcular diferencia absoluta, manejando el salto de -180 a 180
        diff = np.abs(angles_deg - median_angle)
        diff = np.minimum(diff, 360 - diff) # 179 y -179 están a 2 grados
        
        inlier_indices = np.where(diff < angle_threshold_deg)[0]
        
        print(f"   Filtrado por ángulo: {len(inlier_indices)} / {len(lines)} líneas mantenidas (mediana={median_angle:.1f} deg)")
        
        return lines[inlier_indices]

    def paso4_calcular_homografia(self, intersecciones, imagen):
        """
        PASO 4: RANSAC Homography
        Mapea coordenadas de grilla a píxeles de imagen
        """
        print("\n[4/6] Calculando homografía...")
        
        if not intersecciones or len(intersecciones) < 4:
            print("  ⚠️ Muy pocas intersecciones para homografía")
            return False
        
        # Preparar puntos
        img_points = []
        board_points = []
        
        for inter in intersecciones:
            x, y = inter['punto']
            i, j = inter['label']
            
            img_points.append([x, y])
            board_points.append([j * TILE_SIZE, i * TILE_SIZE])
        
        img_points = np.array(img_points, dtype=np.float32)
        board_points = np.array(board_points, dtype=np.float32)
        
        # Calcular homografía con RANSAC
        H, mask = cv2.findHomography(board_points, img_points, cv2.RANSAC, 5.0)
        
        if H is None:
            print("  ⚠️ No se pudo calcular homografía")
            return False
        
        self.H = cv2.findHomography(img_points, board_points, cv2.RANSAC, 5.0)[0]
        self.H_inv = H
        
        inliers = np.sum(mask)
        print(f"  Homografía calculada ({inliers}/{len(intersecciones)} inliers)")
        
        # Debug: dibujar grilla proyectada
        debug_grid = imagen.copy()
        for y in range(-5, 10):
            for x in range(-5, 10):
                board_p = np.array([x * TILE_SIZE, y * TILE_SIZE, 1])
                img_p = self.H_inv.dot(board_p)
                img_p = img_p[:2] / img_p[2]
                img_p = img_p.astype(int)
                
                if 0 <= img_p[0] < imagen.shape[1] and 0 <= img_p[1] < imagen.shape[0]:
                    cv2.circle(debug_grid, tuple(img_p), 3, (0, 255, 0), -1)
        
        cv2.imwrite("debug_05_homography.jpg", debug_grid)
        
        return True
    
    def paso5_extraer_piezas(self, imagen):
        """
        PASO 5: FindTiles usando homografía
        Extrae cada celda usando transformación de perspectiva
        """
        print("\n[5/6] Extrayendo piezas...")
        
        piezas = []
        h, w = imagen.shape[:2]
        
        # Probar grilla de -10 a 10
        for y_idx in range(-10, 10):
            for x_idx in range(-10, 10):
                # Proyectar esquinas de la celda ideal a la imagen
                tl = self._proyectar_punto(x_idx * TILE_SIZE, y_idx * TILE_SIZE)
                tr = self._proyectar_punto((x_idx+1) * TILE_SIZE, y_idx * TILE_SIZE)
                br = self._proyectar_punto((x_idx+1) * TILE_SIZE, (y_idx+1) * TILE_SIZE)
                bl = self._proyectar_punto(x_idx * TILE_SIZE, (y_idx+1) * TILE_SIZE)
                
                # Validar que esté dentro de la imagen
                puntos = [tl, tr, br, bl]
                if any(p is None for p in puntos):
                    continue
                
                if any(not (0 <= p[0] < w and 0 <= p[1] < h) for p in puntos):
                    continue
                
                # Extraer pieza con transformación de perspectiva
                src_points = np.array(puntos, dtype=np.float32)
                dst_points = np.array([[0, 0], [TILE_SIZE, 0], 
                                      [TILE_SIZE, TILE_SIZE], [0, TILE_SIZE]], 
                                     dtype=np.float32)
                
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                pieza = cv2.warpPerspective(imagen, M, (TILE_SIZE, TILE_SIZE))
                
                piezas.append({
                    'img': pieza,
                    'pos': (x_idx, y_idx)
                })
        
        print(f"  Piezas extraídas: {len(piezas)}")
        return piezas
    
    def _proyectar_punto(self, x, y):
        """Proyecta un punto de la grilla ideal a la imagen"""
        if self.H_inv is None:
            return None
        
        board_p = np.array([x, y, 1])
        img_p = self.H_inv.dot(board_p)
        
        if abs(img_p[2]) < 1e-10:
            return None
        
        img_p = img_p[:2] / img_p[2]
        return (int(img_p[0]), int(img_p[1]))
    
    def paso6_clasificar_piezas(self, piezas):
        """
        PASO 6: TileClassifier con NCC
        """
        print("\n[6/6] Clasificando piezas...")
        
        resultados = []
        
        for idx, pieza_data in enumerate(piezas):
            pieza_img = pieza_data['img']
            pos = pieza_data['pos']
            
            # Verificar si es una pieza válida (no solo mesa)
            gray = cv2.cvtColor(pieza_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 175)
            edge_density = np.mean(edges)
            
            if edge_density < 15:
                continue  # Es mesa vacía
            
            # Clasificar
            nombre, rotacion, confianza = self._clasificar_con_ncc(pieza_img, idx)
            
            if nombre != "vacio":
                resultados.append({
                    'nombre': nombre,
                    'rotacion': rotacion,
                    'confianza': confianza,
                    'pos': pos
                })
        
        print(f"  Piezas clasificadas: {len(resultados)}")
        return resultados
    
    def _clasificar_con_ncc(self, pieza_img, idx):
        """Clasificación con Normalized Cross Correlation"""
        if not self.referencias:
            return ("vacio", 0, 0.0)
        
        cv2.imwrite(f"debug_tile_{idx:03d}.jpg", pieza_img)
        
        gray = cv2.cvtColor(pieza_img, cv2.COLOR_BGR2GRAY)
        
        mejor = {'nombre': 'vacio', 'rotacion': 0, 'score': -1}
        
        for nombre, ref in self.referencias.items():
            for rot in range(4):
                gray_rot = np.rot90(gray, rot)
                
                # Normalized Cross Correlation
                result = cv2.matchTemplate(gray_rot, ref['gray'], cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                
                if score > mejor['score']:
                    mejor = {'nombre': nombre, 'rotacion': rot, 'score': score}
        
        if mejor['score'] < MIN_MATCH_SCORE:
            return ("vacio", 0, mejor['score'])
        
        return (mejor['nombre'], mejor['rotacion'], mejor['score'])
    
    def procesar(self, imagen_path):
        """Pipeline completo"""
        imagen_orig = cv2.imread(imagen_path)
        if imagen_orig is None:
            raise ValueError(f"Error cargando {imagen_path}")
        
        print("="*70)
        print("🎮 CARCASSONNE DETECTOR - Pipeline Completo")
        print("="*70)
        
        # PASO 1: Preprocesar
        imagen, blurred = self.paso1_preprocesar(imagen_orig)
        
        # PASO 2: Detectar líneas
        line_segments = self.paso2_detectar_bordes_lineas(blurred,imagen)
        
        if line_segments is None:
            return None
        
        # PASO 3: Intersecciones con votación
        intersecciones, labels = self.paso3_encontrar_intersecciones(imagen, line_segments)
        
        if not intersecciones:
            return None
        
        # PASO 4: Homografía
        if not self.paso4_calcular_homografia(intersecciones, imagen):
            return None
        
        # PASO 5: Extraer piezas
        piezas = self.paso5_extraer_piezas(imagen)
        
        # PASO 6: Clasificar
        resultados = self.paso6_clasificar_piezas(piezas)
        
        # Crear imagen final
        imagen_final = imagen.copy()
        for r in resultados:
            punto = self._proyectar_punto(r['pos'][0] * TILE_SIZE + TILE_SIZE//2, 
                                         r['pos'][1] * TILE_SIZE + TILE_SIZE//2)
            if punto:
                color = (0, 255, 0) if r['confianza'] >= MIN_MATCH_SCORE else (0, 165, 255)
                cv2.circle(imagen_final, punto, 15, color, -1)
                texto = f"{r['nombre'][:6]}R{r['rotacion']}"
                cv2.putText(imagen_final, texto, (punto[0]-20, punto[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite("resultado_final.jpg", imagen_final)
        
        print("\n" + "="*70)
        print("✅ COMPLETADO")
        print("="*70)
        
        return resultados


if __name__ == "__main__":
    try:
        detector = CarcassonneDetector("locetas_referencia")
        resultados = detector.procesar("tablero.jpg")
        
        if resultados:
            print(f"\n📊 {len(resultados)} piezas detectadas:")
            for r in resultados[:20]:
                status = "✅" if r['confianza'] >= MIN_MATCH_SCORE else "⚠️"
                print(f"{status} {r['pos']} {r['nombre']:12s} R{r['rotacion']} ({r['confianza']:.0%})")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
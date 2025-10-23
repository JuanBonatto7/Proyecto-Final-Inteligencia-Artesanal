import cv2
import numpy as np
import os
from itertools import product

# ============= CONFIGURACIÓN =============
TILE_SIZE = 64  # Tamaño estándar de pieza de Carcassonne
MIN_MATCH_SCORE = 0.50

class CarcassonneDetector:
    def __init__(self, referencias_path):
        self.referencias = self._cargar_referencias(referencias_path)
        print(f"📚 Referencias: {len(self.referencias)}")
        
    def _cargar_referencias(self, path):
        refs = {}
        if not os.path.exists(path):
            return refs
            
        for archivo in os.listdir(path):
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                nombre = os.path.splitext(archivo)[0]
                img = cv2.imread(os.path.join(path, archivo))
                if img is not None:
                    img_resize = cv2.resize(img, (TILE_SIZE, TILE_SIZE))
                    refs[nombre] = {
                        'original': img_resize,
                        'gray': cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                    }
                    print(f"  ✓ {nombre}")
        return refs
    
    def detectar_grilla_hough(self, imagen):
        """
        Detecta la grilla usando Hough Lines como en el proyecto de referencia.
        Basado en hough.py y find_intersections.py
        """
        print("\n🔍 PASO 1: Detectando grilla con Hough Transform")
        print("="*70)
        
        # Preprocesamiento como en el pipeline de referencia
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Blur (blur.py)
        blur = cv2.GaussianBlur(gray, (9, 9), 0.75)
        cv2.imwrite("debug_01_blur.jpg", blur)
        
        # Canny (canny.py)
        edges = cv2.Canny(blur, 100, 200)
        cv2.imwrite("debug_02_edges.jpg", edges)
        print("💾 debug_02_edges.jpg")
        
        # Dilate (dilate.py)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        cv2.imwrite("debug_03_edges_dilated.jpg", edges)
        
        # HoughLinesP (hough.py)
        line_segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=30
        )
        
        if line_segments is None:
            print("  ⚠️ No se detectaron líneas")
            return None, None
        
        line_segments = np.squeeze(line_segments, axis=1)
        print(f"  📊 Segmentos de línea detectados: {len(line_segments)}")
        
        # Dibujar líneas detectadas
        debug_lines = imagen.copy()
        for x1, y1, x2, y2 in line_segments:
            cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("debug_04_hough_lines.jpg", debug_lines)
        print("💾 debug_04_hough_lines.jpg - Todas las líneas detectadas")
        
        # Clasificar en horizontales y verticales (line_classifier.py)
        h_lines, v_lines = self._clasificar_lineas(line_segments)
        
        print(f"  📊 Líneas horizontales: {len(h_lines)}")
        print(f"  📊 Líneas verticales: {len(v_lines)}")
        
        # Debug con colores
        debug_classified = imagen.copy()
        for x1, y1, x2, y2 in h_lines:
            cv2.line(debug_classified, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(debug_classified, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite("debug_05_classified_lines.jpg", debug_classified)
        print("💾 debug_05_classified_lines.jpg - Verde=H, Azul=V")
        
        return h_lines, v_lines
    
    def _clasificar_lineas(self, line_segments):
        """Clasifica líneas en horizontales y verticales por ángulo"""
        h_lines = []
        v_lines = []
        
        for x1, y1, x2, y2 in line_segments:
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:
                angulo = 90
            else:
                angulo = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
            
            # Horizontal: cerca de 0° o 180°
            if angulo < 20 or angulo > 160:
                h_lines.append([x1, y1, x2, y2])
            # Vertical: cerca de 90°
            elif 70 < angulo < 110:
                v_lines.append([x1, y1, x2, y2])
        
        return np.array(h_lines) if h_lines else np.array([]), \
               np.array(v_lines) if v_lines else np.array([])
    
    def encontrar_intersecciones(self, imagen, h_lines, v_lines):
        """
        Encuentra intersecciones entre líneas H y V.
        OPTIMIZADO: Filtra líneas primero para evitar explosión combinatoria.
        """
        print("\n🔍 PASO 2: Encontrando intersecciones")
        print("="*70)
        
        if len(h_lines) == 0 or len(v_lines) == 0:
            print("  ⚠️ No hay líneas ortogonales")
            return []
        
        print(f"  📊 Líneas H: {len(h_lines)}, V: {len(v_lines)}")
        print(f"  ⏳ Calculando {len(h_lines) * len(v_lines)} intersecciones posibles...")
        
        # OPTIMIZACIÓN 1: Limitar a las líneas más largas
        MAX_LINES = 30
        
        if len(h_lines) > MAX_LINES:
            # Ordenar por longitud y tomar las más largas
            h_lengths = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in h_lines]
            h_indices = np.argsort(h_lengths)[-MAX_LINES:]
            h_lines = h_lines[h_indices]
            print(f"  🔧 Reducidas líneas H a {len(h_lines)} (las más largas)")
        
        if len(v_lines) > MAX_LINES:
            v_lengths = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in v_lines]
            v_indices = np.argsort(v_lengths)[-MAX_LINES:]
            v_lines = v_lines[v_indices]
            print(f"  🔧 Reducidas líneas V a {len(v_lines)} (las más largas)")
        
        intersecciones = []
        h, w = imagen.shape[:2]
        
        # Convertir segmentos a ecuación de línea ax + by + c = 0
        def linea_desde_segmento(x1, y1, x2, y2):
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            return a, b, c
        
        # Encontrar todas las intersecciones H x V
        total = len(h_lines) * len(v_lines)
        procesadas = 0
        
        for hline in h_lines:
            a1, b1, c1 = linea_desde_segmento(*hline)
            
            for vline in v_lines:
                procesadas += 1
                if procesadas % 100 == 0:
                    print(f"  ⏳ Progreso: {procesadas}/{total}", end='\r')
                
                a2, b2, c2 = linea_desde_segmento(*vline)
                
                # Resolver sistema de ecuaciones
                det = a1*b2 - a2*b1
                if abs(det) < 1e-10:
                    continue
                
                x = (b1*c2 - b2*c1) / det
                y = (a2*c1 - a1*c2) / det
                
                # Validar que esté dentro de la imagen
                if 0 <= x < w and 0 <= y < h:
                    intersecciones.append([int(x), int(y)])
        
        print(f"\n  📊 Intersecciones brutas: {len(intersecciones)}")
        
        # Eliminar duplicados cercanos
        intersecciones = self._eliminar_duplicados(intersecciones, threshold=25)
        
        print(f"  ✅ Intersecciones únicas: {len(intersecciones)}")
        
        # Debug
        debug_int = imagen.copy()
        for x, y in intersecciones:
            cv2.circle(debug_int, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(debug_int, (x, y), 10, (255, 255, 255), 2)
        cv2.imwrite("debug_06_intersections.jpg", debug_int)
        print("💾 debug_06_intersections.jpg")
        
        return np.array(intersecciones)
    
    def _eliminar_duplicados(self, puntos, threshold=20):
        """Elimina puntos muy cercanos entre sí"""
        if not puntos:
            return []
        
        unicos = [puntos[0]]
        
        for p in puntos[1:]:
            es_duplicado = False
            for u in unicos:
                dist = np.sqrt((p[0]-u[0])**2 + (p[1]-u[1])**2)
                if dist < threshold:
                    es_duplicado = True
                    break
            
            if not es_duplicado:
                unicos.append(p)
        
        return unicos
    
    def crear_grilla_desde_intersecciones(self, intersecciones):
        """
        Agrupa intersecciones en filas y columnas.
        Basado en find_intersections.py
        """
        print("\n🔍 PASO 3: Creando grilla")
        print("="*70)
        
        if len(intersecciones) < 4:
            print("  ⚠️ Muy pocas intersecciones")
            return []
        
        # Agrupar por coordenadas X e Y
        xs = [p[0] for p in intersecciones]
        ys = [p[1] for p in intersecciones]
        
        # Agrupar coordenadas similares
        def agrupar_coords(coords, threshold=30):
            sorted_coords = sorted(set(coords))
            grupos = [[sorted_coords[0]]]
            
            for c in sorted_coords[1:]:
                if c - grupos[-1][-1] < threshold:
                    grupos[-1].append(c)
                else:
                    grupos.append([c])
            
            return [int(np.mean(g)) for g in grupos]
        
        xs_agrupadas = agrupar_coords(xs)
        ys_agrupadas = agrupar_coords(ys)
        
        print(f"  📊 Columnas: {len(xs_agrupadas)}")
        print(f"  📊 Filas: {len(ys_agrupadas)}")
        
        # Crear celdas
        celdas = []
        for i in range(len(ys_agrupadas) - 1):
            for j in range(len(xs_agrupadas) - 1):
                x1 = xs_agrupadas[j]
                x2 = xs_agrupadas[j+1]
                y1 = ys_agrupadas[i]
                y2 = ys_agrupadas[i+1]
                
                w = x2 - x1
                h = y2 - y1
                
                # Validar tamaño mínimo
                if w > 20 and h > 20:
                    celdas.append({
                        'bbox': (x1, y1, x2, y2),
                        'row': i,
                        'col': j
                    })
        
        print(f"  ✅ Celdas válidas: {len(celdas)}")
        return celdas
    
    def extraer_pieza(self, imagen, bbox):
        """Extrae y normaliza una pieza. Basado en find_tiles.py"""
        x1, y1, x2, y2 = bbox
        
        h, w = imagen.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        pieza = imagen[y1:y2, x1:x2]
        
        if pieza.size == 0 or pieza.shape[0] < 5 or pieza.shape[1] < 5:
            return None
        
        # Redimensionar a tamaño estándar
        pieza_resize = cv2.resize(pieza, (TILE_SIZE, TILE_SIZE))
        return pieza_resize
    
    def clasificar_pieza(self, pieza_img, pieza_id):
        """Clasifica usando template matching. Basado en tile_classifier.py"""
        if pieza_img is None or not self.referencias:
            return ("vacio", 0, 0.0)
        
        cv2.imwrite(f"debug_pieza_{pieza_id:02d}.jpg", pieza_img)
        
        pieza_gray = cv2.cvtColor(pieza_img, cv2.COLOR_BGR2GRAY)
        
        mejor = {'nombre': 'vacio', 'rotacion': 0, 'score': 0.0}
        
        for nombre, ref_data in self.referencias.items():
            for rot in range(4):
                pieza_rot = np.rot90(pieza_gray, rot)
                
                res = cv2.matchTemplate(pieza_rot, ref_data['gray'], 
                                       cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
                
                if score > mejor['score']:
                    mejor = {'nombre': nombre, 'rotacion': rot, 'score': score}
        
        if mejor['score'] < MIN_MATCH_SCORE:
            return ("vacio", 0, mejor['score'])
        
        return (mejor['nombre'], mejor['rotacion'], mejor['score'])
    
    def procesar_tablero(self, imagen_path):
        """Pipeline completo inspirado en detect.py"""
        if not os.path.exists(imagen_path):
            raise ValueError(f"❌ No existe: {imagen_path}")
        
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            raise ValueError(f"❌ Error al cargar")
        
        print("\n" + "="*70)
        print("🎮 CARCASSONNE DETECTOR v6.0 - Método Grilla (Inspirado en GitHub)")
        print("="*70)
        print(f"📸 Tamaño: {imagen.shape[1]}x{imagen.shape[0]}")
        
        # PASO 1: Detectar líneas
        h_lines, v_lines = self.detectar_grilla_hough(imagen)
        
        if h_lines is None or v_lines is None:
            print("\n❌ No se detectó grilla")
            return None
        
        # PASO 2: Intersecciones
        intersecciones = self.encontrar_intersecciones(imagen, h_lines, v_lines)
        
        if len(intersecciones) < 4:
            print("\n❌ Muy pocas intersecciones")
            return None
        
        # PASO 3: Crear grilla
        celdas = self.crear_grilla_desde_intersecciones(intersecciones)
        
        if not celdas:
            print("\n❌ No se creó la grilla")
            return None
        
        # PASO 4: Clasificar
        print(f"\n🔍 PASO 4: Clasificando {len(celdas)} celdas")
        print("="*70)
        
        resultados = []
        imagen_resultado = imagen.copy()
        
        # Dibujar grilla
        for celda in celdas:
            x1, y1, x2, y2 = celda['bbox']
            cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        for idx, celda in enumerate(celdas, 1):
            bbox = celda['bbox']
            pieza_img = self.extraer_pieza(imagen, bbox)
            
            nombre, rotacion, confianza = self.clasificar_pieza(pieza_img, idx)
            
            resultados.append({
                'id': idx,
                'nombre': nombre,
                'rotacion': rotacion,
                'confianza': confianza,
                'row': celda['row'],
                'col': celda['col'],
                'bbox': bbox
            })
            
            # Dibujar solo si no es vacío
            if nombre != "vacio":
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if confianza >= MIN_MATCH_SCORE else (0, 165, 255)
                
                cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), color, 2)
                
                texto = f"{nombre[:6]}R{rotacion}"
                cv2.putText(imagen_resultado, texto, (x1+3, y1+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                status = "✅" if confianza >= MIN_MATCH_SCORE else "⚠️"
                print(f"  {status} [{celda['row']},{celda['col']}] {nombre:12s} R{rotacion} ({confianza:.1%})")
        
        cv2.imwrite("resultado_final.jpg", imagen_resultado)
        print("\n💾 resultado_final.jpg")
        
        # Crear matriz
        matriz = self._crear_matriz(resultados)
        
        print("\n" + "="*70)
        print("📊 MATRIZ FINAL")
        print("="*70)
        for i, fila in enumerate(matriz):
            print(f"Fila {i}: {fila}")
        
        return resultados, matriz
    
    def _crear_matriz(self, resultados):
        """Convierte resultados en matriz 2D"""
        if not resultados:
            return []
        
        # Filtrar solo no vacíos
        no_vacios = [r for r in resultados if r['nombre'] != 'vacio']
        
        if not no_vacios:
            return []
        
        max_row = max(r['row'] for r in no_vacios)
        max_col = max(r['col'] for r in no_vacios)
        
        matriz = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        for r in no_vacios:
            matriz[r['row']][r['col']] = f"{r['nombre'][:8]}R{r['rotacion']}"
        
        return matriz


if __name__ == "__main__":
    print("🚀 Carcassonne Detector v6.0 - Método Grilla\n")
    
    try:
        detector = CarcassonneDetector("locetas_referencia")
        resultado = detector.procesar_tablero("tablero.jpg")
        
        if resultado:
            print("\n✅ Procesamiento exitoso")
        else:
            print("\n⚠️ Revisa las imágenes debug_*.jpg")
            print("   Especialmente debug_04_hough_lines.jpg")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import json
from scipy import ndimage

# ============= CONFIG =============

@dataclass
class Config:
    TILE_SIZE: int = 150
    
    # 🆕 Grid-based fallback
    USE_GRID_FALLBACK: bool = True
    EXPECTED_GRID_SIZE: int = 5  # Tablero aproximado NxN
    
    # Watershed
    WATERSHED_THRESHOLD: float = 0.4  # Distancia mínima entre piezas
    
    MIN_ASPECT: float = 0.65
    MAX_ASPECT: float = 1.5
    GRID_TOLERANCE: int = 80
    CONFIDENCE_THRESHOLD: float = 0.4
    
    DEBUG_MODE: bool = True
    SAVE_ALL_STEPS: bool = True
    
    REF_FOLDER: str = "locetas_referencia"
    OUTPUT_FOLDER: str = "output"
    DEBUG_FOLDER: str = "output/debug"
    
    TILE_NAMES: List[str] = None
    
    def __post_init__(self):
        if self.TILE_NAMES is None:
            self.TILE_NAMES = [chr(i) for i in range(ord('A'), ord('Z'))]
        
        for folder in [self.OUTPUT_FOLDER, self.DEBUG_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)


# ============= UTILIDADES =============

class ImageUtils:
    @staticmethod
    def save_debug(image, name, config):
        if config.DEBUG_MODE and config.SAVE_ALL_STEPS:
            path = f"{config.DEBUG_FOLDER}/{name}"
            cv2.imwrite(path, image)
            print(f"  💾 {name}")
    
    @staticmethod
    def enhance_contrast(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ============= DETECTOR DE TABLERO =============

class BoardDetector:
    """Detecta y recorta el tablero principal"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect(self, image):
        """Detecta el tablero completo y lo recorta"""
        print("\n🔍 Detectando tablero...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold para separar tablero de mesa
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfología para cerrar huecos
        kernel = np.ones((15, 15), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        ImageUtils.save_debug(thresh, "board_mask.jpg", self.config)
        
        # Encontrar el contorno más grande
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("   ⚠️  No se detectó contorno, usando imagen completa")
            return image, (0, 0, image.shape[1], image.shape[0])
        
        # Contorno más grande (el tablero)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Margen de seguridad
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        
        board = image[y:y+h, x:x+w]
        
        print(f"   ✅ Tablero detectado: {w}x{h} px")
        print(f"   Posición: ({x}, {y})")
        
        return board, (x, y, w, h)


# ============= SEGMENTADOR CON WATERSHED =============

class TileSegmenter:
    """Segmentador usando Watershed para separar piezas pegadas"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def segment(self, board_image):
        print("\n🧩 SEGMENTANDO CON WATERSHED")
        print("="*60)
        
        ImageUtils.save_debug(board_image, "00_board.jpg", self.config)
        
        # Intentar watershed primero
        tiles = self._segment_watershed(board_image)
        
        # 🆕 FALLBACK: Grid-based si falla watershed
        if len(tiles) < 3:  # Muy pocas piezas detectadas
            print("\n⚠️  Watershed falló, usando GRID-BASED FALLBACK")
            tiles = self._segment_grid_based(board_image)
        
        return tiles
    
    def _segment_watershed(self, image):
        """Segmentación usando Distance Transform + Watershed"""
        print("\n🌊 Aplicando Watershed...")
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        ImageUtils.save_debug(enhanced, "01_clahe.jpg", self.config)
        
        # Threshold Otsu
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ImageUtils.save_debug(thresh, "02_threshold.jpg", self.config)
        
        # Limpiar ruido
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        ImageUtils.save_debug(opening, "03_opening.jpg", self.config)
        
        # Sure background (dilatación)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground (Distance Transform)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ImageUtils.save_debug((dist_transform * 20).astype(np.uint8), "04_distance.jpg", self.config)
        
        # Encontrar picos (centros de piezas)
        _, sure_fg = cv2.threshold(dist_transform, 
                                   self.config.WATERSHED_THRESHOLD * dist_transform.max(), 
                                   255, 0)
        sure_fg = np.uint8(sure_fg)
        ImageUtils.save_debug(sure_fg, "05_sure_fg.jpg", self.config)
        
        # Región desconocida
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marcadores
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Background no es 0
        markers[unknown == 255] = 0
        
        print(f"   Marcadores iniciales: {markers.max()}")
        
        # Aplicar Watershed
        markers = cv2.watershed(image, markers)
        
        # Contar regiones
        num_regions = markers.max()
        print(f"   ✅ Regiones detectadas: {num_regions}")
        
        # Visualizar marcadores
        marker_viz = np.zeros_like(image)
        marker_viz[markers == -1] = [0, 0, 255]  # Bordes en rojo
        for i in range(1, num_regions + 1):
            color = np.random.randint(0, 255, 3).tolist()
            marker_viz[markers == i] = color
        ImageUtils.save_debug(marker_viz, "06_watershed_result.jpg", self.config)
        
        # Extraer piezas
        tiles = []
        debug_img = image.copy()
        
        for marker_id in range(2, num_regions + 1):  # Saltar 1 (background)
            mask = np.uint8(markers == marker_id) * 255
            
            # Encontrar contorno
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            # Filtro básico de área
            if area < 1000:  # Muy pequeño
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            
            # Filtro de aspecto
            if not (self.config.MIN_ASPECT < aspect < self.config.MAX_ASPECT):
                continue
            
            # Extraer pieza
            tile_img = image[y:y+h, x:x+w].copy()
            
            # Centro
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            tiles.append({
                'image': tile_img,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'contour': cnt,
                'area': float(area),
                'aspect_ratio': float(aspect)
            })
            
            # Debug
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(debug_img, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(debug_img, str(len(tiles)), (cx-10, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        ImageUtils.save_debug(debug_img, "07_tiles_watershed.jpg", self.config)
        
        print(f"   Piezas válidas: {len(tiles)}")
        
        return tiles
    
    def _segment_grid_based(self, image):
        """🆕 Segmentación asumiendo grid regular (FALLBACK)"""
        print("\n📐 Segmentación basada en GRID...")
        
        h, w = image.shape[:2]
        
        # Detectar el área útil (sin bordes de mesa)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Encontrar región del tablero
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            x, y, board_w, board_h = cv2.boundingRect(main_cnt)
        else:
            x, y, board_w, board_h = 0, 0, w, h
        
        print(f"   Área del tablero: {board_w}x{board_h}")
        
        # Calcular grid dinámicamente
        # Intentar detectar piezas de diferentes tamaños
        grid_sizes = [3, 4, 5, 6, 7]  # Probar diferentes tamaños de grid
        
        best_grid = None
        best_score = 0
        
        for grid_n in grid_sizes:
            cell_w = board_w // grid_n
            cell_h = board_h // grid_n
            
            # Verificar si las celdas son aproximadamente cuadradas
            aspect = cell_w / float(cell_h)
            if not (0.7 < aspect < 1.4):
                continue
            
            # Score: preferir celdas de tamaño razonable
            cell_area = cell_w * cell_h
            if 10000 < cell_area < 100000:  # Rango razonable para una pieza
                score = 100 - abs(grid_n - self.config.EXPECTED_GRID_SIZE) * 10
                if score > best_score:
                    best_score = score
                    best_grid = (grid_n, cell_w, cell_h)
        
        if best_grid is None:
            # Fallback: usar EXPECTED_GRID_SIZE
            grid_n = self.config.EXPECTED_GRID_SIZE
            cell_w = board_w // grid_n
            cell_h = board_h // grid_n
        else:
            grid_n, cell_w, cell_h = best_grid
        
        print(f"   Grid seleccionado: {grid_n}x{grid_n}")
        print(f"   Tamaño de celda: {cell_w}x{cell_h} px")
        
        tiles = []
        debug_img = image.copy()
        
        # Dividir en grid
        for row in range(grid_n):
            for col in range(grid_n):
                # Calcular posición de la celda
                x_start = x + col * cell_w
                y_start = y + row * cell_h
                
                # Asegurar que no se sale de la imagen
                x_end = min(x_start + cell_w, x + board_w)
                y_end = min(y_start + cell_h, y + board_h)
                
                # Extraer celda
                cell = image[y_start:y_end, x_start:x_end]
                
                # Verificar que no esté vacía (mostly background)
                if self._is_empty_cell(cell):
                    continue
                
                # Centro de la celda
                cx = (x_start + x_end) // 2
                cy = (y_start + y_end) // 2
                
                tiles.append({
                    'image': cell,
                    'bbox': (x_start, y_start, x_end - x_start, y_end - y_start),
                    'center': (cx, cy),
                    'contour': None,
                    'area': float((x_end - x_start) * (y_end - y_start)),
                    'aspect_ratio': float((x_end - x_start) / (y_end - y_start))
                })
                
                # Debug
                cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.circle(debug_img, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(debug_img, f"{row},{col}", (cx-15, cy+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        ImageUtils.save_debug(debug_img, "07_tiles_grid.jpg", self.config)
        
        print(f"   ✅ Celdas extraídas: {len(tiles)}")
        
        return tiles
    
    def _is_empty_cell(self, cell):
        """Verifica si una celda está vacía (fondo blanco/mesa)"""
        if cell.size == 0:
            return True
        
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        
        # Si la mayoría de píxeles son muy claros u oscuros (fondo)
        white_ratio = np.sum(gray > 200) / gray.size
        black_ratio = np.sum(gray < 50) / gray.size
        
        return (white_ratio > 0.8) or (black_ratio > 0.8)
    
    def create_grid(self, tiles):
        """Organiza tiles en grilla"""
        if not tiles:
            return []
        
        print("\n📐 Organizando en grilla...")
        
        centers = np.array([t['center'] for t in tiles])
        y_coords = centers[:, 1]
        
        # Clustering en Y
        sorted_indices = np.argsort(y_coords)
        rows = []
        current_row = [sorted_indices[0]]
        
        for idx in sorted_indices[1:]:
            if abs(y_coords[idx] - np.mean(y_coords[current_row])) < self.config.GRID_TOLERANCE:
                current_row.append(idx)
            else:
                rows.append(current_row)
                current_row = [idx]
        rows.append(current_row)
        
        print(f"   Filas: {len(rows)}")
        
        # Ordenar por X
        for row_idx, row in enumerate(rows):
            row_sorted = sorted(row, key=lambda i: centers[i, 0])
            for col_idx, tile_idx in enumerate(row_sorted):
                tiles[tile_idx]['row'] = row_idx
                tiles[tile_idx]['col'] = col_idx
        
        grid_tiles = [t for t in tiles if 'row' in t]
        return grid_tiles


# ============= CLASIFICADOR =============

class TileClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.references = {}
        self._load_references()
    
    def _load_references(self):
        ref_path = Path(self.config.REF_FOLDER)
        if not ref_path.exists():
            return
        
        print(f"\n📚 Cargando referencias...")
        
        loaded = 0
        for letter in self.config.TILE_NAMES:
            tile_files = list(ref_path.glob(f"{letter}.*"))
            if not tile_files:
                continue
            
            img = cv2.imread(str(tile_files[0]))
            if img is None:
                continue
            
            img = cv2.resize(img, (self.config.TILE_SIZE, self.config.TILE_SIZE))
            self.references[letter] = []
            
            for rot in range(4):
                rotated = np.rot90(img, rot).copy()
                features = self._extract_features(rotated)
                
                self.references[letter].append({
                    'rotation': rot * 90,
                    'image': rotated,
                    'features': features
                })
            
            loaded += 1
        
        print(f"✅ {loaded} losetas x 4 = {loaded*4} templates\n")
    
    def _extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=500, fastThreshold=5)
        kp, des = orb.detectAndCompute(gray, None)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist = np.concatenate([
            cv2.normalize(hist_h, hist_h).flatten(),
            cv2.normalize(hist_s, hist_s).flatten()
        ])
        
        template = cv2.resize(gray, (64, 64))
        
        return {
            'orb_kp': kp,
            'orb_des': des,
            'histogram': hist,
            'template': template
        }
    
    def classify(self, tile_image):
        if not self.references:
            return ("?", 0, 0.0)
        
        tile_resized = cv2.resize(tile_image, (self.config.TILE_SIZE, self.config.TILE_SIZE))
        tile_features = self._extract_features(tile_resized)
        
        best_match = {'letter': '?', 'rotation': 0, 'score': 0.0}
        
        for letter, rotations in self.references.items():
            for ref in rotations:
                score = self._compare_features(tile_features, ref['features'])
                
                if score > best_match['score']:
                    best_match = {
                        'letter': letter,
                        'rotation': ref['rotation'] // 90,
                        'score': score
                    }
        
        return (best_match['letter'], best_match['rotation'], min(1.0, best_match['score']))
    
    def _compare_features(self, feat1, feat2):
        score = 0.0
        weights = {'orb': 0.5, 'histogram': 0.25, 'template': 0.25}
        
        if feat1['orb_des'] is not None and feat2['orb_des'] is not None:
            try:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(feat1['orb_des'], feat2['orb_des'], k=2)
                
                good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
                orb_score = len(good) / 30.0
                score += min(1.0, orb_score) * weights['orb']
            except:
                pass
        
        hist_corr = cv2.compareHist(feat1['histogram'], feat2['histogram'], cv2.HISTCMP_CORREL)
        score += max(0, hist_corr) * weights['histogram']
        
        result = cv2.matchTemplate(feat1['template'], feat2['template'], cv2.TM_CCOEFF_NORMED)
        score += max(0, result[0][0]) * weights['template']
        
        return score


# ============= DETECTOR PRINCIPAL =============

class CarcassonneDetector:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        Path(self.config.OUTPUT_FOLDER).mkdir(exist_ok=True)
        Path(self.config.DEBUG_FOLDER).mkdir(exist_ok=True)
        
        self.board_detector = BoardDetector(self.config)
        self.tile_segmenter = TileSegmenter(self.config)
        self.tile_classifier = TileClassifier(self.config)
    
    def process(self, image_path: str):
        print("\n" + "="*70)
        print("🎮 CARCASSONNE DETECTOR v4.0 - WATERSHED + GRID")
        print("="*70)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar: {image_path}")
        
        print(f"\n✅ Imagen: {img.shape[1]}x{img.shape[0]} px")
        
        # Resize si es muy grande
        h, w = img.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Detectar tablero
        board, board_bbox = self.board_detector.detect(img)
        
        # Segmentar
        tiles = self.tile_segmenter.segment(board)
        
        if not tiles:
            print("\n❌ No se detectaron piezas")
            return None
        
        tiles = self.tile_segmenter.create_grid(tiles)
        
        print(f"\n🔍 Clasificando {len(tiles)} piezas...")
        
        results = []
        result_img = board.copy()
        
        for tile in tiles:
            letter, rotation, confidence = self.tile_classifier.classify(tile['image'])
            
            results.append({
                'row': int(tile['row']),
                'col': int(tile['col']),
                'letter': str(letter),
                'rotation': int(rotation),
                'confidence': float(confidence)
            })
            
            x, y, w, h = tile['bbox']
            color = (0, 255, 0) if confidence > self.config.CONFIDENCE_THRESHOLD else (0, 165, 255)
            
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_img, f"{letter}R{rotation}", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(result_img, f"{confidence:.0%}", (x+5, y+h-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(f"{self.config.OUTPUT_FOLDER}/resultado_final.jpg", result_img)
        
        matrix = self._create_matrix(results)
        
        with open(f"{self.config.OUTPUT_FOLDER}/resultado.json", 'w') as f:
            json.dump({
                'matrix': [[{'letter': str(c[0]), 'rotation': int(c[1])} if c else None 
                           for c in row] for row in matrix],
                'tiles': results
            }, f, indent=2)
        
        return matrix, results
    
    def _create_matrix(self, results):
        if not results:
            return []
        
        max_row = max(r['row'] for r in results)
        max_col = max(r['col'] for r in results)
        matrix = [[None] * (max_col + 1) for _ in range(max_row + 1)]
        
        for r in results:
            matrix[r['row']][r['col']] = (r['letter'], r['rotation'])
        
        return matrix
    
    def print_results(self, matrix, results):
        print("\n" + "="*70)
        print("📊 RESULTADOS")
        print("="*70 + "\n")
        
        for i, row in enumerate(matrix):
            print(f"Fila {i}: ", end="")
            for cell in row:
                if cell:
                    print(f"[{cell[0]}R{cell[1]}]", end=" ")
                else:
                    print("[   ]", end=" ")
            print()
        
        print(f"\n{'Pos':^10} | {'Pieza':^8} | {'Rot':^5} | {'Conf':^10}")
        print("-"*50)
        
        high_conf = sum(1 for r in results if r['confidence'] > self.config.CONFIDENCE_THRESHOLD)
        
        for r in sorted(results, key=lambda x: (x['row'], x['col'])):
            status = "✅" if r['confidence'] > self.config.CONFIDENCE_THRESHOLD else "⚠️ "
            print(f"{status} [{r['row']:2d},{r['col']:2d}] | {r['letter']:^6s} | "
                  f"{r['rotation']:^5d} | {r['confidence']:^9.1%}")
        
        print(f"\n📊 Alta confianza: {high_conf}/{len(results)}")
        print("✅ Completado\n")


# ============= MAIN =============

if __name__ == "__main__":
    print("\n🎮 CARCASSONNE v4.0 - WATERSHED + GRID FALLBACK\n")
    
    # 🆕 Necesita scipy
    try:
        from scipy import ndimage
    except ImportError:
        print("❌ Instala scipy:")
        print("   pip install scipy\n")
        input("ENTER...")
        exit()
    
    config = Config()
    config.USE_GRID_FALLBACK = True
    config.EXPECTED_GRID_SIZE = 5  # 🔧 AJUSTA según tu tablero (3, 4, 5, 6, etc.)
    config.WATERSHED_THRESHOLD = 0.4  # 🔧 Ajusta si detecta muy pocas/muchas piezas
    
    tablero_path = None
    for nombre in ["tablero.jpg", "tablero.png"]:
        if Path(nombre).exists():
            tablero_path = nombre
            break
    
    if not tablero_path:
        print("❌ No hay tablero.jpg\n")
        input("ENTER...")
        exit()
    
    print(f"✅ Tablero: {tablero_path}")
    print(f"🔧 Grid esperado: {config.EXPECTED_GRID_SIZE}x{config.EXPECTED_GRID_SIZE}\n")
    
    input("📌 ENTER para iniciar...\n")
    
    try:
        detector = CarcassonneDetector(config)
        result = detector.process(tablero_path)
        
        if result:
            matrix, tiles = result
            detector.print_results(matrix, tiles)
            
            print(f"\n💾 Revisa:")
            print(f"   • {config.OUTPUT_FOLDER}/resultado_final.jpg")
            print(f"   • {config.DEBUG_FOLDER}/07_tiles_*.jpg")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        import traceback
        traceback.print_exc()
    
    input("\nENTER para salir...")
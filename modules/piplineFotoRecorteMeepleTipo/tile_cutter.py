import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Tile:
    """Representa una loseta detectada"""
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray
    grid_row: int
    grid_col: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class ReferencePoint:
    """Punto de referencia marcado por el usuario"""
    x: int
    y: int
    width: int
    height: int
    grid_row: int = 0
    grid_col: int = 0


class CarcassonneTileDetector:
    def __init__(self):
        self.image = None
        self.reference_points: List[ReferencePoint] = []
        self.tiles: List[Tile] = []
        self.selecting = False
        self.start_point = None
        self.current_rect = None
        self.avg_tile_size = None
        
    def load_image(self, image_path: str) -> bool:
        """Carga la imagen desde el archivo"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: No se pudo cargar la imagen {image_path}")
            return False
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para capturar la selección del usuario"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.current_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.current_rect = (self.start_point[0], self.start_point[1], 
                                    x - self.start_point[0], y - self.start_point[1])
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecting:
                self.selecting = False
                w = x - self.start_point[0]
                h = y - self.start_point[1]
                
                # Asegurar valores positivos
                if w < 0:
                    self.start_point = (x, self.start_point[1])
                    w = abs(w)
                if h < 0:
                    self.start_point = (self.start_point[0], y)
                    h = abs(h)
                
                if w > 10 and h > 10:
                    ref = ReferencePoint(self.start_point[0], self.start_point[1], w, h)
                    self.reference_points.append(ref)
    
    def set_reference_points_from_coords(self, coords_list: List[dict]) -> bool:
        """Establece puntos de referencia desde coordenadas proporcionadas
        
        Args:
            coords_list: Lista de dicts con {x, y, width, height} en coordenadas de imagen original
        
        Returns:
            True si se establecieron correctamente
        """
        self.reference_points = []
        
        for coord in coords_list:
            ref = ReferencePoint(
                x=int(coord['x']),
                y=int(coord['y']),
                width=int(coord['width']),
                height=int(coord['height'])
            )
            self.reference_points.append(ref)
        
        return len(self.reference_points) >= 3
    
    def select_reference_tiles(self, num_points: int = 8, auto_detect: bool = False) -> bool:
        """Permite al usuario seleccionar múltiples losetas de referencia o detectarlas automáticamente"""
        
        if auto_detect:
            return self._auto_detect_reference_tiles(num_points)
        
        window_name = "Seleccionar Losetas de Referencia"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        h, w = self.image.shape[:2]
        max_width = 1400
        max_height = 900
        scale = min(max_width / w, max_height / h, 1.0)
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
        
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.reference_points = []
        
        while True:
            display = self.image.copy()
            
            # Dibujar rectángulo de selección actual
            if self.current_rect is not None:
                x, y, w, h = self.current_rect
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Dibujar puntos de referencia ya seleccionados
            colors = [
                (255, 0, 0),    # Rojo
                (0, 255, 0),    # Verde
                (0, 0, 255),    # Azul
                (255, 255, 0),  # Amarillo
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Naranja
                (128, 0, 255)   # Violeta
            ]
            for i, ref in enumerate(self.reference_points):
                color = colors[i % 8]
                cv2.rectangle(display, (ref.x, ref.y), (ref.x + ref.width, ref.y + ref.height), color, 3)
                cv2.putText(display, f"#{i+1}", (ref.x, ref.y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Mostrar contador
            cv2.putText(display, f"Losetas: {len(self.reference_points)}/{num_points}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if len(self.reference_points) >= num_points:
                    cv2.destroyWindow(window_name)
                    return True
            elif key == ord('u'):  # Deshacer
                if self.reference_points:
                    self.reference_points.pop()
            elif key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return False
        
        return False
    
    def _auto_detect_reference_tiles(self, num_points: int = 8) -> bool:
        """Detecta automáticamente puntos de referencia sin intervención del usuario"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold adaptativo para detectar bordes de losetas
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Detectar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área y proporción
        h, w = self.image.shape[:2]
        min_area = (w * h) * 0.005  # Al menos 0.5% del área total
        max_area = (w * h) * 0.1    # Máximo 10% del área total
        
        candidate_tiles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 0
                
                # Buscar rectángulos aproximadamente cuadrados (losetas)
                if 0.7 < aspect_ratio < 1.3:  # Tolerancia para cuadrados
                    candidate_tiles.append({
                        'x': x, 'y': y, 'w': cw, 'h': ch,
                        'area': area,
                        'center': (x + cw//2, y + ch//2)
                    })
        
        if len(candidate_tiles) < num_points:
            print(f"Solo se detectaron {len(candidate_tiles)} losetas, se necesitan {num_points}")
            return False
        
        # Ordenar por área (las más grandes primero) y tomar una muestra distribuida
        candidate_tiles.sort(key=lambda t: t['area'], reverse=True)
        
        # Seleccionar puntos bien distribuidos en el espacio
        selected = []
        grid_divisions = int(np.sqrt(num_points))
        
        # Dividir imagen en cuadrantes y seleccionar uno por región
        for i in range(grid_divisions):
            for j in range(grid_divisions):
                if len(selected) >= num_points:
                    break
                
                # Definir región
                region_x = (w // grid_divisions) * j
                region_y = (h // grid_divisions) * i
                region_w = w // grid_divisions
                region_h = h // grid_divisions
                
                # Buscar loseta en esta región
                for tile in candidate_tiles:
                    cx, cy = tile['center']
                    if (region_x <= cx < region_x + region_w and 
                        region_y <= cy < region_y + region_h and
                        tile not in selected):
                        selected.append(tile)
                        break
        
        # Si no tenemos suficientes, completar con las más grandes restantes
        for tile in candidate_tiles:
            if len(selected) >= num_points:
                break
            if tile not in selected:
                selected.append(tile)
        
        # Convertir a ReferencePoint
        self.reference_points = []
        for tile in selected[:num_points]:
            ref = ReferencePoint(tile['x'], tile['y'], tile['w'], tile['h'])
            self.reference_points.append(ref)
        
        return len(self.reference_points) >= num_points
    
    def assign_grid_positions(self):
        """Asigna posiciones de grilla a los puntos de referencia automáticamente"""
        if len(self.reference_points) < 3:
            print("Error: Se necesitan al menos 3 puntos de referencia")
            return
        
        # Calcular tamaño promedio con mayor precisión
        total_w = sum(p.width for p in self.reference_points)
        total_h = sum(p.height for p in self.reference_points)
        avg_w = total_w / len(self.reference_points)
        avg_h = total_h / len(self.reference_points)
        self.avg_tile_size = (avg_w, avg_h)
        
        # Encontrar el punto más arriba a la izquierda como origen
        origin_idx = 0
        min_score = float('inf')
        
        for i, point in enumerate(self.reference_points):
            # Score basado en distancia desde esquina superior izquierda
            score = point.x + point.y
            if score < min_score:
                min_score = score
                origin_idx = i
        
        origin = self.reference_points[origin_idx]
        origin.grid_row = 0
        origin.grid_col = 0
        
        # Para cada otro punto, calcular su posición en grilla con mayor precisión
        avg_w, avg_h = self.avg_tile_size
        
        for i in range(len(self.reference_points)):
            if i == origin_idx:
                continue
                
            point = self.reference_points[i]
            
            # Calcular desplazamiento desde el origen en píxeles (centro a centro)
            dx = (point.x + point.width/2) - (origin.x + origin.width/2)
            dy = (point.y + point.height/2) - (origin.y + origin.height/2)
            
            # Convertir a posiciones de grilla (redondeando al entero más cercano)
            grid_col = round(dx / avg_w)
            grid_row = round(dy / avg_h)
            
            point.grid_row = grid_row
            point.grid_col = grid_col
    
    def interpolate_tile_position(self, grid_row: int, grid_col: int) -> Tuple[int, int, int, int]:
        """Interpola la posición y tamaño de una loseta usando los puntos de referencia con mayor precisión"""
        if len(self.reference_points) < 3:
            return None
        
        # Encontrar los 4 puntos de referencia más cercanos para interpolación bilineal
        distances = []
        for ref in self.reference_points:
            dr = grid_row - ref.grid_row
            dc = grid_col - ref.grid_col
            distance = np.sqrt(dr**2 + dc**2)
            distances.append((distance, ref))
        
        # Ordenar por distancia y tomar los 4 más cercanos (o todos si hay menos)
        distances.sort(key=lambda x: x[0])
        closest_refs = [ref for dist, ref in distances[:min(4, len(distances))]]
        
        if len(closest_refs) == 0:
            return None
        
        # Calcular pesos usando inverse distance weighting con potencia mayor para mejor precisión
        weights = []
        positions = []
        
        for ref in closest_refs:
            dr = grid_row - ref.grid_row
            dc = grid_col - ref.grid_col
            distance = np.sqrt(dr**2 + dc**2) + 0.01  # Pequeño epsilon
            
            # Peso inverso al cuadrado de la distancia (más peso a puntos cercanos)
            weight = 1.0 / (distance ** 2)
            weights.append(weight)
            
            # Calcular posición estimada desde este punto de referencia (centro a centro)
            est_center_x = (ref.x + ref.width/2) + dc * ref.width
            est_center_y = (ref.y + ref.height/2) + dr * ref.height
            
            # Convertir de centro a esquina superior izquierda
            est_x = est_center_x - ref.width/2
            est_y = est_center_y - ref.height/2
            
            positions.append((est_x, est_y, ref.width, ref.height))
        
        # Normalizar pesos
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Interpolar posición y tamaño
        final_x = sum(w * pos[0] for w, pos in zip(weights, positions))
        final_y = sum(w * pos[1] for w, pos in zip(weights, positions))
        final_w = sum(w * pos[2] for w, pos in zip(weights, positions))
        final_h = sum(w * pos[3] for w, pos in zip(weights, positions))
        
        return (int(final_x), int(final_y), int(final_w), int(final_h))
    
    def detect_tiles_interpolated(self) -> List[Tile]:
        """Detecta todas las losetas usando interpolación de puntos de referencia con mejor precisión"""
        if len(self.reference_points) < 3:
            print("Error: Se necesitan al menos 3 puntos de referencia")
            return []
        
        h, w = self.image.shape[:2]
        tiles = []
        
        # Determinar rango de grilla a explorar basado en referencias
        min_row = min(ref.grid_row for ref in self.reference_points) - 12
        max_row = max(ref.grid_row for ref in self.reference_points) + 12
        min_col = min(ref.grid_col for ref in self.reference_points) - 12
        max_col = max(ref.grid_col for ref in self.reference_points) + 12
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                # Interpolar posición
                result = self.interpolate_tile_position(row, col)
                if result is None:
                    continue
                
                x, y, tw, th = result
                
                # Verificar límites con margen de seguridad
                if x < -5 or y < -5 or x + tw > w + 5 or y + th > h + 5:
                    continue
                
                # Ajustar a límites seguros
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                tw = min(tw, w - x)
                th = min(th, h - y)
                
                if tw < 10 or th < 10:
                    continue
                
                # Extraer región
                tile_img = self.image[int(y):int(y+th), int(x):int(x+tw)]
                
                if tile_img.size == 0:
                    continue
                
                # Verificar si contiene contenido válido con criterios más estrictos
                gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
                
                # Usar múltiples métricas para validar la loseta
                std_val = np.std(gray)
                mean_val = np.mean(gray)
                
                # Detectar bordes para confirmar que hay contenido
                edges = cv2.Canny(gray, 30, 100)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Filtrar fondos uniformes con criterios más estrictos
                # - Debe tener variación (std > 15)
                # - No debe ser muy blanco (mean < 240)
                # - Debe tener algunos bordes (edge_density > 0.01)
                if std_val > 15 and mean_val < 240 and edge_density > 0.01:
                    tile = Tile(int(x), int(y), int(tw), int(th), tile_img, row, col)
                    tiles.append(tile)

        self.tiles = tiles
        return tiles
    
    def create_grid_overlay(self) -> np.ndarray:
        """Crea visualización con grilla y losetas numeradas"""
        result = self.image.copy()
        
        if not self.tiles:
            return result
        
        # Dibujar puntos de referencia
        colors = [
            (255, 0, 0),    # Rojo
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Amarillo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Naranja
            (128, 0, 255)   # Violeta
        ]
        for i, ref in enumerate(self.reference_points):
            color = colors[i % 8]
            cv2.rectangle(result, (ref.x, ref.y), 
                         (ref.x + ref.width, ref.y + ref.height), color, 4)
            cv2.putText(result, f"REF{i+1} ({ref.grid_row},{ref.grid_col})", 
                       (ref.x, ref.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Dibujar losetas detectadas
        for i, tile in enumerate(self.tiles):
            x, y, tw, th = tile.bbox
            
            # Borde verde para losetas detectadas
            cv2.rectangle(result, (x, y), (x + tw, y + th), (0, 255, 0), 2)
            
            # Número en el centro
            cx, cy = tile.center
            cv2.circle(result, (cx, cy), 12, (0, 0, 255), -1)
            cv2.putText(result, str(i), (cx - 7, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def save_individual_tiles(self, output_dir: str = "tiles"):
        """Guarda cada loseta como imagen individual"""
        import os
        import shutil
        
        # Vaciar la carpeta tiles/ si existe
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Crear carpeta nueva
        os.makedirs(output_dir)
        
        for i, tile in enumerate(self.tiles):
            filename = os.path.join(output_dir, f"tile_{i:03d}_r{tile.grid_row}_c{tile.grid_col}.png")
            cv2.imwrite(filename, tile.image)


def main():
    import sys
    
    if len(sys.argv) < 2:
        return
    
    image_path = sys.argv[1]
    
    detector = CarcassonneTileDetector()
    
    if not detector.load_image(image_path):
        return
    
    # Seleccionar 8 losetas de referencia
    if not detector.select_reference_tiles(num_points=8):
        print("Selección cancelada")
        return
    
    # Asignar posiciones de grilla
    detector.assign_grid_positions()
    
    # Detectar todas las losetas
    tiles = detector.detect_tiles_interpolated()
    
    if not tiles:
        print("No se detectaron losetas.")
        return
    
    result = detector.create_grid_overlay()
    
    cv2.namedWindow("Losetas Detectadas", cv2.WINDOW_NORMAL)
    
    h, w = result.shape[:2]
    max_width = 1400
    max_height = 900
    scale = min(max_width / w, max_height / h, 1.0)
    cv2.resizeWindow("Losetas Detectadas", int(w * scale), int(h * scale))
    
    cv2.imshow("Losetas Detectadas", result)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            detector.save_individual_tiles()
        elif key == ord('r'):
            cv2.imwrite("resultado_deteccion.png", result)
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
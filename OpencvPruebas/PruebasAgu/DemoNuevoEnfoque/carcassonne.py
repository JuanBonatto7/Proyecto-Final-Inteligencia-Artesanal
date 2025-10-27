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
                    print(f"Punto {len(self.reference_points)}: ({ref.x}, {ref.y}) - {ref.width}x{ref.height}")
    
    def select_reference_tiles(self, num_points: int = 8) -> bool:
        """Permite al usuario seleccionar múltiples losetas de referencia"""
        print("\n=== SELECCIÓN DE LOSETAS DE REFERENCIA ===")
        print(f"Instrucciones:")
        print(f"1. Selecciona {num_points} losetas COMPLETAS bien distribuidas:")
        print("   - Cuatro en las ESQUINAS (superior-izq, superior-der, inferior-izq, inferior-der)")
        print("   - Cuatro en los BORDES (centro-superior, centro-inferior, centro-izquierdo, centro-derecho)")
        print("   O simplemente distribuye las 8 losetas uniformemente por todo el tablero")
        print("2. Presiona ENTER cuando termines")
        print("3. Presiona 'u' para deshacer última selección")
        print("4. Presiona ESC para cancelar")
        print("=" * 50)
        
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
                else:
                    print(f"Necesitas seleccionar al menos {num_points} losetas")
            elif key == ord('u'):  # Deshacer
                if self.reference_points:
                    self.reference_points.pop()
                    print(f"Punto eliminado. Total: {len(self.reference_points)}")
            elif key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return False
        
        return False
    
    def assign_grid_positions(self):
        """Asigna posiciones de grilla a los puntos de referencia automáticamente"""
        if len(self.reference_points) < 3:
            print("Error: Se necesitan al menos 3 puntos de referencia")
            return
        
        # Calcular tamaño promedio
        total_w = sum(p.width for p in self.reference_points)
        total_h = sum(p.height for p in self.reference_points)
        self.avg_tile_size = (total_w // len(self.reference_points), total_h // len(self.reference_points))
        
        print(f"\nTamaño promedio de loseta: {self.avg_tile_size[0]}x{self.avg_tile_size[1]}")
        
        # Calcular posiciones de grilla basadas en distancias relativas
        print("\n=== CALCULANDO POSICIONES DE GRILLA AUTOMÁTICAMENTE ===")
        
        # Usar el primer punto como origen (0, 0)
        origin = self.reference_points[0]
        origin.grid_row = 0
        origin.grid_col = 0
        print(f"Loseta #1 (origen): posición ({origin.x}, {origin.y}) → grilla (0, 0)")
        
        # Para cada otro punto, calcular su posición en grilla
        avg_w, avg_h = self.avg_tile_size
        
        for i in range(1, len(self.reference_points)):
            point = self.reference_points[i]
            
            # Calcular desplazamiento desde el origen en píxeles
            dx = point.x - origin.x
            dy = point.y - origin.y
            
            # Convertir a posiciones de grilla (redondeando al entero más cercano)
            grid_col = round(dx / avg_w)
            grid_row = round(dy / avg_h)
            
            point.grid_row = grid_row
            point.grid_col = grid_col
            
            print(f"Loseta #{i+1}: posición ({point.x}, {point.y}) → grilla ({grid_row}, {grid_col})")
    
    def interpolate_tile_position(self, grid_row: int, grid_col: int) -> Tuple[int, int, int, int]:
        """Interpola la posición y tamaño de una loseta usando los puntos de referencia"""
        if len(self.reference_points) < 3:
            return None
        
        # Usar interpolación bilineal basada en los puntos de referencia más cercanos
        weights = []
        positions = []
        
        for ref in self.reference_points:
            # Calcular distancia en grilla
            dr = grid_row - ref.grid_row
            dc = grid_col - ref.grid_col
            distance = np.sqrt(dr**2 + dc**2) + 0.1  # +0.1 para evitar división por cero
            
            weight = 1.0 / distance**2  # Peso inverso al cuadrado de la distancia
            weights.append(weight)
            
            # Calcular posición estimada desde este punto de referencia
            est_x = ref.x + dc * ref.width
            est_y = ref.y + dr * ref.height
            
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
        """Detecta todas las losetas usando interpolación de puntos de referencia"""
        if len(self.reference_points) < 3:
            print("Error: Se necesitan al menos 3 puntos de referencia")
            return []
        
        print("\n=== DETECTANDO LOSETAS CON INTERPOLACIÓN ===")
        
        h, w = self.image.shape[:2]
        tiles = []
        
        # Determinar rango de grilla a explorar
        min_row = min(ref.grid_row for ref in self.reference_points) - 10
        max_row = max(ref.grid_row for ref in self.reference_points) + 10
        min_col = min(ref.grid_col for ref in self.reference_points) - 10
        max_col = max(ref.grid_col for ref in self.reference_points) + 10
        
        print(f"Explorando grilla: filas [{min_row}, {max_row}], columnas [{min_col}, {max_col}]")
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                # Interpolar posición
                result = self.interpolate_tile_position(row, col)
                if result is None:
                    continue
                
                x, y, tw, th = result
                
                # Verificar límites
                if x < 0 or y < 0 or x + tw > w or y + th > h:
                    continue
                
                # Extraer región
                tile_img = self.image[y:y+th, x:x+tw]
                
                # Verificar si contiene contenido válido
                gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
                std_val = np.std(gray)
                mean_val = np.mean(gray)
                
                # Filtrar fondos uniformes
                if std_val > 20 and mean_val < 210:
                    tile = Tile(x, y, tw, th, tile_img, row, col)
                    tiles.append(tile)
        
        print(f"Detectadas {len(tiles)} losetas válidas")
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
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n=== GUARDANDO LOSETAS ===")
        for i, tile in enumerate(self.tiles):
            filename = os.path.join(output_dir, f"tile_{i:03d}_r{tile.grid_row}_c{tile.grid_col}.png")
            cv2.imwrite(filename, tile.image)
        
        print(f"Guardadas {len(self.tiles)} losetas en '{output_dir}/'")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python script.py <ruta_imagen>")
        print("Ejemplo: python script.py carcassonne.jpg")
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
    
    print("\n=== VISUALIZACIÓN ===")
    print("Presiona:")
    print("  's' - Guardar losetas individuales")
    print("  'r' - Guardar imagen resultado")
    print("  'q' - Salir")
    
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
            print("Imagen guardada como 'resultado_deteccion.png'")
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
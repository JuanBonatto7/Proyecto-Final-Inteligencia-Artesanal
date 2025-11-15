#!/usr/bin/env python3
"""
Pipeline completo: Detecta losetas Y meeples en cada una
"""

import cv2
import os
import json
from pathlib import Path
from carcassonne import CarcassonneTileDetector
from MeepleDetectorSimple import MeepleDetector


class CarcassonneMeeplePipeline:
    """Pipeline que combina detección de losetas y meeples"""
    
    def __init__(self):
        self.tile_detector = CarcassonneTileDetector()
        self.meeple_detector = MeepleDetector()
        self.results = []
    
    def process_board(self, image_path: str, num_reference_points: int = 8) -> bool:
        """
        Procesa todo el tablero: detecta losetas y meeples
        
        Args:
            image_path: Ruta a la imagen del tablero
            num_reference_points: Número de puntos de referencia para calibración
            
        Returns:
            True si el procesamiento fue exitoso
        """
        print("\n" + "=" * 70)
        print("PIPELINE: DETECCIÓN DE LOSETAS Y MEEPLES")
        print("=" * 70)
        
        # 1. Cargar imagen
        print("\n[1/4] Cargando imagen...")
        if not self.tile_detector.load_image(image_path):
            return False
        print("✓ Imagen cargada")
        
        # 2. Seleccionar puntos de referencia
        print(f"\n[2/4] Seleccionando {num_reference_points} losetas de referencia...")
        if not self.tile_detector.select_reference_tiles(num_points=num_reference_points):
            print("✗ Selección cancelada")
            return False
        print(f"✓ {len(self.tile_detector.reference_points)} puntos de referencia seleccionados")
        
        # 3. Detectar todas las losetas
        print("\n[3/4] Detectando todas las losetas...")
        self.tile_detector.assign_grid_positions()
        tiles = self.tile_detector.detect_tiles_interpolated()
        
        if not tiles:
            print("✗ No se detectaron losetas")
            return False
        print(f"✓ {len(tiles)} losetas detectadas")
        
        # 4. Analizar meeples en cada loseta
        print("\n[4/4] Analizando meeples en cada loseta...")
        self.results = []
        
        stats = {
            'total_tiles': len(tiles),
            'tiles_with_meeple': 0,
            'blue_meeples': 0,
            'black_meeples': 0,
            'unknown_meeples': 0
        }
        
        for i, tile in enumerate(tiles):
            print(f"\r  Procesando {i+1}/{len(tiles)}...", end='', flush=True)
            
            # Detectar meeple en esta loseta
            # Guardar temporalmente la imagen de la loseta
            temp_path = f"temp_tile_{i}.png"
            cv2.imwrite(temp_path, tile.image)
            
            meeple_result = self.meeple_detector.detect_meeple(temp_path)
            
            # Limpiar archivo temporal
            os.remove(temp_path)
            
            # Guardar resultado combinado
            result = {
                'tile_index': i,
                'grid_position': (tile.grid_row, tile.grid_col),
                'bbox': tile.bbox,
                'has_meeple': meeple_result['has_meeple'],
                'meeple_color': meeple_result['color'],
                'meeple_position': meeple_result['position'],
                'confidence': meeple_result['confidence']
            }
            
            self.results.append(result)
            
            # Actualizar estadísticas
            if meeple_result['has_meeple']:
                stats['tiles_with_meeple'] += 1
                if meeple_result['color'] == 'blue':
                    stats['blue_meeples'] += 1
                elif meeple_result['color'] == 'black':
                    stats['black_meeples'] += 1
                else:
                    stats['unknown_meeples'] += 1
        
        print()  # Nueva línea después del progreso
        
        # Mostrar estadísticas
        print("\n" + "=" * 70)
        print("ESTADÍSTICAS")
        print("=" * 70)
        print(f"Total de losetas: {stats['total_tiles']}")
        print(f"Losetas con meeple: {stats['tiles_with_meeple']} ({stats['tiles_with_meeple']/stats['total_tiles']*100:.1f}%)")
        print(f"  🔵 Meeples azules: {stats['blue_meeples']}")
        print(f"  ⚫ Meeples negros: {stats['black_meeples']}")
        print(f"  ❓ Meeples desconocidos: {stats['unknown_meeples']}")
        
        return True
    
    def create_visualization(self, output_path: str = "resultado_completo.png"):
        """
        Crea visualización con losetas y meeples detectados
        """
        result = self.tile_detector.image.copy()
        
        # Dibujar cada loseta con información de meeple
        for data in self.results:
            x, y, w, h = data['bbox']
            
            # Color del borde según si tiene meeple
            if data['has_meeple']:
                # Color según el color del meeple
                if data['meeple_color'] == 'blue':
                    border_color = (255, 0, 0)  # Azul
                    label = "BLUE"
                elif data['meeple_color'] == 'black':
                    border_color = (50, 50, 50)  # Gris oscuro
                    label = "BLACK"
                else:
                    border_color = (0, 255, 255)  # Amarillo
                    label = "???"
                
                # Borde grueso para losetas con meeple
                cv2.rectangle(result, (x, y), (x + w, y + h), border_color, 4)
                
                # Etiqueta
                cv2.putText(result, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
            else:
                # Borde delgado verde para losetas sin meeple
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        cv2.imwrite(output_path, result)
        print(f"\n✓ Visualización guardada en: {output_path}")
        return result
    
    def save_tiles_with_info(self, output_dir: str = "tiles_analyzed"):
        """
        Guarda cada loseta con información de meeple en el nombre
        """
        import shutil
        
        # Limpiar/crear directorio
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        print(f"\n=== GUARDANDO LOSETAS CON INFORMACIÓN ===")
        
        for i, data in enumerate(self.results):
            tile = self.tile_detector.tiles[data['tile_index']]
            row, col = data['grid_position']
            
            # Nombre descriptivo
            if data['has_meeple']:
                color = data['meeple_color'] or 'unknown'
                pos = data['meeple_position']
                filename = f"tile_{i:03d}_r{row}_c{col}_{color}_pos{pos}.png"
            else:
                filename = f"tile_{i:03d}_r{row}_c{col}_empty.png"
            
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, tile.image)
        
        print(f"✓ {len(self.results)} losetas guardadas en '{output_dir}/'")
    
    def save_results_json(self, output_path: str = "deteccion_resultados.json"):
        """Guarda resultados en formato JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ Resultados guardados en: {output_path}")
    
    def show_results(self):
        """Muestra resultados interactivamente"""
        result = self.create_visualization("temp_visualization.png")
        
        print("\n=== VISUALIZACIÓN INTERACTIVA ===")
        print("Presiona:")
        print("  's' - Guardar losetas individuales con información")
        print("  'j' - Guardar resultados en JSON")
        print("  'r' - Guardar visualización")
        print("  'q' - Salir")
        
        cv2.namedWindow("Resultados", cv2.WINDOW_NORMAL)
        
        h, w = result.shape[:2]
        max_width = 1400
        max_height = 900
        scale = min(max_width / w, max_height / h, 1.0)
        cv2.resizeWindow("Resultados", int(w * scale), int(h * scale))
        
        cv2.imshow("Resultados", result)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):
                self.save_tiles_with_info()
            elif key == ord('j'):
                self.save_results_json()
            elif key == ord('r'):
                self.create_visualization("resultado_completo.png")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Limpiar archivo temporal
        if os.path.exists("temp_visualization.png"):
            os.remove("temp_visualization.png")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python carcassonne_meeple_pipeline.py <imagen_tablero>")
        print("Ejemplo: python carcassonne_meeple_pipeline.py tablero.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen: {image_path}")
        return
    
    # Crear pipeline
    pipeline = CarcassonneMeeplePipeline()
    
    # Procesar tablero
    if not pipeline.process_board(image_path, num_reference_points=8):
        print("\nProcesamiento fallido")
        return
    
    # Mostrar resultados
    pipeline.show_results()
    
    print("\n✓ Pipeline completado")


if __name__ == "__main__":
    main()
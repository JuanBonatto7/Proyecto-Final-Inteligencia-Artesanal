#!/usr/bin/env python3
"""
Pipeline COMPLETO: Detecta losetas, tipo, rotación Y meeples
"""

import cv2
import os
import json
import sys
from pathlib import Path
from carcassonne import CarcassonneTileDetector
from MeepleDetectorSimple import MeepleDetector
from tile_detector import CarcassonneTileDetector as TileTypeDetector
from rotation_detector import CarcassonneRotationDetector


class CarcassonneCompletePipeline:
    """Pipeline completo: losetas + tipo + rotación + meeples"""
    
    def __init__(self):
        self.tile_detector = CarcassonneTileDetector()  # Recorte de losetas
        self.meeple_detector = MeepleDetector()  # Detección de meeples
        self.type_detector = TileTypeDetector()  # Detección de tipo
        self.rotation_detector = CarcassonneRotationDetector()  # Detección de rotación
        self.results = []
    
    def process_board(self, image_path: str, num_reference_points: int = 8) -> bool:
        """
        Procesa todo el tablero: detecta losetas, tipo, rotación y meeples
        
        Args:
            image_path: Ruta a la imagen del tablero
            num_reference_points: Número de puntos de referencia para calibración
            
        Returns:
            True si el procesamiento fue exitoso
        """
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETO: LOSETAS + TIPO + ROTACIÓN + MEEPLES")
        print("=" * 80)
        
        # 1. Cargar imagen
        print("\n[1/5] Cargando imagen...")
        if not self.tile_detector.load_image(image_path):
            return False
        print("✓ Imagen cargada")
        
        # 2. Seleccionar puntos de referencia
        print(f"\n[2/5] Seleccionando {num_reference_points} losetas de referencia...")
        if not self.tile_detector.select_reference_tiles(num_points=num_reference_points):
            print("✗ Selección cancelada")
            return False
        print(f"✓ {len(self.tile_detector.reference_points)} puntos de referencia seleccionados")
        
        # 3. Detectar todas las losetas (recortar)
        print("\n[3/5] Detectando y recortando todas las losetas...")
        self.tile_detector.assign_grid_positions()
        tiles = self.tile_detector.detect_tiles_interpolated()
        
        if not tiles:
            print("✗ No se detectaron losetas")
            return False
        print(f"✓ {len(tiles)} losetas detectadas y recortadas")
        
        # 4. Analizar cada loseta: tipo, rotación y meeples
        print(f"\n[4/5] Analizando cada loseta (tipo + rotación + meeples)...")
        self.results = []
        
        stats = {
            'total_tiles': len(tiles),
            'tiles_with_meeple': 0,
            'blue_meeples': 0,
            'black_meeples': 0,
            'unknown_meeples': 0,
            'tile_types': {},
            'rotations': {0: 0, 90: 0, 180: 0, 270: 0}
        }
        
        for i, tile in enumerate(tiles):
            print(f"\r  Procesando loseta {i+1}/{len(tiles)}...", end='', flush=True)
            
            # Guardar temporalmente la imagen de la loseta
            temp_path = f"temp_tile_{i}.png"
            cv2.imwrite(temp_path, tile.image)
            
            # A. Detectar TIPO de loseta
            try:
                tile_type = self.type_detector.detect_tile(temp_path)
            except Exception as e:
                print(f"\n  ⚠ Error detectando tipo en loseta {i}: {e}")
                tile_type = "?"
            
            # B. Detectar ROTACIÓN
            try:
                rotation = self.rotation_detector.detect_rotation(tile_type, temp_path)
            except Exception as e:
                print(f"\n  ⚠ Error detectando rotación en loseta {i}: {e}")
                rotation = 0
            
            # C. Detectar MEEPLE
            try:
                meeple_result = self.meeple_detector.detect_meeple(temp_path)
            except Exception as e:
                print(f"\n  ⚠ Error detectando meeple en loseta {i}: {e}")
                meeple_result = {
                    'has_meeple': False,
                    'color': None,
                    'position': None,
                    'confidence': 0.0
                }
            
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Guardar resultado completo
            result = {
                'tile_index': i,
                'grid_position': (tile.grid_row, tile.grid_col),
                'bbox': tile.bbox,
                'tile_type': tile_type,
                'rotation': rotation,
                'has_meeple': meeple_result['has_meeple'],
                'meeple_color': meeple_result['color'],
                'meeple_position': meeple_result['position'],
                'meeple_confidence': meeple_result['confidence']
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
            
            # Estadísticas de tipos
            if tile_type not in stats['tile_types']:
                stats['tile_types'][tile_type] = 0
            stats['tile_types'][tile_type] += 1
            
            # Estadísticas de rotaciones
            if rotation in stats['rotations']:
                stats['rotations'][rotation] += 1
        
        print()  # Nueva línea después del progreso
        
        # 5. Guardar resultados automáticamente
        print("\n[5/5] Guardando resultados...")
        self.save_tiles_with_complete_info()
        self.save_results_json()
        self.create_visualization("resultado_completo.png")
        print("✓ Todos los resultados guardados")
        
        # Mostrar estadísticas
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS COMPLETAS")
        print("=" * 80)
        print(f"\n📊 LOSETAS:")
        print(f"  Total de losetas: {stats['total_tiles']}")
        
        print(f"\n🎲 TIPOS DE LOSETAS:")
        for tile_type, count in sorted(stats['tile_types'].items()):
            print(f"  Tipo {tile_type}: {count}")
        
        print(f"\n🔄 ROTACIONES:")
        for rotation, count in sorted(stats['rotations'].items()):
            print(f"  {rotation}°: {count}")
        
        print(f"\n👤 MEEPLES:")
        print(f"  Losetas con meeple: {stats['tiles_with_meeple']} ({stats['tiles_with_meeple']/stats['total_tiles']*100:.1f}%)")
        print(f"  🔵 Meeples azules: {stats['blue_meeples']}")
        print(f"  ⚫ Meeples negros: {stats['black_meeples']}")
        print(f"  ❓ Meeples desconocidos: {stats['unknown_meeples']}")
        
        return True
    
    def create_visualization(self, output_path: str = "resultado_completo.png"):
        """
        Crea visualización con TODA la información
        """
        result = self.tile_detector.image.copy()
        
        # Dibujar cada loseta con información completa
        for data in self.results:
            x, y, w, h = data['bbox']
            
            # Color del borde según si tiene meeple
            if data['has_meeple']:
                if data['meeple_color'] == 'blue':
                    border_color = (255, 0, 0)  # Azul
                elif data['meeple_color'] == 'black':
                    border_color = (50, 50, 50)  # Gris oscuro
                else:
                    border_color = (0, 255, 255)  # Amarillo
                thickness = 4
            else:
                border_color = (0, 255, 0)  # Verde
                thickness = 2
            
            # Dibujar borde
            cv2.rectangle(result, (x, y), (x + w, y + h), border_color, thickness)
            
            # Crear etiqueta compacta con toda la info
            tile_type = data['tile_type']
            rotation = data['rotation']
            
            if data['has_meeple']:
                meeple_info = f"{data['meeple_color'][0].upper()}{data['meeple_position']}"
                label = f"{tile_type}-{rotation}° [{meeple_info}]"
            else:
                label = f"{tile_type}-{rotation}°"
            
            # Dibujar etiqueta con fondo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Fondo semi-transparente
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y - text_height - 10), 
                         (x + text_width + 5, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
            
            # Texto
            cv2.putText(result, label, (x + 2, y - 5),
                       font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imwrite(output_path, result)
        print(f"✓ Visualización guardada en: {output_path}")
        return result
    
    def save_tiles_with_complete_info(self, output_dir: str = "tiles_complete"):
        """
        Guarda cada loseta con TODA la información en el nombre
        """
        import shutil
        
        # Limpiar/crear directorio
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        print(f"\n=== GUARDANDO LOSETAS CON INFORMACIÓN COMPLETA ===")
        
        for i, data in enumerate(self.results):
            tile = self.tile_detector.tiles[data['tile_index']]
            row, col = data['grid_position']
            tile_type = data['tile_type']
            rotation = data['rotation']
            
            # Nombre super descriptivo
            if data['has_meeple']:
                color = data['meeple_color'] or 'unknown'
                pos = data['meeple_position']
                filename = f"tile_{i:03d}_r{row}_c{col}_{tile_type}_rot{rotation}_{color}_pos{pos}.png"
            else:
                filename = f"tile_{i:03d}_r{row}_c{col}_{tile_type}_rot{rotation}_empty.png"
            
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, tile.image)
        
        print(f"✓ {len(self.results)} losetas guardadas en '{output_dir}/'")
    
    def save_results_json(self, output_path: str = "deteccion_completa.json"):
        """Guarda resultados completos en formato JSON"""
        output_data = {
            'total_tiles': len(self.results),
            'tiles': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Resultados JSON guardados en: {output_path}")
    
    def show_interactive_results(self):
        """Muestra resultados interactivamente (opcional)"""
        result = cv2.imread("resultado_completo.png")
        
        if result is None:
            print("⚠ No se pudo cargar la visualización")
            return
        
        print("\n=== VISUALIZACIÓN INTERACTIVA ===")
        print("Presiona 'q' para salir")
        
        cv2.namedWindow("Resultados Completos", cv2.WINDOW_NORMAL)
        
        h, w = result.shape[:2]
        max_width = 1400
        max_height = 900
        scale = min(max_width / w, max_height / h, 1.0)
        cv2.resizeWindow("Resultados Completos", int(w * scale), int(h * scale))
        
        cv2.imshow("Resultados Completos", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Uso: python pipeline_completo.py <imagen_tablero>")
        print("Ejemplo: python pipeline_completo.py tablero.jpg")
        print("\nEste pipeline detecta:")
        print("  1. Recorta todas las losetas del tablero")
        print("  2. Detecta el TIPO de cada loseta (A-X)")
        print("  3. Detecta la ROTACIÓN (0°, 90°, 180°, 270°)")
        print("  4. Detecta MEEPLES (color y posición)")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen: {image_path}")
        return
    
    # Verificar que existen los archivos necesarios
    required_files = [
        'carcassonne.py',
        'MeepleDetectorSimple.py',
        'tile_detector.py',
        'rotation_detector.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("⚠ ADVERTENCIA: Faltan archivos necesarios:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nEl pipeline puede fallar. Asegúrate de tener todos los archivos.")
        response = input("\n¿Continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            return
    
    # Crear pipeline
    print("\n🚀 Iniciando pipeline completo...")
    pipeline = CarcassonneCompletePipeline()
    
    # Procesar tablero
    if not pipeline.process_board(image_path, num_reference_points=8):
        print("\n❌ Procesamiento fallido")
        return
    
    # Preguntar si quiere ver visualización
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  📁 tiles_complete/           - Losetas con información completa")
    print("  📄 deteccion_completa.json   - Todos los datos en JSON")
    print("  🖼️  resultado_completo.png    - Visualización del tablero")
    
    show_viz = input("\n¿Deseas ver la visualización? (s/n): ")
    if show_viz.lower() == 's':
        pipeline.show_interactive_results()
    
    print("\n✨ ¡Todo listo!")


if __name__ == "__main__":
    main()
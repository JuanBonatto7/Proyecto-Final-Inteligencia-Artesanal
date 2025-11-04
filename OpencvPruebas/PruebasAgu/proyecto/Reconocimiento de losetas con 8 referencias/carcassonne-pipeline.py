"""
Pipeline completo para detección y clasificación de losetas de Carcassonne
Integra el detector de losetas con la CNN de clasificación
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple

# Importar módulos propios
from carcassonne import CarcassonneTileDetector
from carcassonne_cnn import CarcassonnePredictor


class CarcassonnePipeline:
    """Pipeline completo: detección + clasificación de losetas"""
    
    def __init__(self, model_path: str):
        self.detector = CarcassonneTileDetector()
        self.predictor = CarcassonnePredictor(model_path)
        self.results = []
    
    def process_board_image(self, image_path: str, num_reference_points: int = 8) -> List[Dict]:
        """
        Procesa una imagen completa del tablero
        
        Args:
            image_path: Ruta de la imagen del tablero
            num_reference_points: Número de puntos de referencia a seleccionar
        
        Returns:
            Lista de diccionarios con información de cada loseta
        """
        print("\n" + "="*60)
        print("PIPELINE DE RECONOCIMIENTO DE CARCASSONNE")
        print("="*60)
        
        # Paso 1: Cargar imagen
        print("\n[1/4] Cargando imagen...")
        if not self.detector.load_image(image_path):
            print("Error al cargar la imagen")
            return []
        
        # Paso 2: Seleccionar puntos de referencia
        print("\n[2/4] Selección de puntos de referencia...")
        if not self.detector.select_reference_tiles(num_points=num_reference_points):
            print("Selección cancelada")
            return []
        
        # Paso 3: Detectar todas las losetas
        print("\n[3/4] Detectando losetas...")
        self.detector.assign_grid_positions()
        tiles = self.detector.detect_tiles_interpolated()
        
        if not tiles:
            print("No se detectaron losetas")
            return []
        
        print(f"✓ {len(tiles)} losetas detectadas")
        
        # Paso 4: Clasificar cada loseta con CNN
        print("\n[4/4] Clasificando losetas con CNN...")
        self.results = []
        
        for i, tile in enumerate(tiles):
            print(f"  Procesando loseta {i+1}/{len(tiles)}...", end='\r')
            
            # Guardar loseta temporalmente
            temp_path = f'/tmp/temp_tile_{i}.png'
            cv2.imwrite(temp_path, tile.image)
            
            # Predecir con CNN
            prediction = self.predictor.predict(temp_path)
            
            # Combinar información
            result = {
                'tile_id': i,
                'grid_position': (tile.grid_row, tile.grid_col),
                'pixel_position': (tile.x, tile.y),
                'size': (tile.width, tile.height),
                'tile_type': prediction['tile_type'],
                'tile_type_confidence': prediction['tile_type_confidence'],
                'rotation_degrees': prediction['rotation'],
                'rotation_confidence': prediction['rotation_confidence'],
                'has_meeple': prediction['has_meeple'],
                'meeple_confidence': prediction['meeple_confidence'],
                'meeple_position': prediction['meeple_position']
            }
            
            self.results.append(result)
        
        print(f"\n✓ Clasificación completada")
        
        return self.results
            # Mover a Reconocimiento de losetas con 8 referencias
    
    def visualize_results(self, output_path: str = 'board_analysis.png'):
        """Crea visualización completa del tablero analizado"""
        if not self.results:
            print("No hay resultados para visualizar")
            return
        
        img = self.detector.image.copy()
        
        # Dibujar cada loseta con su información
        for result in self.results:
            x, y = result['pixel_position']
            w, h = result['size']
            
            # Color según confianza
            conf = result['tile_type_confidence']
            if conf > 0.9:
                color = (0, 255, 0)  # Verde: alta confianza
            elif conf > 0.7:
                color = (255, 255, 0)  # Amarillo: confianza media
            else:
                color = (0, 0, 255)  # Rojo: baja confianza
            
            # Dibujar borde
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Información de la loseta
            text_lines = [
                f"T:{result['tile_type']}",
                f"R:{result['rotation_degrees']}°",
            ]
            
            if result['has_meeple']:
                text_lines.append(f"M:{result['meeple_position']}")
            
            # Dibujar texto
            y_offset = y - 10
            for line in reversed(text_lines):
                cv2.putText(img, line, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset -= 15
            
            # Círculo en el centro con el ID
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(img, (cx, cy), 8, (255, 0, 0), -1)
            cv2.putText(img, str(result['tile_id']), (cx - 5, cy + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Guardar
        cv2.imwrite(output_path, img)
        print(f"✓ Visualización guardada en {output_path}")
        
        # Mostrar
        h, w = img.shape[:2]
        scale = min(1400 / w, 900 / h, 1.0)
        display = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        cv2.imshow('Análisis del Tablero', display)
        print("\nPresiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def create_detailed_report(self):
        """Crea un reporte detallado con matplotlib"""
        if not self.results:
            print("No hay resultados para el reporte")
            return
        
        # Estadísticas
        total_tiles = len(self.results)
        tiles_with_meeples = sum(1 for r in self.results if r['has_meeple'])
        avg_confidence = np.mean([r['tile_type_confidence'] for r in self.results])
        
        # Distribución de tipos
        tile_types = [r['tile_type'] for r in self.results]
        type_counts = {}
        for t in tile_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Distribución de rotaciones
        rotations = [r['rotation_degrees'] for r in self.results]
        rotation_counts = {0: 0, 90: 0, 180: 0, 270: 0}
        for r in rotations:
            rotation_counts[r] = rotation_counts.get(r, 0) + 1
        
        # Crear figura
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Imagen del tablero
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        img_rgb = cv2.cvtColor(self.detector.image, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        ax1.set_title('Tablero Completo', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Dibujar losetas
        for result in self.results:
            x, y = result['pixel_position']
            w, h = result['size']
            rect = Rectangle((x, y), w, h, linewidth=1, 
                           edgecolor='lime', facecolor='none', alpha=0.7)
            ax1.add_patch(rect)
        
        # 2. Estadísticas generales
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        stats_text = f"""
        ESTADÍSTICAS GENERALES
        {'='*25}
        
        Total de losetas: {total_tiles}
        Con fichas: {tiles_with_meeples}
        Sin fichas: {total_tiles - tiles_with_meeples}
        
        Confianza promedio: {avg_confidence:.1%}
        
        Tipos únicos: {len(type_counts)}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 3. Distribución de tipos
        ax3 = fig.add_subplot(gs[1, 2])
        types = sorted(type_counts.keys())
        counts = [type_counts[t] for t in types]
        ax3.bar(types, counts, color='steelblue', alpha=0.7)
        ax3.set_xlabel('Tipo de Loseta')
        ax3.set_ylabel('Cantidad')
        ax3.set_title('Distribución de Tipos', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Distribución de rotaciones
        ax4 = fig.add_subplot(gs[2, 0])
        rots = list(rotation_counts.keys())
        rot_counts = list(rotation_counts.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        ax4.pie(rot_counts, labels=[f'{r}°' for r in rots], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax4.set_title('Rotaciones', fontweight='bold')
        
        # 5. Confianza por loseta
        ax5 = fig.add_subplot(gs[2, 1])
        tile_ids = [r['tile_id'] for r in self.results]
        confidences = [r['tile_type_confidence'] for r in self.results]
        scatter = ax5.scatter(tile_ids, confidences, c=confidences, 
                            cmap='RdYlGn', s=50, alpha=0.6)
        ax5.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Alta')
        ax5.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Media')
        ax5.set_xlabel('ID Loseta')
        ax5.set_ylabel('Confianza')
        ax5.set_title('Confianza de Clasificación', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax5)
        
        # 6. Tabla de losetas problemáticas
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        low_conf = [r for r in self.results if r['tile_type_confidence'] < 0.7]
        if low_conf:
            table_text = "LOSETAS BAJA CONFIANZA\n" + "="*22 + "\n"
            for r in sorted(low_conf, key=lambda x: x['tile_type_confidence'])[:5]:
                table_text += f"ID {r['tile_id']}: {r['tile_type_confidence']:.1%}\n"
        else:
            table_text = "✓ Todas las losetas con\n  alta confianza"
        
        ax6.text(0.1, 0.5, table_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('Análisis Completo del Tablero de Carcassonne', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('board_analysis_report.png', dpi=150, bbox_inches='tight')
        print("✓ Reporte detallado guardado en board_analysis_report.png")
        plt.show()
    
    def export_results(self, output_file: str = 'board_results.json'):
        """Exporta resultados a JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Resultados exportados a {output_file}")
    
    def export_game_state(self, output_file: str = 'game_state.json'):
        """
        Exporta el estado del juego en formato estructurado
        Útil para integración con software de análisis de partidas
        """
        game_state = {
            'board_size': {
                'rows': max(r['grid_position'][0] for r in self.results) + 1,
                'cols': max(r['grid_position'][1] for r in self.results) + 1
            },
            'total_tiles': len(self.results),
            'tiles': []
        }
        
        for result in self.results:
            tile_state = {
                'id': result['tile_id'],
                'grid_row': result['grid_position'][0],
                'grid_col': result['grid_position'][1],
                'type': result['tile_type'],
                'rotation': result['rotation_degrees'] // 90,
                'meeple': {
                    'present': result['has_meeple'],
                    'position': result['meeple_position'] if result['has_meeple'] else None
                },
                'confidence': {
                    'type': round(result['tile_type_confidence'], 3),
                    'rotation': round(result['rotation_confidence'], 3)
                }
            }
            game_state['tiles'].append(tile_state)
        
        with open(output_file, 'w') as f:
            json.dump(game_state, f, indent=2)
        
        print(f"✓ Estado del juego exportado a {output_file}")
        return game_state


def main():
    """Función principal"""
    if len(sys.argv) < 3:
        print("Uso: python pipeline.py <modelo.pth> <imagen_tablero.jpg>")
        print("\nEjemplo:")
        print("  python pipeline.py best_carcassonne_model.pth tablero.jpg")
        return
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Verificar archivos
    if not Path(model_path).exists():
        print(f"Error: No se encuentra el modelo {model_path}")
        return
    
    if not Path(image_path).exists():
        print(f"Error: No se encuentra la imagen {image_path}")
        return
    
    # Crear pipeline
    pipeline = CarcassonnePipeline(model_path)
    
    # Procesar tablero
    results = pipeline.process_board_image(image_path)
    
    if not results:
        print("No se obtuvieron resultados")
        return
    
    # Generar visualizaciones y reportes
    print("\n" + "="*60)
    print("GENERANDO REPORTES")
    print("="*60)
    
    pipeline.visualize_results('board_analysis.png')
    pipeline.create_detailed_report()
    pipeline.export_results('board_results.json')
    pipeline.export_game_state('game_state.json')
    
    print("\n" + "="*60)
    print("✓ PROCESO COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("  - board_analysis.png: Visualización del tablero")
    print("  - board_analysis_report.png: Reporte detallado")
    print("  - board_results.json: Datos completos")
    print("  - game_state.json: Estado del juego")


if __name__ == "__main__":
    main()

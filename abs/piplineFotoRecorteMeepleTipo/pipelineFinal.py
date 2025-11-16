#!/usr/bin/env python3
"""
Pipeline COMPLETO con ORIGIN MATRIX - REFACTORIZADO
Detecta losetas, tipo, rotación Y meeples → Genera Board con coordenadas correctas
"""

import cv2
import os
import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from carcassonne import CarcassonneTileDetector
from MeepleDetectorSimple import MeepleDetector
from tile_detector import CarcassonneTileDetector as TileTypeDetector
from rotation_detector import CarcassonneRotationDetector
from origin_matrix import Tile, Board
from board_image_generator import BoardImageGenerator


class CarcassonneCompletePipeline:
    """Pipeline completo: losetas + tipo + rotación + meeples → Origin Matrix"""
    
    def __init__(self):
        self.tile_detector = CarcassonneTileDetector()
        self.meeple_detector = MeepleDetector()
        self.type_detector = TileTypeDetector()
        self.rotation_detector = CarcassonneRotationDetector()
        self.results = []
        self.board_matrix: Optional[Board] = None
        self.min_row = 0
        self.min_col = 0
    
    def process_board(self, image_path: str, num_reference_points: int = 8) -> Optional[Board]:
        """
        Procesa todo el tablero y genera la Origin Matrix
        
        Args:
            image_path: Ruta a la imagen del tablero
            num_reference_points: Número de puntos de referencia
            
        Returns:
            Board con la matriz completa o None si falla
        """
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETO: LOSETAS + TIPO + ROTACIÓN + MEEPLES → ORIGIN MATRIX")
        print("=" * 80)
        
        # 1. Cargar imagen
        print("\n[1/6] Cargando imagen...")
        if not self.tile_detector.load_image(image_path):
            return None
        print("✓ Imagen cargada")
        
        # 2. Seleccionar puntos de referencia
        print(f"\n[2/6] Seleccionando {num_reference_points} losetas de referencia...")
        if not self.tile_detector.select_reference_tiles(num_points=num_reference_points):
            print("✗ Selección cancelada")
            return None
        print(f"✓ {len(self.tile_detector.reference_points)} puntos seleccionados")
        
        # 3. Detectar todas las losetas
        print("\n[3/6] Detectando y recortando todas las losetas...")
        self.tile_detector.assign_grid_positions()
        tiles = self.tile_detector.detect_tiles_interpolated()
        
        if not tiles:
            print("✗ No se detectaron losetas")
            return None
        print(f"✓ {len(tiles)} losetas detectadas")
        
        # 4. Calcular dimensiones y offsets
        print("\n[4/6] Calculando dimensiones del tablero...")
        self._calculate_board_dimensions(tiles)
        print(f"✓ Tablero: {self.rows}x{self.cols}")
        print(f"  Offset: min_row={self.min_row}, min_col={self.min_col}")
        
        # 5. Analizar cada loseta
        print(f"\n[5/6] Analizando cada loseta (tipo + rotación + meeples)...")
        self.results = []
        stats = self._process_all_tiles(tiles)
        
        # 6. Crear la Origin Matrix
        print("\n[6/6] Generando Origin Matrix (Board)...")
        self.board_matrix = self._create_board_matrix()
        print(f"✓ Board {self.rows}x{self.cols} generado")
        
        # Guardar resultados
        self._save_all_outputs(stats)
        
        # Mostrar estadísticas
        self._print_statistics(stats)
        
        return self.board_matrix
    
    def _calculate_board_dimensions(self, tiles):
        """
        Calcula las dimensiones del tablero y los offsets necesarios
        """
        # Encontrar límites de la grilla
        self.min_row = min(tile.grid_row for tile in tiles)
        self.max_row = max(tile.grid_row for tile in tiles)
        self.min_col = min(tile.grid_col for tile in tiles)
        self.max_col = max(tile.grid_col for tile in tiles)
        
        # Calcular dimensiones (incluyendo posiciones negativas)
        self.rows = self.max_row - self.min_row + 1
        self.cols = self.max_col - self.min_col + 1
    
    def _process_all_tiles(self, tiles) -> Dict:
        """
        Procesa todas las losetas y recopila estadísticas
        """
        stats = {
            'total_tiles': len(tiles),
            'tiles_with_meeple': 0,
            'blue_meeples': 0,
            'black_meeples': 0,
            'tile_types': {},
            'rotations': {0: 0, 90: 0, 180: 0, 270: 0}
        }
        
        for i, tile in enumerate(tiles):
            print(f"\r  Procesando loseta {i+1}/{len(tiles)}...", end='', flush=True)
            
            # Guardar temporalmente
            temp_path = f"temp_tile_{i}.png"
            cv2.imwrite(temp_path, tile.image)
            
            # Detectar TIPO
            try:
                tile_type = self.type_detector.detect_tile(temp_path)
            except Exception as e:
                print(f"\n  ⚠ Error detectando tipo: {e}")
                tile_type = "?"
            
            # Detectar ROTACIÓN
            try:
                rotation = self.rotation_detector.detect_rotation(tile_type, temp_path)
            except Exception as e:
                print(f"\n  ⚠ Error detectando rotación: {e}")
                rotation = 0
            
            # Detectar MEEPLE
            try:
                meeple_result = self.meeple_detector.detect_meeple(temp_path)
            except Exception as e:
                print(f"\n  ⚠ Error detectando meeple: {e}")
                meeple_result = {
                    'has_meeple': False,
                    'color': None,
                    'position': None
                }
            
            # Limpiar
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Guardar resultado
            result = {
                'tile_index': i,
                'grid_position': (tile.grid_row, tile.grid_col),
                'bbox': tile.bbox,
                'tile_type': tile_type,
                'rotation': rotation,
                'has_meeple': meeple_result['has_meeple'],
                'meeple_color': meeple_result.get('color'),
                'meeple_position': meeple_result.get('position')
            }
            
            self.results.append(result)
            
            # Actualizar estadísticas
            self._update_stats(stats, result, meeple_result)
        
        print()  # Nueva línea
        return stats
    
    def _update_stats(self, stats: Dict, result: Dict, meeple_result: Dict):
        """Actualiza las estadísticas con la loseta procesada"""
        # Solo contar losetas que no son BLANCO
        tile_type = result['tile_type']
        
        if tile_type == 'BLANCO' or tile_type == '?':
            # No contar en estadísticas
            return
        
        if meeple_result['has_meeple']:
            stats['tiles_with_meeple'] += 1
            if meeple_result.get('color') == 'blue':
                stats['blue_meeples'] += 1
            elif meeple_result.get('color') == 'black':
                stats['black_meeples'] += 1
        
        if tile_type not in stats['tile_types']:
            stats['tile_types'][tile_type] = 0
        stats['tile_types'][tile_type] += 1
        
        rotation = result['rotation']
        if rotation in stats['rotations']:
            stats['rotations'][rotation] += 1
    
    def _create_board_matrix(self) -> Board:
        """
        Crea la matriz Board según origin_matrix.py
        Maneja correctamente las coordenadas negativas usando offsets
        """
        # Inicializar matriz vacía con las dimensiones correctas
        matrix: List[List[Optional[Tile]]] = [[None] * self.cols for _ in range(self.rows)]
        
        # Llenar matriz con los datos detectados
        for result in self.results:
            grid_row, grid_col = result['grid_position']
            
            # Convertir coordenadas de grilla a índices de matriz
            # Restamos min_row y min_col para manejar coordenadas negativas
            matrix_row = grid_row - self.min_row
            matrix_col = grid_col - self.min_col
            
            # Verificar límites (por seguridad)
            if 0 <= matrix_row < self.rows and 0 <= matrix_col < self.cols:
                tile_type = result['tile_type']
                
                # FILTRO: Ignorar losetas detectadas como BLANCO
                if tile_type == 'BLANCO' or tile_type == '?':
                    # Dejar como None (casilla vacía)
                    continue
                
                rotation = result['rotation']
                
                # Procesar información de meeple
                meeple_info = None
                if result['has_meeple'] and result['meeple_position'] is not None:
                    # Determinar jugador según color
                    if result['meeple_color'] == 'blue':
                        player = 1  # Jugador 1 = azul/morado
                    elif result['meeple_color'] == 'black':
                        player = 2  # Jugador 2 = negro
                    else:
                        player = 0  # Desconocido
                    
                    # Posición del meeple (1-9 según grid)
                    meeple_pos = result['meeple_position'] + 1 if result['meeple_position'] is not None else 5
                    
                    meeple_info = (player, meeple_pos)
                
                # Crear objeto Tile
                tile = Tile(
                    type=tile_type,
                    rotation=rotation,
                    meeple_info=meeple_info
                )
                
                # Asignar a la matriz
                matrix[matrix_row][matrix_col] = tile
        
        return Board(board=matrix)
    
    def _save_all_outputs(self, stats: dict):
        """Guarda todos los archivos de salida"""
        print("\n💾 Guardando resultados...")
        
        # 1. Guardar losetas individuales
        self.save_tiles_with_complete_info()
        
        # 2. Guardar JSON detallado
        self.save_results_json()
        
        # 3. Guardar Board Matrix
        self.save_board_matrix_json()
        
        # 4. Visualización
        self.create_visualization("resultado_completo.png")
        
        # 5. NUEVO: Generar imagen del tablero usando BoardImageGenerator
        self.generate_board_image()
        
        print("✓ Todos los archivos guardados")
    
    def generate_board_image(self, output_path: str = "tablero_generado.jpg"):
        """
        Genera una imagen del tablero usando BoardImageGenerator
        """
        if self.board_matrix is None:
            print("⚠ No hay Board generado")
            return
        
        try:
            # Usar la carpeta tiles con las texturas
            tiles_folder = "tiles"
            
            if not os.path.exists(tiles_folder):
                print(f"⚠ Carpeta de tiles no encontrada: {tiles_folder}")
                return
            
            # Crear generador de imágenes
            generator = BoardImageGenerator(tiles_folder=tiles_folder, tile_size=200)
            
            # Generar imagen
            board_img = generator.generate_board_image(self.board_matrix, output_path)
            
            print(f"✓ Imagen del tablero generada: {output_path}")
            
        except Exception as e:
            print(f"⚠ Error generando imagen del tablero: {e}")
            import traceback
            traceback.print_exc()
    
    def save_board_matrix_json(self, output_path: str = "board_matrix.json"):
        """
        Guarda la Origin Matrix en formato JSON legible
        Incluye información sobre los offsets de la grilla
        """
        if self.board_matrix is None:
            print("⚠ No hay Board generado")
            return
        
        # Convertir Board a formato serializable
        board_data = {
            'dimensions': {
                'rows': self.rows,
                'cols': self.cols
            },
            'grid_offsets': {
                'min_row': self.min_row,
                'max_row': self.max_row,
                'min_col': self.min_col,
                'max_col': self.max_col
            },
            'board': []
        }
        
        tiles_count = 0  # Contador de tiles no vacíos
        
        for row_idx, row in enumerate(self.board_matrix.board):
            row_data = []
            # Calcular la coordenada real de grilla
            real_grid_row = row_idx + self.min_row
            
            for col_idx, tile in enumerate(row):
                real_grid_col = col_idx + self.min_col
                
                if tile is None:
                    tile_data = {
                        'grid_position': [real_grid_row, real_grid_col],
                        'tile': None
                    }
                else:
                    tiles_count += 1
                    tile_data = {
                        'grid_position': [real_grid_row, real_grid_col],
                        'tile': {
                            'type': tile.type,
                            'rotation': tile.rotation,
                            'meeple': None
                        }
                    }
                    
                    if tile.meeple_info is not None:
                        player, position = tile.meeple_info
                        tile_data['tile']['meeple'] = {
                            'player': player,
                            'player_name': 'blue' if player == 1 else 'black' if player == 2 else 'unknown',
                            'position': position
                        }
                
                row_data.append(tile_data)
            
            board_data['board'].append(row_data)
        
        # Agregar información adicional
        board_data['summary'] = {
            'total_tiles': tiles_count,
            'empty_cells': (self.rows * self.cols) - tiles_count
        }
        
        # Guardar JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(board_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Origin Matrix guardada en: {output_path}")
    
    def print_board_matrix(self):
        """
        Imprime la matriz del tablero en formato legible
        """
        if self.board_matrix is None:
            print("⚠ No hay Board generado")
            return
        
        print("\n" + "=" * 80)
        print("ORIGIN MATRIX (BOARD)")
        print("=" * 80)
        
        print(f"\nDimensiones: {self.rows}x{self.cols}")
        print(f"Rango grilla: filas [{self.min_row}, {self.max_row}], columnas [{self.min_col}, {self.max_col}]")
        print("\nLeyenda: [Tipo-Rotación°] Meeple(Jugador,Posición)")
        print("Jugador: 1=Azul, 2=Negro\n")
        
        # Imprimir encabezado de columnas (coordenadas reales)
        print("       ", end="")
        for col_idx in range(self.cols):
            real_col = col_idx + self.min_col
            print(f"  C{real_col:+3d}  ", end="")
        print()
        
        # Imprimir cada fila
        for row_idx, row in enumerate(self.board_matrix.board):
            real_row = row_idx + self.min_row
            print(f"F{real_row:+3d} ", end="")
            
            for tile in row:
                if tile is None:
                    print("[  VACÍO  ]", end=" ")
                else:
                    # Formato: [A-90°] o [A-90°]M(1,5)
                    base = f"[{tile.type}-{tile.rotation}°]"
                    
                    if tile.meeple_info is not None:
                        player, pos = tile.meeple_info
                        base += f"M({player},{pos})"
                    
                    # Ajustar ancho
                    print(f"{base:11s}", end=" ")
            print()
        
        print("\n" + "=" * 80)
    
    def get_board_matrix(self) -> Optional[Board]:
        """Retorna la Origin Matrix generada"""
        return self.board_matrix
    
    def create_visualization(self, output_path: str = "resultado_completo.png"):
        """Crea visualización con toda la información"""
        result = self.tile_detector.image.copy()
        
        for data in self.results:
            x, y, w, h = data['bbox']
            
            # Color del borde según meeple
            if data['has_meeple']:
                if data['meeple_color'] == 'blue':
                    border_color = (255, 0, 0)
                elif data['meeple_color'] == 'black':
                    border_color = (50, 50, 50)
                else:
                    border_color = (0, 255, 255)
                thickness = 4
            else:
                border_color = (0, 255, 0)
                thickness = 2
            
            cv2.rectangle(result, (x, y), (x + w, y + h), border_color, thickness)
            
            # Etiqueta
            tile_type = data['tile_type']
            rotation = data['rotation']
            grid_row, grid_col = data['grid_position']
            
            if data['has_meeple'] and data['meeple_position'] is not None:
                color_initial = data['meeple_color'][0].upper() if data['meeple_color'] else '?'
                meeple_pos = data['meeple_position'] + 1
                label = f"{tile_type}-{rotation}° [{color_initial}{meeple_pos}] ({grid_row},{grid_col})"
            else:
                label = f"{tile_type}-{rotation}° ({grid_row},{grid_col})"
            
            # Dibujar etiqueta
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Fondo
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y - text_height - 10), 
                         (x + text_width + 5, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
            
            # Texto
            cv2.putText(result, label, (x + 2, y - 5),
                       font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imwrite(output_path, result)
        print(f"✓ Visualización guardada: {output_path}")
        return result
    
    def save_tiles_with_complete_info(self, output_dir: str = "tiles_complete"):
        """Guarda cada loseta con toda la información en el nombre"""
        import shutil
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        for i, data in enumerate(self.results):
            tile = self.tile_detector.tiles[data['tile_index']]
            row, col = data['grid_position']
            tile_type = data['tile_type']
            rotation = data['rotation']
            
            if data['has_meeple'] and data['meeple_position'] is not None:
                color = data['meeple_color'] or 'unknown'
                pos = data['meeple_position'] + 1
                filename = f"tile_{i:03d}_r{row:+03d}_c{col:+03d}_{tile_type}_rot{rotation}_{color}_pos{pos}.png"
            else:
                filename = f"tile_{i:03d}_r{row:+03d}_c{col:+03d}_{tile_type}_rot{rotation}_empty.png"
            
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, tile.image)
        
        print(f"✓ {len(self.results)} losetas guardadas en '{output_dir}/'")
    
    def save_results_json(self, output_path: str = "deteccion_completa.json"):
        """Guarda resultados detallados en JSON"""
        output_data = {
            'total_tiles': len(self.results),
            'grid_info': {
                'min_row': self.min_row,
                'max_row': self.max_row,
                'min_col': self.min_col,
                'max_col': self.max_col
            },
            'tiles': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Resultados JSON guardados: {output_path}")
    
    def _print_statistics(self, stats: dict):
        """Imprime estadísticas finales"""
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS COMPLETAS")
        print("=" * 80)
        
        # Calcular tiles reales (excluyendo BLANCO)
        total_real_tiles = sum(stats['tile_types'].values())
        total_detected = stats['total_tiles']
        blank_tiles = total_detected - total_real_tiles
        
        print(f"\n🗺️ TABLERO:")
        print(f"  Dimensiones: {self.rows}x{self.cols}")
        print(f"  Rango grilla: filas [{self.min_row}, {self.max_row}], cols [{self.min_col}, {self.max_col}]")
        print(f"  Total detecciones: {total_detected}")
        print(f"  Losetas válidas: {total_real_tiles}")
        print(f"  Espacios vacíos (BLANCO): {blank_tiles}")
        
        print(f"\n🎲 TIPOS DE LOSETAS:")
        for tile_type, count in sorted(stats['tile_types'].items()):
            print(f"  Tipo {tile_type}: {count}")
        
        print(f"\n🔄 ROTACIONES:")
        for rotation, count in sorted(stats['rotations'].items()):
            print(f"  {rotation}°: {count}")
        
        if total_real_tiles > 0:
            print(f"\n👤 MEEPLES:")
            print(f"  Losetas con meeple: {stats['tiles_with_meeple']} ({stats['tiles_with_meeple']/total_real_tiles*100:.1f}%)")
            print(f"  🔵 Meeples azules: {stats['blue_meeples']}")
            print(f"  ⚫ Meeples negros: {stats['black_meeples']}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python pipelineFinal.py <imagen_tablero>")
        print("\nEste pipeline detecta:")
        print("  1. Recorta todas las losetas del tablero")
        print("  2. Detecta el TIPO (A-X)")
        print("  3. Detecta la ROTACIÓN (0°, 90°, 180°, 270°)")
        print("  4. Detecta MEEPLES (color y posición)")
        print("  5. Genera ORIGIN MATRIX (Board)")
        print("  6. Genera imagen del tablero usando referencias")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ No se encontró la imagen: {image_path}")
        return
    
    # Crear pipeline
    print("\n🚀 Iniciando pipeline completo...")
    pipeline = CarcassonneCompletePipeline()
    
    # Procesar tablero
    board = pipeline.process_board(image_path, num_reference_points=8)
    
    if board is None:
        print("\n❌ Procesamiento fallido")
        return
    
    # Imprimir matriz
    pipeline.print_board_matrix()
    
    # Mostrar archivos generados
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 80)
    print("\n📁 Archivos generados:")
    print("  📊 board_matrix.json         - Origin Matrix (formato JSON)")
    print("  📄 deteccion_completa.json   - Datos detallados")
    print("  📂 tiles_complete/           - Losetas individuales")
    print("  🖼️  resultado_completo.png    - Visualización de detección")
    print("  🎨 tablero_generado.jpg      - Imagen del tablero renderizado")
    
    # Preguntar si mostrar visualización
    show_viz = input("\n¿Ver visualización? (s/n): ")
    if show_viz.lower() == 's':
        result = cv2.imread("resultado_completo.png")
        if result is not None:
            cv2.namedWindow("Resultado Completo", cv2.WINDOW_NORMAL)
            cv2.imshow("Resultado Completo", result)
            print("\nPresiona cualquier tecla para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n✨ ¡Todo listo!")


if __name__ == "__main__":
    main()
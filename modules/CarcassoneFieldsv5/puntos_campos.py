"""
Programa principal para análisis de campos en Carcassonne.
Incluye detección de límites del tablero y castillos incompletos.
"""
import sys
import os
import cv2
import shutil
import numpy as np

# Configurar encoding para Windows
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from typing import Dict
from src.image_processor import ImageProcessor
from src.board_detector import BoardDetector
from src.castle_analyzer import CastleAnalyzer
from src.field_detector import FieldDetector
from src.scoring import FieldScorer
from src.visualizer import FieldVisualizer
from config.colors import PLAYER_NAMES, FIELD_DETECTION_CONFIG, WHITE_THRESHOLD


def calculate_field_scores(image_path: str) -> Dict[int, int]:
    """
    Calcula puntos de campos para un tablero.
    FUNCIÓN PÚBLICA PARA INTEGRACIÓN CON OTROS PROGRAMAS.
    
    Args:
        image_path: Ruta de la imagen del tablero
    
    Returns:
        Dict[int, int]: {1: puntos_jugador_1, 2: puntos_jugador_2}
    
    Example:
        >>> scores = calculate_field_scores("tablero.png")
        >>> print(scores)  # {1: 6, 2: 9}
    """
    # Crear carpeta incremental para resultados
    base_results_folder = "resultados"
    results_folder = create_incremental_folder(base_results_folder)
    
    # Copiar imagen original a la carpeta incremental de resultados
    original_filename = os.path.basename(image_path)
    destination_path = os.path.join(results_folder, original_filename)
    shutil.copy2(image_path, destination_path)
    print(f"Imagen original copiada a: {destination_path}")
    
    # 1. Procesar imagen
    processor = ImageProcessor(image_path)
    
    # 2. Detectar límites del tablero
    board_detector = BoardDetector(processor.image)
    board_mask = board_detector.create_board_mask()
    white_areas = board_detector.detect_white_areas()
    
    # 3. Crear máscaras
    field_mask = board_detector.filter_mask_by_board(processor.create_mask('FIELD'))
    castle_mask = board_detector.filter_mask_by_board(processor.create_mask('CASTLE'))
    barrier_mask = board_detector.filter_mask_by_board(processor.get_combined_barrier_mask())
    
    meeple_masks = {
        'MEEPLE_1': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_1')),
        'MEEPLE_2': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_2')),
    }
    
    # 4. Analizar castillos
    castle_analyzer = CastleAnalyzer(castle_mask, board_detector)
    
    # 5. Detectar campos
    detector = FieldDetector(field_mask, barrier_mask, castle_mask)
    config = FIELD_DETECTION_CONFIG
    labeled_fields, num_fields = detector.detect_fields(
        expand_barriers_iterations=config['barrier_expansion'],
        min_area=config['min_field_area']
    )
    
    road_mask = board_detector.filter_mask_by_board(processor.create_mask('ROAD'))
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks, road_mask=road_mask)
    
    # 6. Calcular puntuación
    scorer = FieldScorer(castle_mask, castle_analyzer=castle_analyzer)
    field_results = scorer.calculate_all_scores(fields)
    player_totals = scorer.calculate_player_totals(field_results)
    
    # 7. Crear estructura de carpetas organizada
    imagen_original_folder = os.path.join(results_folder, "imagen_original")
    resultados_folder = os.path.join(results_folder, "resultados_principales")
    campos_folder = os.path.join(results_folder, "analisis_campos")
    debug_folder = os.path.join(results_folder, "debug")
    debug_profundo_folder = os.path.join(results_folder, "debug_profundo")
    
    os.makedirs(imagen_original_folder, exist_ok=True)
    os.makedirs(resultados_folder, exist_ok=True)
    os.makedirs(campos_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    os.makedirs(debug_profundo_folder, exist_ok=True)
    
    # Mover imagen original a su carpeta
    shutil.move(destination_path, os.path.join(imagen_original_folder, original_filename))
    
    # 8. Generar y guardar visualizaciones
    visualizer = FieldVisualizer(processor.image)
    
    # Resultados principales
    result_image = visualizer.draw_field_boundaries(fields, field_results)
    cv2.imwrite(os.path.join(resultados_folder, "resultado.png"), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    summary_image = visualizer.create_summary_image(field_results, player_totals)
    cv2.imwrite(os.path.join(resultados_folder, "resultado_summary.png"), cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
    
    # Análisis de campos
    castle_viz = visualizer.visualize_all_castles(castle_analyzer, fields, scorer)
    cv2.imwrite(os.path.join(campos_folder, "castillos_detectados.png"), cv2.cvtColor(castle_viz, cv2.COLOR_RGB2BGR))
    
    scorer.save_castle_details(fields, field_results, campos_folder, processor.image, meeple_masks)
    
    # Debug (meeples y otros)
    meeple_validity = detector.analyze_meeple_validity(meeple_masks, labeled_fields, road_mask=road_mask)
    meeples_viz = visualizer.visualize_meeples(meeple_masks, meeple_validity)
    cv2.imwrite(os.path.join(debug_folder, "meeples_debug.png"), cv2.cvtColor(meeples_viz, cv2.COLOR_RGB2BGR))
    
    debug_board = processor.image.copy()
    debug_board[white_areas] = [255, 0, 0]
    cv2.imwrite(os.path.join(debug_folder, "debug_board_limits.png"), cv2.cvtColor(debug_board, cv2.COLOR_RGB2BGR))
    
    debug_image = processor.image.copy()
    for field in fields:
        debug_image[field.pixels] = [255, 255, 0]
    cv2.imwrite(os.path.join(debug_folder, "debug_campos_limpios.png"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    # 9. Crear debug profundo con visualizaciones detalladas
    create_deep_debug_visualizations(
        processor, board_detector, fields, meeple_masks, labeled_fields,
        castle_mask, field_mask, barrier_mask, road_mask, white_areas,
        castle_analyzer, scorer, field_results, debug_profundo_folder
    )
    
    # 10. Convertir formato: {'MEEPLE_1': x, 'MEEPLE_2': y} -> {1: x, 2: y}
    return {
        1: player_totals.get('MEEPLE_1', 0),
        2: player_totals.get('MEEPLE_2', 0)
    }


def print_safe(text):
    """Imprime texto de forma segura en cualquier plataforma."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def create_deep_debug_visualizations(processor, board_detector, fields, meeple_masks, 
                                     labeled_fields, castle_mask, field_mask, barrier_mask,
                                     road_mask, white_areas, castle_analyzer, scorer, 
                                     field_results, debug_profundo_folder):
    """
    Crea visualizaciones detalladas de debug para análisis profundo.
    Más de 10 visualizaciones mostrando todas las relaciones y procesos.
    """
    
    # 1. Mapa de calor de densidad de campos
    print_safe("   [Debug Profundo 1/15] Mapa de calor de densidad de campos...")
    heat_map = np.zeros_like(processor.image)
    for i, field in enumerate(fields):
        color_intensity = int((i / len(fields)) * 255)
        heat_map[field.pixels] = [color_intensity, 255 - color_intensity, 128]
    cv2.imwrite(os.path.join(debug_profundo_folder, "01_mapa_calor_campos.png"), 
                cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR))
    
    # 2. Relación Meeple-Campo individual para cada meeple
    print_safe("   [Debug Profundo 2/15] Relaciones individuales meeple-campo...")
    for player in ['MEEPLE_1', 'MEEPLE_2']:
        debug_img = processor.image.copy()
        meeple_pixels = np.where(meeple_masks[player])
        
        # Marcar cada meeple y su campo correspondiente
        for field in fields:
            if field.meeples.get(player, 0) > 0:
                # Campo en verde transparente
                overlay = debug_img.copy()
                overlay[field.pixels] = [0, 255, 0]
                debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
                
                # Meeples en rojo brillante
                debug_img[meeple_masks[player]] = [255, 0, 0]
                
                # Añadir texto con ID del campo
                if len(field.pixels[0]) > 0:
                    center_y = int(np.mean(field.pixels[0]))
                    center_x = int(np.mean(field.pixels[1]))
                    cv2.putText(debug_img, f"Campo {field.id}", (center_x, center_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        player_num = player.split('_')[1]
        cv2.imwrite(os.path.join(debug_profundo_folder, f"02_relacion_meeple_{player_num}_campos.png"),
                   cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    # 3. Campos numerados con áreas
    print_safe("   [Debug Profundo 3/15] Campos numerados con información de área...")
    numbered_img = processor.image.copy()
    for field in fields:
        # Color único por campo
        color = [np.random.randint(50, 255) for _ in range(3)]
        numbered_img[field.pixels] = color
        
        # Añadir número y área
        if len(field.pixels[0]) > 0:
            center_y = int(np.mean(field.pixels[0]))
            center_x = int(np.mean(field.pixels[1]))
            text = f"#{field.id}\n{field.area}px"
            cv2.putText(numbered_img, f"#{field.id}", (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(numbered_img, f"{field.area}px", (center_x, center_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "03_campos_numerados_con_areas.png"),
               cv2.cvtColor(numbered_img, cv2.COLOR_RGB2BGR))
    
    # 4. Máscaras base (campo, castillo, barrera) separadas
    print_safe("   [Debug Profundo 4/15] Máscaras base individuales...")
    masks_img = np.zeros((processor.image.shape[0], processor.image.shape[1] * 3, 3), dtype=np.uint8)
    
    # Campo en verde
    masks_img[:, :processor.image.shape[1]][field_mask] = [0, 255, 0]
    cv2.putText(masks_img, "CAMPOS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Castillos en gris
    masks_img[:, processor.image.shape[1]:processor.image.shape[1]*2][castle_mask] = [150, 150, 150]
    cv2.putText(masks_img, "CASTILLOS", (processor.image.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Barreras en rojo
    masks_img[:, processor.image.shape[1]*2:][barrier_mask] = [255, 0, 0]
    cv2.putText(masks_img, "BARRERAS", (processor.image.shape[1] * 2 + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "04_mascaras_base_comparacion.png"),
               cv2.cvtColor(masks_img, cv2.COLOR_RGB2BGR))
    
    # 5. Detección de bordes de campos
    print_safe("   [Debug Profundo 5/15] Bordes de campos detectados...")
    edges_img = processor.image.copy()
    for field in fields:
        field_mask_single = np.zeros(processor.image.shape[:2], dtype=np.uint8)
        field_mask_single[field.pixels] = 255
        
        # Detectar bordes
        contours, _ = cv2.findContours(field_mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges_img, contours, -1, (255, 255, 0), 2)
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "05_bordes_campos.png"),
               cv2.cvtColor(edges_img, cv2.COLOR_RGB2BGR))
    
    # 6. Campos con meeples vs sin meeples
    print_safe("   [Debug Profundo 6/15] Comparación campos con/sin meeples...")
    meeple_comparison = processor.image.copy()
    for field in fields:
        total_meeples = sum(field.meeples.values())
        if total_meeples > 0:
            # Verde brillante si tiene meeples
            meeple_comparison[field.pixels] = [0, 255, 0]
        else:
            # Rojo si no tiene meeples
            meeple_comparison[field.pixels] = [255, 0, 0]
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "06_campos_con_sin_meeples.png"),
               cv2.cvtColor(meeple_comparison, cv2.COLOR_RGB2BGR))
    
    # 7. Tamaño de campos (pequeño, mediano, grande)
    print_safe("   [Debug Profundo 7/15] Clasificación por tamaño de campos...")
    areas = [f.area for f in fields]
    tercil_1 = np.percentile(areas, 33)
    tercil_2 = np.percentile(areas, 66)
    
    size_img = processor.image.copy()
    for field in fields:
        if field.area < tercil_1:
            color = [255, 0, 0]  # Pequeño - Rojo
        elif field.area < tercil_2:
            color = [255, 255, 0]  # Mediano - Amarillo
        else:
            color = [0, 255, 0]  # Grande - Verde
        
        size_img[field.pixels] = color
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "07_clasificacion_tamano_campos.png"),
               cv2.cvtColor(size_img, cv2.COLOR_RGB2BGR))
    
    # 8. Castillos completos vs incompletos
    print_safe("   [Debug Profundo 8/15] Castillos completos vs incompletos...")
    castle_status_img = processor.image.copy()
    
    for castle_id in range(1, castle_analyzer.num_castles + 1):
        castle_pixels = (castle_analyzer.labeled_castles == castle_id)
        is_complete = castle_id in castle_analyzer.complete_castles
        
        if is_complete:
            castle_status_img[castle_pixels] = [0, 255, 0]  # Verde - Completo
        else:
            castle_status_img[castle_pixels] = [255, 165, 0]  # Naranja - Incompleto
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "08_castillos_completos_vs_incompletos.png"),
               cv2.cvtColor(castle_status_img, cv2.COLOR_RGB2BGR))
    
    # 9. Proximidad de meeples a castillos
    print_safe("   [Debug Profundo 9/15] Proximidad meeples-castillos...")
    proximity_img = processor.image.copy()
    
    # Dilatar castillos para mostrar área de proximidad
    castle_dilated = cv2.dilate(castle_mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    proximity_img[castle_dilated > 0] = [255, 255, 0]  # Amarillo - Zona de proximidad
    proximity_img[castle_mask] = [150, 150, 150]  # Gris - Castillos
    
    # Marcar meeples
    for player, mask in meeple_masks.items():
        if player == 'MEEPLE_1':
            proximity_img[mask] = [255, 0, 0]  # Rojo
        else:
            proximity_img[mask] = [0, 0, 255]  # Azul
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "09_proximidad_meeples_castillos.png"),
               cv2.cvtColor(proximity_img, cv2.COLOR_RGB2BGR))
    
    # 10. Campos que tocan caminos
    print_safe("   [Debug Profundo 10/15] Campos adyacentes a caminos...")
    road_adjacency = processor.image.copy()
    road_dilated = cv2.dilate(road_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    
    for field in fields:
        field_mask_single = np.zeros(processor.image.shape[:2], dtype=np.uint8)
        field_mask_single[field.pixels] = 255
        
        # Verificar si toca caminos
        touches_road = np.any(np.logical_and(field_mask_single, road_dilated))
        
        if touches_road:
            road_adjacency[field.pixels] = [255, 0, 255]  # Magenta - Toca camino
        else:
            road_adjacency[field.pixels] = [0, 255, 255]  # Cyan - No toca camino
    
    # Mostrar caminos en blanco
    road_adjacency[road_mask] = [255, 255, 255]
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "10_campos_adyacentes_caminos.png"),
               cv2.cvtColor(road_adjacency, cv2.COLOR_RGB2BGR))
    
    # 11. Distribución de puntos por campo
    print_safe("   [Debug Profundo 11/15] Distribución de puntos por campo...")
    points_img = processor.image.copy()
    
    max_points = max([fr['score'] for fr in field_results.values()]) if field_results else 1
    
    for field_id, result in field_results.items():
        field = next(f for f in fields if f.id == field_id)
        points = result['score']
        
        # Color basado en puntos (gradiente de azul a rojo)
        if max_points > 0:
            intensity = int((points / max_points) * 255)
            color = [intensity, 0, 255 - intensity]
        else:
            color = [0, 0, 255]
        
        points_img[field.pixels] = color
        
        # Añadir texto con puntos
        if len(field.pixels[0]) > 0:
            center_y = int(np.mean(field.pixels[0]))
            center_x = int(np.mean(field.pixels[1]))
            cv2.putText(points_img, f"{points}pts", (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "11_distribucion_puntos_campos.png"),
               cv2.cvtColor(points_img, cv2.COLOR_RGB2BGR))
    
    # 12. Mapa de control de jugadores
    print_safe("   [Debug Profundo 12/15] Mapa de control por jugador...")
    control_img = processor.image.copy()
    
    for field_id, result in field_results.items():
        field = next(f for f in fields if f.id == field_id)
        owner = result['owner']
        is_tie = result['is_tie']
        
        if is_tie:
            control_img[field.pixels] = [255, 0, 255]  # Magenta - Empate
        elif owner == 'MEEPLE_1':
            control_img[field.pixels] = [255, 0, 0]  # Rojo - Jugador 1
        elif owner == 'MEEPLE_2':
            control_img[field.pixels] = [0, 0, 255]  # Azul - Jugador 2
        else:
            control_img[field.pixels] = [128, 128, 128]  # Gris - Sin dueño
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "12_mapa_control_jugadores.png"),
               cv2.cvtColor(control_img, cv2.COLOR_RGB2BGR))
    
    # 13. Conteo de meeples por campo (visual)
    print_safe("   [Debug Profundo 13/15] Conteo visual de meeples por campo...")
    meeple_count_img = processor.image.copy()
    
    for field in fields:
        total = sum(field.meeples.values())
        
        # Color según cantidad de meeples
        if total == 0:
            color = [50, 50, 50]  # Gris oscuro
        elif total == 1:
            color = [0, 255, 0]  # Verde
        elif total == 2:
            color = [255, 255, 0]  # Amarillo
        else:
            color = [255, 0, 0]  # Rojo
        
        meeple_count_img[field.pixels] = color
        
        # Texto con conteo
        if len(field.pixels[0]) > 0:
            center_y = int(np.mean(field.pixels[0]))
            center_x = int(np.mean(field.pixels[1]))
            cv2.putText(meeple_count_img, f"{total}M", (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "13_conteo_meeples_por_campo.png"),
               cv2.cvtColor(meeple_count_img, cv2.COLOR_RGB2BGR))
    
    # 14. Área del tablero vs área blanca
    print_safe("   [Debug Profundo 14/15] Visualización tablero vs fuera...")
    board_area_img = processor.image.copy()
    board_area_img[white_areas] = [255, 0, 0]  # Rojo - Fuera del tablero
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "14_area_tablero_vs_fuera.png"),
               cv2.cvtColor(board_area_img, cv2.COLOR_RGB2BGR))
    
    # 15. Superposición completa de todos los elementos
    print_safe("   [Debug Profundo 15/15] Superposición completa de elementos...")
    overlay_img = processor.image.copy()
    
    # Campos con transparencia
    for i, field in enumerate(fields):
        color = [np.random.randint(100, 255) for _ in range(3)]
        temp = overlay_img.copy()
        temp[field.pixels] = color
        overlay_img = cv2.addWeighted(overlay_img, 0.7, temp, 0.3, 0)
    
    # Castillos en amarillo
    overlay_img[castle_mask] = [255, 255, 0]
    
    # Caminos en blanco
    overlay_img[road_mask] = [255, 255, 255]
    
    # Meeples en colores brillantes
    overlay_img[meeple_masks['MEEPLE_1']] = [255, 0, 0]
    overlay_img[meeple_masks['MEEPLE_2']] = [0, 0, 255]
    
    cv2.imwrite(os.path.join(debug_profundo_folder, "15_superposicion_completa.png"),
               cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    
    print_safe(f"   [OK] 15 visualizaciones de debug profundo creadas en: {debug_profundo_folder}")


class Logger:
    """Captura toda la salida de consola a un archivo."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def create_incremental_folder(base_folder):
    """Crea una carpeta incremental dentro de base_folder."""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    folder_index = 1
    while True:
        new_folder = os.path.join(base_folder, f"resultados{folder_index}")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            return new_folder
        folder_index += 1


def main(image_path: str, output_path: str = None):
    """
    Ejecuta el análisis completo de campos con visualización y debug.
    
    Args:
        image_path: Ruta de la imagen del tablero
        output_path: Ruta para guardar resultado (opcional)
    """
    # Crear carpeta incremental para resultados
    base_results_folder = "resultados"
    results_folder = create_incremental_folder(base_results_folder)
    
    # Crear estructura de carpetas organizada primero
    imagen_original_folder = os.path.join(results_folder, "imagen_original")
    resultados_folder = os.path.join(results_folder, "resultados_principales")
    campos_folder = os.path.join(results_folder, "analisis_campos")
    debug_folder = os.path.join(results_folder, "debug")
    debug_profundo_folder = os.path.join(results_folder, "debug_profundo")
    
    os.makedirs(imagen_original_folder, exist_ok=True)
    os.makedirs(resultados_folder, exist_ok=True)
    os.makedirs(campos_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    os.makedirs(debug_profundo_folder, exist_ok=True)
    
    # Copiar imagen original a su carpeta
    original_filename = os.path.basename(image_path)
    destination_path = os.path.join(imagen_original_folder, original_filename)
    shutil.copy2(image_path, destination_path)
    
    # Iniciar logger en la carpeta correcta desde el principio
    log_file = os.path.join(resultados_folder, "log_ejecucion.txt")
    logger = Logger(log_file)
    sys.stdout = logger
    
    print_safe("=" * 60)
    print_safe("ANALISIS DE CAMPOS - CARCASSONNE v3")
    print_safe("Con deteccion de tablero y castillos incompletos")
    print_safe("=" * 60)
    print_safe(f"Carpeta de resultados: {results_folder}")
    print_safe(f"Imagen original: {destination_path}")
    
    # 1. Procesar imagen
    print_safe("\n[1/6] Procesando imagen...")
    processor = ImageProcessor(image_path)
    
    # 1.5 Detectar límites del tablero
    print_safe("[1.5/6] Detectando limites del tablero...")
    board_detector = BoardDetector(processor.image)
    board_mask = board_detector.create_board_mask()
    white_areas = board_detector.detect_white_areas()
    
    print_safe(f"   Pixeles dentro del tablero: {board_mask.sum()}")
    print_safe(f"   Pixeles fuera (blancos): {white_areas.sum()}")
    
    # 2. Crear máscaras
    print_safe("[2/6] Creando mascaras...")
    field_mask = processor.create_mask('FIELD')
    castle_mask = processor.create_mask('CASTLE')
    
    # Filtrar máscaras para solo incluir área del tablero
    field_mask = board_detector.filter_mask_by_board(field_mask)
    castle_mask = board_detector.filter_mask_by_board(castle_mask)
    
    # Analizar castillos completos vs incompletos
    print_safe("[2.5/6] Analizando castillos...")
    castle_analyzer = CastleAnalyzer(castle_mask, board_detector)
    castle_stats = castle_analyzer.get_castle_statistics()
    
    print_safe(f"   Total de castillos: {castle_stats['total_castles']}")
    print_safe(f"   Castillos completos: {castle_stats['complete_castles']}")
    print_safe(f"   Castillos incompletos: {castle_stats['incomplete_castles']}")
    
    # Crear máscara de barreras (incluye TODOS los castillos + caminos)
    barrier_mask = processor.get_combined_barrier_mask()
    barrier_mask = board_detector.filter_mask_by_board(barrier_mask)
    
    meeple_masks = {
        'MEEPLE_1': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_1')),
        'MEEPLE_2': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_2')),
    }
    
    # Debug
    print_safe(f"   Pixeles verdes (campos): {field_mask.sum()}")
    print_safe(f"   Pixeles de barreras: {barrier_mask.sum()}")
    print_safe(f"   Pixeles de castillos: {castle_mask.sum()}")
    
    # 3. Detectar campos
    print_safe("[3/6] Detectando campos...")
    detector = FieldDetector(field_mask, barrier_mask, castle_mask)
    
    config = FIELD_DETECTION_CONFIG
    labeled_fields, num_fields = detector.detect_fields(
        expand_barriers_iterations=config['barrier_expansion'],
        min_area=config['min_field_area']
    )
    
    print_safe(f"   [OK] {num_fields} campos detectados")
    
    road_mask = processor.create_mask('ROAD')
    road_mask = board_detector.filter_mask_by_board(road_mask)
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks, road_mask=road_mask)
    
    # Debug: mostrar info de campos
    print_safe("\n   Detalles de campos:")
    for field in fields:
        print_safe(f"     - Campo {field.id}: {field.area} pixels, "
                  f"Meeples: {sum(field.meeples.values())}")
    
    # 4. Calcular puntuación (SOLO castillos completos)
    print_safe("\n[4/6] Calculando puntuacion...")
    print_safe("   (Solo castillos completos cuentan para puntos)")
    scorer = FieldScorer(castle_mask, castle_analyzer=castle_analyzer)
    field_results = scorer.calculate_all_scores(fields)
    
    player_totals = scorer.calculate_player_totals(field_results)
    
    # 4.5 Guardar información detallada de castillos por campo
    print_safe("\n[4.5/6] Guardando informacion detallada de castillos...")
    scorer.save_castle_details(fields, field_results, campos_folder, processor.image, meeple_masks)
    
    # DEBUG DE MEEPLES
    print_safe("\n" + "=" * 70)
    print_safe("DEBUG DE MEEPLES")
    print_safe("=" * 70)
    
    # Calcular totales de meeples
    total_meeples_1 = sum(field.meeples.get('MEEPLE_1', 0) for field in fields)
    total_meeples_2 = sum(field.meeples.get('MEEPLE_2', 0) for field in fields)
    
    print_safe(f"Total Meeples Jugador 1: {total_meeples_1}")
    print_safe(f"Total Meeples Jugador 2: {total_meeples_2}")
    
    # Listar campos con meeples
    print_safe("\nCampos con Meeples:")
    for field in fields:
        total_m = sum(field.meeples.values())
        if total_m > 0:
            print_safe(f"  Campo {field.id}: {field.meeples}")
    
    # 5. Visualizar resultados
    print_safe("[5/6] Generando visualizacion...")
    visualizer = FieldVisualizer(processor.image)
    result_image = visualizer.draw_field_boundaries(fields, field_results)
    summary_image = visualizer.create_summary_image(field_results, player_totals)
    
    # Generar visualización de castillos
    print_safe("   Generando visualizacion de castillos...")
    castle_viz = visualizer.visualize_all_castles(castle_analyzer, fields, scorer)
    
    # Generar análisis de validez de meeples
    print_safe("   Analizando validez de meeples...")
    meeple_validity = detector.analyze_meeple_validity(
        meeple_masks,
        labeled_fields,
        road_mask=road_mask
    )
    
    # Generar visualización de meeples con clasificación válido/inválido
    print_safe("   Generando debug de meeples...")
    meeples_viz = visualizer.visualize_meeples(meeple_masks, meeple_validity)
    
    # Actualizar rutas de salida organizadas
    result_image_path = os.path.join(resultados_folder, "resultado.png")
    summary_image_path = os.path.join(resultados_folder, "resultado_summary.png")
    castle_image_path = os.path.join(campos_folder, "castillos_detectados.png")
    meeples_image_path = os.path.join(debug_folder, "meeples_debug.png")
    board_mask_path = os.path.join(debug_folder, "debug_board_limits.png")
    fields_clean_path = os.path.join(debug_folder, "debug_campos_limpios.png")

    # Guardar y mostrar imágenes
    cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(summary_image_path, cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(castle_image_path, cv2.cvtColor(castle_viz, cv2.COLOR_RGB2BGR))
    cv2.imwrite(meeples_image_path, cv2.cvtColor(meeples_viz, cv2.COLOR_RGB2BGR))
    
    # Guardar imágenes de debug
    debug_board = processor.image.copy()
    debug_board[white_areas] = [255, 0, 0]
    cv2.imwrite(board_mask_path, cv2.cvtColor(debug_board, cv2.COLOR_RGB2BGR))
    
    debug_image = processor.image.copy()
    for field in fields:
        debug_image[field.pixels] = [255, 255, 0]
    cv2.imwrite(fields_clean_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    # Crear visualizaciones de debug profundo
    print_safe("\n[5.5/6] Generando debug profundo (15 visualizaciones detalladas)...")
    create_deep_debug_visualizations(
        processor, board_detector, fields, meeple_masks, labeled_fields,
        castle_mask, field_mask, barrier_mask, road_mask, white_areas,
        castle_analyzer, scorer, field_results, debug_profundo_folder
    )
    
    print_safe(f"\n[OK] Resultados guardados en: {results_folder}")
    print_safe(f"  Imagen Original: {imagen_original_folder}")
    print_safe(f"  Resultados Principales: {resultados_folder}")
    print_safe(f"  Análisis de Campos: {campos_folder}")
    print_safe(f"  Debug: {debug_folder}")
    print_safe(f"  Debug Profundo: {debug_profundo_folder}")

    # 6. Mostrar resultados en consola
    print_safe("\n" + "=" * 60)
    print_safe("RESULTADOS")
    print_safe("=" * 60)
    
    if len(field_results) == 0:
        print_safe("\n[ADVERTENCIA] No se detectaron campos validos")
    else:
        for field_id, data in sorted(field_results.items()):
            owner_name = PLAYER_NAMES.get(data['owner'], 'Sin dueno')
            if data['is_tie']:
                owner_name = 'EMPATE'
            
            print_safe(f"\nCampo {field_id}:")
            print_safe(f"  Dueno: {owner_name}")
            print_safe(f"  Puntos: {data['score']}")
            
            # Mostrar desglose de castillos
            complete = data.get('castles_complete', data['castles'])
            incomplete = data.get('castles_incomplete', 0)
            
            if incomplete > 0:
                print_safe(f"  Castillos: {complete} completos + {incomplete} incompletos")
                print_safe(f"    -> Solo los {complete} completos cuentan para puntos")
            else:
                print_safe(f"  Castillos completos: {complete}")
            
            print_safe(f"  Meeples: {data['meeples']}")
            print_safe(f"  Area: {data['area']} pixels")
        
        print_safe("\n" + "-" * 60)
        print_safe("PUNTUACION TOTAL:")
        print_safe("-" * 60)
        
        # Mostrar SIEMPRE ambos jugadores
        for player in ['MEEPLE_1', 'MEEPLE_2']:
            player_name = PLAYER_NAMES.get(player, player)
            total = player_totals.get(player, 0)
            print_safe(f"{player_name}: {total} puntos")
    
    # Mostrar imágenes
    cv2.imshow('Campos Detectados', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Resumen', cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Castillos', cv2.cvtColor(castle_viz, cv2.COLOR_RGB2BGR))
    cv2.imshow('Debug Meeples', cv2.cvtColor(meeples_viz, cv2.COLOR_RGB2BGR))
    cv2.imshow('Debug: Limites del Tablero (rojo=fuera)', cv2.cvtColor(debug_board, cv2.COLOR_RGB2BGR))
    cv2.imshow('Debug: Campos Limpios', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    print_safe("\nPresiona cualquier tecla en la ventana de imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Cerrar logger
    logger.close()
    sys.stdout = logger.terminal
    
    return field_results, player_totals
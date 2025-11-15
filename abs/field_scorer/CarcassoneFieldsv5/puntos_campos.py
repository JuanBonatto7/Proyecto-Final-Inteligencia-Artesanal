"""
Programa principal para análisis de campos en Carcassonne.
Incluye detección de límites del tablero y castillos incompletos.
"""
import sys
import os
import cv2

# Configurar encoding para Windows
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from src.image_processor import ImageProcessor
from src.board_detector import BoardDetector
from src.castle_analyzer import CastleAnalyzer
from src.field_detector import FieldDetector
from src.scoring import FieldScorer
from src.visualizer import FieldVisualizer
from config.colors import PLAYER_NAMES, FIELD_DETECTION_CONFIG


def print_safe(text):
    """Imprime texto de forma segura en cualquier plataforma."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


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


def main(image_path: str, output_path: str = None, white_threshold: int = 200):
    """
    Ejecuta el análisis completo de campos.
    
    Args:
        image_path: Ruta de la imagen del tablero
        output_path: Ruta para guardar resultado (opcional)
        white_threshold: Umbral para detectar blanco (0-255)
    """
    print_safe("=" * 60)
    print_safe("ANALISIS DE CAMPOS - CARCASSONNE v3")
    print_safe("Con deteccion de tablero y castillos incompletos")
    print_safe("=" * 60)
    
    # 1. Procesar imagen
    print_safe("\n[1/6] Procesando imagen...")
    processor = ImageProcessor(image_path)
    
    # 1.5 Detectar límites del tablero
    print_safe("[1.5/6] Detectando limites del tablero...")
    board_detector = BoardDetector(processor.image, white_threshold=white_threshold)
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
    detector = FieldDetector(field_mask, barrier_mask)
    
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
    
    # 5. Visualizar resultados
    print_safe("[5/6] Generando visualizacion...")
    visualizer = FieldVisualizer(processor.image)
    result_image = visualizer.draw_field_boundaries(fields, field_results)
    summary_image = visualizer.create_summary_image(field_results, player_totals)
    
    # Crear carpeta incremental para resultados
    base_results_folder = "resultados"
    results_folder = create_incremental_folder(base_results_folder)

    # Actualizar rutas de salida
    result_image_path = os.path.join(results_folder, "resultado.png")
    summary_image_path = os.path.join(results_folder, "resultado_summary.png")

    # Guardar y mostrar imágenes
    cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(summary_image_path, cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
    print_safe(f"\n[OK] Resultados guardados en: {result_image_path}")
    print_safe(f"[OK] Resumen guardado en: {summary_image_path}")

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
    
    # Debug: Mostrar máscara del tablero
    debug_board = processor.image.copy()
    debug_board[white_areas] = [255, 0, 0]  # Blanco en rojo
    cv2.imshow('Debug: Limites del Tablero (rojo=fuera)', cv2.cvtColor(debug_board, cv2.COLOR_RGB2BGR))
    
    # Debug: Campos limpios
    debug_image = processor.image.copy()
    for field in fields:
        debug_image[field.pixels] = [255, 255, 0]  # Amarillo
    cv2.imshow('Debug: Campos Limpios', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    print_safe("\nPresiona cualquier tecla en la ventana de imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return field_results, player_totals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_safe("Uso: python main.py <ruta_imagen> [ruta_salida] [umbral_blanco]")
        print_safe("\nEjemplo:")
        print_safe("  python main.py tablero.png resultado.png")
        print_safe("  python main.py tablero.png resultado.png 210")
        print_safe("\nParametros:")
        print_safe("  umbral_blanco: 0-255, default=200 (mayor=mas estricto para blanco)")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "resultado.png"
    white_thresh = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    if not os.path.exists(input_image):
        print_safe(f"[ERROR] No existe el archivo: {input_image}")
        sys.exit(1)
    
    try:
        main(input_image, output_image, white_threshold=white_thresh)
    except Exception as e:
        print_safe(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
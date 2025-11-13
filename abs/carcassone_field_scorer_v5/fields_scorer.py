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

from .src.image_processor import ImageProcessor
from .src.board_detector import BoardDetector
from .src.castle_analyzer import CastleAnalyzer
from .src.field_detector import FieldDetector
from .src.scoring import FieldScorer
from .src.visualizer import FieldVisualizer
from .config.colors import PLAYER_NAMES, FIELD_DETECTION_CONFIG


def print_safe(text):
    """Imprime texto de forma segura en cualquier plataforma."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def main(image_path: str, output_path: str = None, white_threshold: int = 200):
    """
    Ejecuta el análisis completo de campos.

    Args:
        image_path: Ruta de la imagen del tablero / Path to the board image
        output_path: Ruta para guardar resultado (opcional) / Path to save results (optional)
        white_threshold: Umbral para detectar blanco (0-255) / Threshold to detect white (0-255)
    """
    print_safe("=" * 60)
    print_safe("ANÁLISIS DE CAMPOS - CARCASSONNE v3 / FIELD ANALYSIS - CARCASSONNE v3")
    print_safe("Con detección de tablero y castillos incompletos / With board detection and incomplete castles")
    print_safe("=" * 60)

    # 1. Procesar imagen / Process image
    print_safe("\n[1/6] Procesando imagen... / Processing image...")
    processor = ImageProcessor(image_path)

    # 1.5 Detectar límites del tablero / Detect board limits
    print_safe("[1.5/6] Detectando límites del tablero... / Detecting board limits...")
    board_detector = BoardDetector(processor.image, white_threshold=white_threshold)
    board_mask = board_detector.create_board_mask()
    white_areas = board_detector.detect_white_areas()

    print_safe(f"   Pixeles dentro del tablero: {board_mask.sum()} / Pixels inside the board: {board_mask.sum()}")
    print_safe(f"   Pixeles fuera (blancos): {white_areas.sum()} / Pixels outside (white): {white_areas.sum()}")

    # 2. Crear máscaras / Create masks
    print_safe("[2/6] Creando máscaras... / Creating masks...")
    field_mask = processor.create_mask('FIELD')
    castle_mask = processor.create_mask('CASTLE')

    # Filtrar máscaras para solo incluir área del tablero / Filter masks to include only board area
    field_mask = board_detector.filter_mask_by_board(field_mask)
    castle_mask = board_detector.filter_mask_by_board(castle_mask)

    # Analizar castillos completos vs incompletos / Analyze complete vs incomplete castles
    print_safe("[2.5/6] Analizando castillos... / Analyzing castles...")
    castle_analyzer = CastleAnalyzer(castle_mask, board_detector)
    castle_stats = castle_analyzer.get_castle_statistics()

    print_safe(f"   Total de castillos: {castle_stats['total_castles']} / Total castles: {castle_stats['total_castles']}")
    print_safe(f"   Castillos completos: {castle_stats['complete_castles']} / Complete castles: {castle_stats['complete_castles']}")
    print_safe(f"   Castillos incompletos: {castle_stats['incomplete_castles']} / Incomplete castles: {castle_stats['incomplete_castles']}")

    # Crear máscara de barreras (incluye TODOS los castillos + caminos) / Create barrier mask (includes ALL castles + roads)
    barrier_mask = processor.get_combined_barrier_mask()
    barrier_mask = board_detector.filter_mask_by_board(barrier_mask)

    meeple_masks = {
        'MEEPLE_1': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_1')),
        'MEEPLE_2': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_2')),
    }

    # Debug
    print_safe(f"   Pixeles verdes (campos): {field_mask.sum()} / Green pixels (fields): {field_mask.sum()}")
    print_safe(f"   Pixeles de barreras: {barrier_mask.sum()} / Barrier pixels: {barrier_mask.sum()}")
    print_safe(f"   Pixeles de castillos: {castle_mask.sum()} / Castle pixels: {castle_mask.sum()}")

    # 3. Detectar campos / Detect fields
    print_safe("[3/6] Detectando campos... / Detecting fields...")
    detector = FieldDetector(field_mask, barrier_mask)

    config = FIELD_DETECTION_CONFIG
    labeled_fields, num_fields = detector.detect_fields(
        expand_barriers_iterations=config['barrier_expansion'],
        min_area=config['min_field_area']
    )

    print_safe(f"   [OK] {num_fields} campos detectados / {num_fields} fields detected")

    road_mask = processor.create_mask('ROAD')
    road_mask = board_detector.filter_mask_by_board(road_mask)
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks, road_mask=road_mask)

    # Debug: mostrar info de campos / Debug: show field info
    print_safe("\n   Detalles de campos: / Field details:")
    for field in fields:
        print_safe(f"     - Campo {field.id}: {field.area} pixels, "
                  f"Meeples: {sum(field.meeples.values())}")

    # 4. Calcular puntuación (SOLO castillos completos) / Calculate score (ONLY complete castles)
    print_safe("\n[4/6] Calculando puntuación... / Calculating score...")
    print_safe("   (Solo castillos completos cuentan para puntos / Only complete castles count for points)")
    scorer = FieldScorer(castle_mask, castle_analyzer=castle_analyzer)
    field_results = scorer.calculate_all_scores(fields)
    player_totals = scorer.calculate_player_totals(field_results)

    # 5. Visualizar resultados / Visualize results
    print_safe("[5/6] Generando visualización... / Generating visualization...")
    visualizer = FieldVisualizer(processor.image)
    result_image = visualizer.draw_field_boundaries(fields, field_results)
    summary_image = visualizer.create_summary_image(field_results, player_totals)

    # 6. Mostrar resultados en consola / Show results in console
    print_safe("\n" + "=" * 60)
    print_safe("RESULTADOS / RESULTS")
    print_safe("=" * 60)

    if len(field_results) == 0:
        print_safe("\n[ADVERTENCIA] No se detectaron campos válidos / [WARNING] No valid fields detected")
    else:
        for field_id, data in sorted(field_results.items()):
            owner_name = PLAYER_NAMES.get(data['owner'], 'Sin dueño / No owner')
            if data['is_tie']:
                owner_name = 'EMPATE / TIE'

            print_safe(f"\nCampo {field_id} / Field {field_id}:")
            print_safe(f"  Dueño: {owner_name} / Owner: {owner_name}")
            print_safe(f"  Puntos: {data['score']} / Points: {data['score']}")

            # Mostrar desglose de castillos / Show castle breakdown
            complete = data.get('castles_complete', data['castles'])
            incomplete = data.get('castles_incomplete', 0)

            if incomplete > 0:
                print_safe(f"  Castillos: {complete} completos + {incomplete} incompletos / Castles: {complete} complete + {incomplete} incomplete")
                print_safe(f"    -> Solo los {complete} completos cuentan para puntos / Only the {complete} complete count for points")
            else:
                print_safe(f"  Castillos completos: {complete} / Complete castles: {complete}")

            print_safe(f"  Meeples: {data['meeples']}")
            print_safe(f"  Área: {data['area']} pixels / Area: {data['area']} pixels")

        print_safe("\n" + "-" * 60)
        print_safe("PUNTUACIÓN TOTAL: / TOTAL SCORE:")
        print_safe("-" * 60)

        # Mostrar SIEMPRE ambos jugadores / ALWAYS show both players
        for player in ['MEEPLE_1', 'MEEPLE_2']:
            player_name = PLAYER_NAMES.get(player, player)
            total = player_totals.get(player, 0)
            print_safe(f"{player_name}: {total} puntos / {player_name}: {total} points")

    # Guardar y mostrar imágenes / Save and show images
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        summary_path = output_path.replace('.', '_summary.')
        cv2.imwrite(summary_path, cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
        print_safe(f"\n[OK] Resultados guardados en: {output_path} / Results saved in: {output_path}")
        print_safe(f"[OK] Resumen guardado en: {summary_path} / Summary saved in: {summary_path}")

    # Mostrar imágenes / Show images
    cv2.imshow('Campos Detectados / Detected Fields', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Resumen / Summary', cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))

    # Debug: Mostrar máscara del tablero / Debug: Show board mask
    debug_board = processor.image.copy()
    debug_board[white_areas] = [255, 0, 0]  # Blanco en rojo / White in red
    cv2.imshow('Debug: Límites del Tablero (rojo=fuera) / Debug: Board Limits (red=outside)', cv2.cvtColor(debug_board, cv2.COLOR_RGB2BGR))

    # Debug: Campos limpios / Debug: Clean fields
    debug_image = processor.image.copy()
    for field in fields:
        debug_image[field.pixels] = [255, 255, 0]  # Amarillo / Yellow
    cv2.imshow('Debug: Campos Limpios / Debug: Clean Fields', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

    print_safe("\nPresiona cualquier tecla en la ventana de imagen para cerrar... / Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return field_results, player_totals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_safe("Uso: python main.py <ruta_imagen> [ruta_salida] [umbral_blanco] / Usage: python main.py <image_path> [output_path] [white_threshold]")
        print_safe("\nEjemplo / Example:")
        print_safe("  python main.py tablero.png resultado.png")
        print_safe("  python main.py tablero.png resultado.png 210")
        print_safe("\nParámetros / Parameters:")
        print_safe("  umbral_blanco: 0-255, default=200 (mayor=más estricto para blanco) / white_threshold: 0-255, default=200 (higher=stricter for white)")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "resultado.png"
    white_thresh = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    if not os.path.exists(input_image):
        print_safe(f"[ERROR] No existe el archivo: {input_image} / File does not exist: {input_image}")
        sys.exit(1)

    try:
        main(input_image, output_image, white_threshold=white_thresh)
    except Exception as e:
        print_safe(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
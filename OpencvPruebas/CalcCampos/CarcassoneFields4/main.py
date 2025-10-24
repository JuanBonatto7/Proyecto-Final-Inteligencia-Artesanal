"""
Programa principal para análisis de campos en Carcassonne (MEJORADO).
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
from src.field_detector import FieldDetector
from src.scoring import FieldScorer
from src.visualizer import FieldVisualizer
from config.colors import PLAYER_NAMES, FIELD_DETECTION_CONFIG


def print_safe(text):
    """Imprime texto de forma segura en cualquier plataforma."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remover caracteres especiales
        print(text.encode('ascii', 'replace').decode('ascii'))


def main(image_path: str, output_path: str = None):
    """
    Ejecuta el análisis completo de campos.
    
    Args:
        image_path: Ruta de la imagen del tablero
        output_path: Ruta para guardar resultado (opcional)
    """
    print_safe("=" * 60)
    print_safe("ANALISIS DE CAMPOS - CARCASSONNE (MEJORADO)")
    print_safe("=" * 60)
    
    # 1. Procesar imagen
    print_safe("\n[1/5] Procesando imagen...")
    processor = ImageProcessor(image_path)
    
    # 2. Crear máscaras
    print_safe("[2/5] Creando mascaras...")
    field_mask = processor.create_mask('FIELD')
    barrier_mask = processor.get_combined_barrier_mask()
    castle_mask = processor.create_mask('CASTLE')
    
    meeple_masks = {
        'MEEPLE_1': processor.create_mask('MEEPLE_1'),
        'MEEPLE_2': processor.create_mask('MEEPLE_2'),
    }
    
    # Debug: mostrar estadísticas de máscaras
    print_safe(f"   Pixeles verdes (campos): {field_mask.sum()}")
    print_safe(f"   Pixeles de barreras: {barrier_mask.sum()}")
    print_safe(f"   Pixeles de castillos: {castle_mask.sum()}")
    
    # 3. Detectar campos con configuración mejorada
    print_safe("[3/5] Detectando campos...")
    detector = FieldDetector(field_mask, barrier_mask)
    
    # Usar configuración para mejor detección
    config = FIELD_DETECTION_CONFIG
    labeled_fields, num_fields = detector.detect_fields(
        expand_barriers_iterations=config['barrier_expansion'],
        min_area=config['min_field_area']
    )
    
    print_safe(f"   [OK] {num_fields} campos detectados")
    
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks)
    
    # Debug: mostrar info de campos
    print_safe("\n   Detalles de campos:")
    for field in fields:
        print_safe(f"     - Campo {field.id}: {field.area} pixels, "
                  f"Meeples: {sum(field.meeples.values())}")
    
    # 4. Calcular puntuación
    print_safe("\n[4/5] Calculando puntuacion...")
    scorer = FieldScorer(castle_mask)
    field_results = scorer.calculate_all_scores(fields)
    player_totals = scorer.calculate_player_totals(field_results)
    
    # 5. Visualizar resultados
    print_safe("[5/5] Generando visualizacion...")
    visualizer = FieldVisualizer(processor.image)
    result_image = visualizer.draw_field_boundaries(fields, field_results)
    summary_image = visualizer.create_summary_image(field_results, player_totals)
    
    # Mostrar resultados en consola
    print_safe("\n" + "=" * 60)
    print_safe("RESULTADOS")
    print_safe("=" * 60)
    
    if len(field_results) == 0:
        print_safe("\n[ADVERTENCIA] No se detectaron campos validos")
        print_safe("Posibles causas:")
        print_safe("  - Los colores en la imagen no coinciden con config/colors.py")
        print_safe("  - La tolerancia COLOR_TOLERANCE es muy baja")
        print_safe("  - El area minima es muy alta")
        print_safe("\nEjecuta: python tools/debug_colors.py <imagen>")
    else:
        for field_id, data in sorted(field_results.items()):
            owner_name = PLAYER_NAMES.get(data['owner'], 'Sin dueno')
            if data['is_tie']:
                owner_name = 'EMPATE'
            
            print_safe(f"\nCampo {field_id}:")
            print_safe(f"  Dueno: {owner_name}")
            print_safe(f"  Puntos: {data['score']}")
            print_safe(f"  Castillos adyacentes: {data['castles']}")
            print_safe(f"  Meeples: {data['meeples']}")
            print_safe(f"  Area: {data['area']} pixels")
        
        print_safe("\n" + "-" * 60)
        print_safe("PUNTUACION TOTAL:")
        print_safe("-" * 60)
        
        if len(player_totals) == 0:
            print_safe("Ningun jugador obtuvo puntos")
        else:
            for player, total in sorted(player_totals.items(), key=lambda x: x[1], reverse=True):
                player_name = PLAYER_NAMES.get(player, player)
                print_safe(f"{player_name}: {total} puntos")
    
    # Guardar y mostrar imágenes
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        summary_path = output_path.replace('.', '_summary.')
        cv2.imwrite(summary_path, cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
        print_safe(f"\n[OK] Resultados guardados en: {output_path}")
        print_safe(f"[OK] Resumen guardado en: {summary_path}")
    
    # Mostrar imágenes
    cv2.imshow('Campos Detectados', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Resumen', cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))
    
    # También mostrar máscara de campos detectados para debug
    debug_image = processor.image.copy()
    for field in fields:
        debug_image[field.pixels] = [255, 255, 0]  # Amarillo para campos
    cv2.imshow('Debug: Campos Limpios', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    print_safe("\nPresiona cualquier tecla en la ventana de imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return field_results, player_totals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_safe("Uso: python main.py <ruta_imagen> [ruta_salida]")
        print_safe("\nEjemplo:")
        print_safe("  python main.py tablero.png resultado.png")
        print_safe("\nHerramientas de debug:")
        print_safe("  python tools/debug_colors.py <imagen>    - Ver mascaras de colores")
        print_safe("  python tools/auto_detect_colors.py <imagen> - Detectar colores automaticamente")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "resultado.png"
    
    if not os.path.exists(input_image):
        print_safe(f"[ERROR] No existe el archivo: {input_image}")
        sys.exit(1)
    
    try:
        main(input_image, output_image)
    except Exception as e:
        print_safe(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
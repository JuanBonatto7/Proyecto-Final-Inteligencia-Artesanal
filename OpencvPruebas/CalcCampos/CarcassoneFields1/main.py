"""
Programa principal para análisis de campos en Carcassonne.
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
from config.colors import PLAYER_NAMES


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
    print_safe("ANALISIS DE CAMPOS - CARCASSONNE")
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
    
    # 3. Detectar campos
    print_safe("[3/5] Detectando campos...")
    detector = FieldDetector(field_mask, barrier_mask)
    labeled_fields, num_fields = detector.detect_fields()
    
    print_safe(f"   [OK] {num_fields} campos detectados")
    
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks)
    
    # 4. Calcular puntuación
    print_safe("[4/5] Calculando puntuacion...")
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
    
    for field_id, data in sorted(field_results.items()):
        owner_name = PLAYER_NAMES.get(data['owner'], 'Sin dueno')
        if data['is_tie']:
            owner_name = 'EMPATE'
        
        print_safe(f"\nCampo {field_id}:")
        print_safe(f"  Dueno: {owner_name}")
        print_safe(f"  Puntos: {data['score']}")
        print_safe(f"  Castillos adyacentes: {data['castles']}")
        print_safe(f"  Meeples: {data['meeples']}")
    
    print_safe("\n" + "-" * 60)
    print_safe("PUNTUACION TOTAL:")
    print_safe("-" * 60)
    for player, total in player_totals.items():
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
    print_safe("\nPresiona cualquier tecla en la ventana de imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return field_results, player_totals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_safe("Uso: python main.py <ruta_imagen> [ruta_salida]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "resultado.png"
    
    try:
        main(input_image, output_image)
    except Exception as e:
        print_safe(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
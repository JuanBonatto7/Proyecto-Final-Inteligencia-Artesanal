# ============================================================================
# main.py
# ============================================================================

from ast import Dict

from CalcCampos.carcassonne_scorer.processors.castle_detector import CastleDetector
from CalcCampos.carcassonne_scorer.processors.field_detector import FieldDetector
from CalcCampos.carcassonne_scorer.processors.image_processor import ImageProcessor
from CalcCampos.carcassonne_scorer.scorers.field_scorer import FieldScorer
from CalcCampos.carcassonne_scorer.utils.visualization import Visualizer
from config.settings import Config


def score(image_path: str) -> dict[int, int]:
    """
    Función principal para calcular puntuaciones de campos.
    
    LÓGICA:
    - Cada campo es controlado por el jugador con más meeples
    - Cada campo gana 3 puntos por cada CASTILLO CERRADO que lo toque
    - Un castillo está CERRADO si NO toca ningún píxel blanco
    - En empate, ambos jugadores ganan los puntos del campo
    
    Args:
        image_path: Ruta a la imagen del tablero
        
    Returns:
        Diccionario con puntuaciones {1: puntos_j1, 2: puntos_j2}
    """
    print("\n" + "="*60)
    print("CARCASSONNE FIELD SCORER")
    print("="*60 + "\n")
    
    # 1. Procesar imagen
    print("1. Cargando y procesando imagen...")
    processor = ImageProcessor(image_path)
    processor.load_image()
    processor.create_all_masks()
    print("   ✓ Imagen procesada\n")
    
    # 2. Detectar campos
    print("2. Detectando campos...")
    field_detector = FieldDetector(processor)
    
    # 3. Detectar castillos
    print("3. Analizando castillos...")
    castle_detector = CastleDetector(processor)
    
    # 4. Calcular puntuaciones
    print("4. Calculando puntuaciones...")
    scorer = FieldScorer(processor, field_detector, castle_detector)
    game_state = scorer.score_fields()
    
    # 5. Generar visualizaciones
    print("5. Generando visualizaciones...")
    visualizer = Visualizer()
    visualizer.save_masks(processor)
    visualizer.save_castle_analysis(processor, castle_detector)
    visualizer.save_fields_visualization(processor.image, game_state.fields)
    visualizer.save_final_visualization(processor.image, game_state)
    print()
    
    return game_state.total_scores


if __name__ == "__main__":
    import sys
    import os
    import glob
    
    print(f"📂 Directorio actual: {os.getcwd()}")
    
    # Si se pasa una imagen como argumento, usarla
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Buscar automáticamente imágenes disponibles
        print("\n🔍 Buscando imágenes disponibles...")
        
        # Posibles ubicaciones
        search_paths = [
            "CalcCampos/Imagenes_para_probar/*.jpg",
            "CalcCampos/Imagenes_para_probar/*.png",
            "Imagenes_para_probar/*.jpg",
            "Imagenes_para_probar/*.png",
            "*.jpg",
            "*.png"
        ]
        
        found_images = []
        for pattern in search_paths:
            found_images.extend(glob.glob(pattern))
        
        if found_images:
            print(f"\n✅ Imágenes encontradas:")
            for i, img in enumerate(found_images, 1):
                print(f"   {i}. {img}")
            
            # Usar la primera imagen encontrada
            image_path = found_images[0]
            print(f"\n📸 Usando: {image_path}")
        else:
            print(f"\n❌ No se encontraron imágenes")
            print(f"\n📂 Archivos en directorio actual:")
            for item in os.listdir('.'):
                print(f"   - {item}")
            
            # Buscar en CalcCampos
            if os.path.exists('CalcCampos'):
                print(f"\n📂 Contenido de CalcCampos/:")
                for item in os.listdir('CalcCampos'):
                    print(f"   - {item}")
            
            print(f"\n💡 Uso: python main.py <ruta_imagen>")
            sys.exit(1)
    
    # Verificar si el archivo existe
    if not os.path.exists(image_path):
        # Intentar con ruta absoluta
        abs_path = os.path.abspath(image_path)
        if os.path.exists(abs_path):
            image_path = abs_path
        else:
            print(f"\n❌ ERROR: No se encuentra la imagen en: {image_path}")
            print(f"📂 Ruta absoluta intentada: {abs_path}")
            print(f"\n💡 Verifica que el archivo exista")
            sys.exit(1)
    
    print(f"\n✅ Imagen encontrada: {image_path}\n")
    
    # Ejecutar
    scores = score(image_path)
    
    # Los scores pueden sumarse a los existentes del main
    # scores_main[1] += scores[1]
    # scores_main[2] += scores[2]
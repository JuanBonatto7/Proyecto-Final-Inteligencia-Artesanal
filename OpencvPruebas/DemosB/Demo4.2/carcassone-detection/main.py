"""
Punto de entrada principal del sistema de detección de Carcassonne.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from pipeline.pipeline import Pipeline
from steps import (
    ResizeStep,
    BlurStep,
    CannyEdgeDetectorStep,
    DilateStep,
    HoughLineTransformStep,
    FindIntersectionsStep,
    RANSACHomographyStep,
    FindTilesStep,
    TileClassifierStep,
    ReconstructBoardStep
)
from config import Config
from utils.image_utils import load_image


def create_detection_pipeline() -> Pipeline:
    """
    Crea y configura el pipeline completo de detección.
    
    Returns:
        Pipeline configurado con todos los pasos
    """
    pipeline = Pipeline("Carcassonne Board Detector")
    
    # Agregar todos los pasos en orden
    pipeline.add_step(ResizeStep())
    pipeline.add_step(BlurStep())
    pipeline.add_step(CannyEdgeDetectorStep())
    pipeline.add_step(DilateStep())
    pipeline.add_step(HoughLineTransformStep())
    pipeline.add_step(FindIntersectionsStep())
    pipeline.add_step(RANSACHomographyStep())
    pipeline.add_step(FindTilesStep())
    pipeline.add_step(TileClassifierStep())
    pipeline.add_step(ReconstructBoardStep())
    
    return pipeline


def process_image(image_path: str, visualize: bool = True) -> dict:
    """
    Procesa una imagen del tablero de Carcassonne.
    
    Args:
        image_path: Ruta a la imagen del tablero
        visualize: Si True, muestra visualizaciones de cada paso
        
    Returns:
        Diccionario con resultados del procesamiento
    """
    # Cargar imagen
    print(f"\n📷 Cargando imagen: {image_path}")
    image = load_image(image_path)
    
    if image is None:
        print("❌ Error: No se pudo cargar la imagen")
        return None
    
    print(f"✓ Imagen cargada: {image.shape[1]}x{image.shape[0]} píxeles")
    
    # Crear pipeline
    pipeline = create_detection_pipeline()
    
    # Preparar inputs
    inputs = {
        'img': image,
        'camera_points': None  # Opcional: puntos de calibración de cámara
    }
    
    # Ejecutar pipeline
    results = pipeline.run(inputs, visualize=visualize)
    
    return results


def save_results(results: dict, output_dir: str = "output"):
    """
    Guarda los resultados del procesamiento.
    
    Args:
        results: Resultados del pipeline
        output_dir: Directorio de salida
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n💾 Guardando resultados en: {output_dir}")
    
    # Guardar matriz del tablero
    if 'board_matrix' in results:
        matrix_file = output_path / "board_matrix.npy"
        np.save(matrix_file, results['board_matrix'])
        print(f"  ✓ Matriz guardada: {matrix_file}")
    
    # Guardar visualización del tablero
    if 'debug_image' in results:
        vis_file = output_path / "board_visualization.png"
        cv2.imwrite(str(vis_file), results['debug_image'])
        print(f"  ✓ Visualización guardada: {vis_file}")
    
    # Guardar imagen con perspectiva corregida
    if 'img_warped' in results:
        warped_file = output_path / "board_warped.png"
        cv2.imwrite(str(warped_file), results['img_warped'])
        print(f"  ✓ Imagen corregida guardada: {warped_file}")
    
    # Guardar datos de fichas en texto
    if 'board' in results:
        board = results['board']
        matrix = board.get_matrix_representation()
        
        text_file = output_path / "board_state.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("  ESTADO DEL TABLERO DE CARCASSONNE\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Dimensiones: {board.rows}x{board.cols}\n")
            f.write(f"Fichas detectadas: {len(board.tiles)}\n\n")
            
            f.write("Matriz del tablero:\n")
            f.write("-"*60 + "\n")
            
            for row in matrix:
                row_str = " | ".join([str(cell) if cell else "  ---  " for cell in row])
                f.write(row_str + "\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Detalles de fichas:\n")
            f.write("="*60 + "\n\n")
            
            for tile in board.tiles:
                f.write(f"Posición ({tile.position[0]}, {tile.position[1]}): "
                       f"{tile.tile_type} - Rotación {tile.rotation} - "
                       f"Confianza {tile.confidence:.2f}\n")
        
        print(f"  ✓ Estado del tablero guardado: {text_file}")


def main():
    """Función principal."""
    print("\n" + "="*60)
    print("  🎮 SISTEMA DE DETECCIÓN DE TABLERO DE CARCASSONNE")
    print("="*60)
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("\n❌ Error: Debe proporcionar la ruta a una imagen")
        print("\nUso:")
        print(f"  python {sys.argv[0]} <ruta_imagen> [--no-visualize]")
        print("\nEjemplo:")
        print(f"  python {sys.argv[0]} tablero.jpg")
        print(f"  python {sys.argv[0]} tablero.jpg --no-visualize")
        sys.exit(1)
    
    image_path = sys.argv[1]
    visualize = '--no-visualize' not in sys.argv
    
    # Verificar que el archivo existe
    if not Path(image_path).exists():
        print(f"\n❌ Error: El archivo no existe: {image_path}")
        sys.exit(1)
    
    try:
        # Procesar imagen
        results = process_image(image_path, visualize=visualize)
        
        if results is None:
            print("\n❌ Error durante el procesamiento")
            sys.exit(1)
        
        # Guardar resultados
        save_results(results)
        
        # Mostrar matriz final en consola
        if 'board_matrix' in results:
            print("\n📊 Matriz del tablero:")
            print("-"*60)
            matrix = results['board_matrix']
            for row in matrix:
                row_str = " | ".join([str(cell) if cell else "  ---  " for cell in row])
                print(row_str)
            print("-"*60)
        
        print("\n✅ Procesamiento completado exitosamente!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Script para detectar y enmascarar meeples (fichas azules y negras) en las losetas
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def detect_and_mask_meeples(image_path, output_path=None, show_debug=False):
    """
    Detecta meeples (fichas azules y negras) y los enmascara
    
    Args:
        image_path: Ruta a la imagen de entrada
        output_path: Ruta donde guardar la imagen procesada (opcional)
        show_debug: Si True, muestra visualización del proceso
    
    Returns:
        Imagen con meeples enmascarados
    """
    # Leer imagen
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Máscaras para detectar meeples azules y negros
    # Azul: Rango amplio para capturar diferentes tonos de azul
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Negro: Detectar valores muy oscuros
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combinar máscaras
    mask_meeples = cv2.bitwise_or(mask_blue, mask_black)
    
    # Dilatar para capturar toda la región del meeple
    kernel = np.ones((15, 15), np.uint8)
    mask_meeples = cv2.dilate(mask_meeples, kernel, iterations=2)
    
    # Suavizar bordes de la máscara
    mask_meeples = cv2.GaussianBlur(mask_meeples, (15, 15), 0)
    
    # Usar inpainting para rellenar las regiones con el contexto circundante
    img_no_meeples = cv2.inpaint(img, mask_meeples, 3, cv2.INPAINT_TELEA)
    img_no_meeples_rgb = cv2.cvtColor(img_no_meeples, cv2.COLOR_BGR2RGB)
    
    # Guardar si se especifica ruta
    if output_path:
        cv2.imwrite(str(output_path), img_no_meeples)
    
    # Visualización de debug
    if show_debug:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_blue, cmap='gray')
        axes[0, 1].set_title('Máscara Azul')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask_black, cmap='gray')
        axes[0, 2].set_title('Máscara Negro')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(mask_meeples, cmap='gray')
        axes[1, 0].set_title('Máscara Combinada')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_no_meeples_rgb)
        axes[1, 1].set_title('Sin Meeples (Inpainting)')
        axes[1, 1].axis('off')
        
        # Comparación lado a lado
        comparison = np.hstack([img_rgb, img_no_meeples_rgb])
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('Antes | Después')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return img_no_meeples_rgb


def batch_remove_meeples(input_dir, output_dir, show_samples=5):
    """
    Procesa todas las imágenes de un directorio removiendo meeples
    
    Args:
        input_dir: Directorio con imágenes originales
        output_dir: Directorio donde guardar imágenes procesadas
        show_samples: Número de muestras a mostrar para verificación
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Obtener todas las imágenes
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO IMÁGENES - REMOVIENDO MEEPLES")
    print(f"{'='*70}")
    print(f"📁 Entrada: {input_dir}")
    print(f"📁 Salida: {output_dir}")
    print(f"🖼️  Total de imágenes: {len(image_files)}\n")
    
    # Procesar cada imagen
    sample_count = 0
    for i, img_file in enumerate(image_files, 1):
        output_file = output_path / img_file.name
        
        # Mostrar debug para las primeras muestras
        show_debug = sample_count < show_samples
        
        try:
            detect_and_mask_meeples(
                img_file, 
                output_file,
                show_debug=show_debug
            )
            
            if show_debug:
                sample_count += 1
            
            print(f"  [{i:3d}/{len(image_files)}] ✓ {img_file.name}")
            
        except Exception as e:
            print(f"  [{i:3d}/{len(image_files)}] ✗ {img_file.name} - Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ PROCESAMIENTO COMPLETADO")
    print(f"📊 {len(list(output_path.glob('*')))} imágenes guardadas en {output_dir}")
    print(f"{'='*70}\n")


def test_single_image(image_path):
    """
    Prueba el procesamiento en una sola imagen con visualización
    
    Args:
        image_path: Ruta a la imagen de prueba
    """
    print(f"\n🔍 Probando con: {image_path}\n")
    detect_and_mask_meeples(image_path, show_debug=True)


def adjust_meeple_detection(image_path, blue_range=None, black_threshold=None):
    """
    Permite ajustar los rangos de detección de meeples interactivamente
    
    Args:
        image_path: Imagen de prueba
        blue_range: Tupla ((h_min, s_min, v_min), (h_max, s_max, v_max)) para azul
        black_threshold: Valor máximo de V para considerar negro
    """
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Usar rangos por defecto o los proporcionados
    if blue_range:
        lower_blue, upper_blue = blue_range
    else:
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
    
    if black_threshold:
        upper_black = np.array([180, 255, black_threshold])
    else:
        upper_black = np.array([180, 255, 50])
    
    lower_black = np.array([0, 0, 0])
    
    # Crear máscaras
    mask_blue = cv2.inRange(hsv, np.array(lower_blue), np.array(upper_blue))
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_combined = cv2.bitwise_or(mask_blue, mask_black)
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_blue, cmap='gray')
    axes[0, 1].set_title(f'Azul: HSV {lower_blue} - {upper_blue}')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask_black, cmap='gray')
    axes[1, 0].set_title(f'Negro: V < {upper_black[2]}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_combined, cmap='gray')
    axes[1, 1].set_title('Máscara Combinada')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Ajusta los rangos si no detecta bien los meeples:")
    print(f"   - Azul actual: {lower_blue} - {upper_blue}")
    print(f"   - Negro actual: V < {upper_black[2]}")
    print("\n   Ejemplo de uso:")
    print("   adjust_meeple_detection('foto.jpg', blue_range=([80, 40, 40], [140, 255, 255]))")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python mask_meeples.py <comando> [argumentos]")
        print("\nComandos:")
        print("  test <imagen.jpg>                    - Prueba con una imagen")
        print("  process <dir_entrada> <dir_salida>   - Procesa todas las imágenes")
        print("  adjust <imagen.jpg>                  - Ajustar detección de meeples")
        print("\nEjemplos:")
        print("  python mask_meeples.py test dataset/unlabeled/foto1.jpg")
        print("  python mask_meeples.py process dataset/unlabeled dataset/unlabeled_clean")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "test" and len(sys.argv) == 3:
        test_single_image(sys.argv[2])
    
    elif command == "process" and len(sys.argv) == 4:
        batch_remove_meeples(sys.argv[2], sys.argv[3])
    
    elif command == "adjust" and len(sys.argv) == 3:
        adjust_meeple_detection(sys.argv[2])
    
    else:
        print("❌ Comando inválido o argumentos faltantes")
        sys.exit(1)
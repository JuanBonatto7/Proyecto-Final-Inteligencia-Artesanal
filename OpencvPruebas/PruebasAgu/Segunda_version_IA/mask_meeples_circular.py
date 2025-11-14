"""
Script mejorado para detectar SOLO fichas redondas (meeples circulares)
Ignora los meeples pequeños de las ciudades
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def detect_circular_meeples(image_path, output_path=None, show_debug=False,
                           min_radius=20, max_radius=100,
                           blue_threshold=0.3, black_threshold=0.15):
    """
    Detecta SOLO fichas redondas (círculos) azules y negras
    
    Args:
        image_path: Ruta a la imagen
        output_path: Ruta de salida (opcional)
        show_debug: Mostrar visualización
        min_radius: Radio mínimo del círculo (en píxeles)
        max_radius: Radio máximo del círculo (en píxeles)
        blue_threshold: Umbral de proporción de azul para considerar el círculo como azul
        black_threshold: Umbral de brillo promedio para considerar el círculo como negro
    
    Returns:
        Imagen con meeples enmascarados
    """
    # Leer imagen
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    height, width = img.shape[:2]
    
    # Ajustar radios según tamaño de imagen
    img_diagonal = np.sqrt(height**2 + width**2)
    min_r = int(img_diagonal * 0.08)  # ~8% del diagonal
    max_r = int(img_diagonal * 0.35)  # ~35% del diagonal
    
    # Detectar círculos usando Hough Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_r,  # Distancia mínima entre círculos
        param1=50,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r
    )
    
    # Crear máscara vacía
    mask_meeples = np.zeros(gray.shape, dtype=np.uint8)
    circles_info = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Máscara para detectar azules
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue_color = cv2.inRange(hsv, lower_blue, upper_blue)
        
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Crear máscara para este círculo
            circle_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # Obtener píxeles dentro del círculo
            circle_pixels = img[circle_mask == 255]
            
            if len(circle_pixels) == 0:
                continue
            
            # Calcular proporción de píxeles azules en el círculo
            circle_blue_pixels = mask_blue_color[circle_mask == 255]
            blue_ratio = np.sum(circle_blue_pixels == 255) / len(circle_blue_pixels)
            
            # Calcular brillo promedio
            gray_pixels = gray[circle_mask == 255]
            avg_brightness = np.mean(gray_pixels) / 255.0
            
            # Determinar si es un meeple (azul o negro)
            is_blue_meeple = blue_ratio > blue_threshold
            is_black_meeple = avg_brightness < black_threshold
            
            if is_blue_meeple or is_black_meeple:
                # Agregar a la máscara con un margen extra
                margin = int(r * 0.2)  # 20% más grande
                cv2.circle(mask_meeples, (x, y), r + margin, 255, -1)
                
                meeple_type = "AZUL" if is_blue_meeple else "NEGRO"
                circles_info.append({
                    'x': x, 'y': y, 'r': r,
                    'type': meeple_type,
                    'blue_ratio': blue_ratio,
                    'brightness': avg_brightness
                })
    
    # Suavizar bordes de la máscara
    if np.any(mask_meeples):
        mask_meeples = cv2.GaussianBlur(mask_meeples, (21, 21), 0)
        
        # Usar inpainting para rellenar
        img_no_meeples = cv2.inpaint(img, mask_meeples, 5, cv2.INPAINT_TELEA)
    else:
        img_no_meeples = img.copy()
    
    img_no_meeples_rgb = cv2.cvtColor(img_no_meeples, cv2.COLOR_BGR2RGB)
    
    # Guardar si se especifica
    if output_path:
        cv2.imwrite(str(output_path), img_no_meeples)
    
    # Visualización de debug
    if show_debug:
        fig = plt.figure(figsize=(16, 10))
        
        # Imagen original con círculos detectados
        ax1 = plt.subplot(2, 3, 1)
        img_with_circles = img_rgb.copy()
        if circles is not None:
            for info in circles_info:
                color = (0, 0, 255) if info['type'] == "AZUL" else (0, 0, 0)
                cv2.circle(img_with_circles, (info['x'], info['y']), info['r'], color, 3)
                cv2.circle(img_with_circles, (info['x'], info['y']), 2, color, -1)
        ax1.imshow(img_with_circles)
        ax1.set_title(f'Círculos Detectados: {len(circles_info)}', fontsize=12)
        ax1.axis('off')
        
        # Máscara de azul
        ax2 = plt.subplot(2, 3, 2)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue_viz = cv2.inRange(hsv, lower_blue, upper_blue)
        ax2.imshow(mask_blue_viz, cmap='gray')
        ax2.set_title('Detección de Color Azul', fontsize=12)
        ax2.axis('off')
        
        # Máscara final de meeples
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(mask_meeples, cmap='gray')
        ax3.set_title('Máscara Final de Meeples', fontsize=12)
        ax3.axis('off')
        
        # Resultado
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(img_no_meeples_rgb)
        ax4.set_title('Sin Meeples (Inpainting)', fontsize=12)
        ax4.axis('off')
        
        # Comparación
        ax5 = plt.subplot(2, 3, 5)
        comparison = np.hstack([img_rgb, img_no_meeples_rgb])
        ax5.imshow(comparison)
        ax5.set_title('Antes | Después', fontsize=12)
        ax5.axis('off')
        
        # Info de meeples detectados
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        info_text = "MEEPLES DETECTADOS:\n" + "="*40 + "\n"
        if circles_info:
            for i, info in enumerate(circles_info, 1):
                info_text += f"\nMeeple {i} ({info['type']}):\n"
                info_text += f"  Posición: ({info['x']}, {info['y']})\n"
                info_text += f"  Radio: {info['r']} px\n"
                info_text += f"  Azul: {info['blue_ratio']:.1%}\n"
                info_text += f"  Brillo: {info['brightness']:.1%}\n"
        else:
            info_text += "\nNo se detectaron meeples circulares"
        
        ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    print(f"  Detectados: {len(circles_info)} meeples circulares")
    for info in circles_info:
        print(f"    → {info['type']} en ({info['x']}, {info['y']}) radio={info['r']}px")
    
    return img_no_meeples_rgb


def batch_remove_circular_meeples(input_dir, output_dir, show_samples=3):
    """
    Procesa todas las imágenes removiendo solo fichas circulares
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO IMÁGENES - REMOVIENDO FICHAS CIRCULARES")
    print(f"{'='*70}")
    print(f"📁 Entrada: {input_dir}")
    print(f"📁 Salida: {output_dir}")
    print(f"🖼️  Total: {len(image_files)}\n")
    
    sample_count = 0
    total_meeples = 0
    
    for i, img_file in enumerate(image_files, 1):
        output_file = output_path / img_file.name
        show_debug = sample_count < show_samples
        
        try:
            print(f"[{i:3d}/{len(image_files)}] {img_file.name}")
            detect_circular_meeples(
                img_file,
                output_file,
                show_debug=show_debug
            )
            
            if show_debug:
                sample_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETADO")
    print(f"📊 {len(list(output_path.glob('*')))} imágenes en {output_dir}")
    print(f"{'='*70}\n")


def test_single_image(image_path):
    """Prueba con una sola imagen"""
    print(f"\n🔍 Analizando: {image_path}\n")
    detect_circular_meeples(image_path, show_debug=True)


def adjust_detection(image_path):
    """
    Ajusta parámetros de detección interactivamente
    """
    print("\n" + "="*70)
    print("🔧 AJUSTE DE PARÁMETROS")
    print("="*70)
    
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    img_diagonal = np.sqrt(height**2 + width**2)
    
    print(f"\n📐 Dimensiones de imagen: {width}x{height}")
    print(f"   Diagonal: {img_diagonal:.0f} píxeles")
    print(f"\n🎯 Radios automáticos:")
    print(f"   Mínimo: {int(img_diagonal * 0.08)} px (8% diagonal)")
    print(f"   Máximo: {int(img_diagonal * 0.35)} px (35% diagonal)")
    
    print("\n" + "-"*70)
    print("Parámetros actuales:")
    print("-"*70)
    print("blue_threshold = 0.3   # 30% del círculo debe ser azul")
    print("black_threshold = 0.15  # Brillo < 15% para negro")
    
    print("\n💡 Para ajustar:")
    print("   - Si detecta muchos falsos positivos → aumenta blue_threshold")
    print("   - Si no detecta algunos meeples → baja blue_threshold")
    print("   - Si no detecta negros → aumenta black_threshold")
    
    print("\n🧪 Probando con diferentes umbrales...")
    
    for blue_th in [0.2, 0.3, 0.4]:
        print(f"\n{'='*70}")
        print(f"Probando con blue_threshold = {blue_th}")
        print('='*70)
        detect_circular_meeples(image_path, show_debug=False,
                               blue_threshold=blue_th, black_threshold=0.15)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUso:")
        print("  python mask_meeples_circular.py <comando> [argumentos]")
        print("\nComandos:")
        print("  test <imagen.jpg>                    - Prueba con una imagen")
        print("  process <dir_entrada> <dir_salida>   - Procesa todas las imágenes")
        print("  adjust <imagen.jpg>                  - Ajustar parámetros")
        print("\nEjemplos:")
        print("  python mask_meeples_circular.py test dataset/unlabeled/foto1.jpg")
        print("  python mask_meeples_circular.py process dataset/unlabeled dataset/unlabeled_clean")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "test" and len(sys.argv) == 3:
        test_single_image(sys.argv[2])
    
    elif command == "process" and len(sys.argv) == 4:
        batch_remove_circular_meeples(sys.argv[2], sys.argv[3])
    
    elif command == "adjust" and len(sys.argv) == 3:
        adjust_detection(sys.argv[2])
    
    else:
        print("❌ Comando inválido o argumentos faltantes")
        sys.exit(1)
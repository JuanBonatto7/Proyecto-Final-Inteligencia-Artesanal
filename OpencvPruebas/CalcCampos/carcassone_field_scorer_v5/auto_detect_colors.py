"""
Detecta los colores únicos en la imagen.
"""
import sys
import cv2
import numpy as np


def detect_colors(image_path):
    """Detecta los colores más comunes en la imagen."""

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar {image_path} / Could not load {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Imagen cargada: {img.shape} / Image loaded: {img.shape}")
    print(f"Total de pixeles: {img.shape[0] * img.shape[1]} / Total pixels: {img.shape[0] * img.shape[1]}")

    # Reshape para obtener lista de colores
    pixels = img.reshape(-1, 3)

    # Contar colores únicos
    unique_colors = {}
    for pixel in pixels:
        color_tuple = tuple(pixel)
        unique_colors[color_tuple] = unique_colors.get(color_tuple, 0) + 1

    print(f"\nColores únicos encontrados: {len(unique_colors)} / Unique colors found: {len(unique_colors)}")

    # Ordenar por frecuencia
    sorted_colors = sorted(unique_colors.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("COLORES MÁS FRECUENTES / MOST FREQUENT COLORS:")
    print("=" * 70)
    print(f"{'#':<4} {'RGB':<20} {'Pixeles / Pixels':<20} {'Porcentaje / Percentage':<20} {'Posible / Possible'}")
    print("-" * 70)

    COLOR_NAMES = ['FIELD?', 'CASTLE?', 'ROAD?', 'CHURCH?', 'MEEPLE_1?', 'MEEPLE_2?', 
                   'Otro / Other', 'Otro / Other', 'Otro / Other', 'Otro / Other']

    for i, (color, count) in enumerate(sorted_colors[:10]):
        percentage = count / len(pixels) * 100
        print(f"{i+1:<4} RGB{str(color):<15} {count:<20} {percentage:>6.2f}%      {COLOR_NAMES[i]}")

    # Crear visualización
    print("\n" + "=" * 70)
    print("SUGERENCIA DE CONFIGURACIÓN / CONFIGURATION SUGGESTION:")
    print("=" * 70)
    print("\nCopia esto en config/colors.py / Copy this into config/colors.py:\n")
    print("COLORS = {")

    # Intentar adivinar qué color es cada uno
    suggestions = {
        'FIELD': None,      # Probablemente verde / Probably green
        'CASTLE': None,     # Probablemente naranja / Probably orange
        'ROAD': None,       # Probablemente azul / Probably blue
        'CHURCH': None,     # Probablemente rojo / Probably red
        'MEEPLE_1': None,   # Probablemente violeta / Probably violet
        'MEEPLE_2': None,   # Probablemente negro / Probably black
    }

    for i, (color, count) in enumerate(sorted_colors[:10]):
        r, g, b = color

        # Ignorar blanco y negro muy común (probablemente fondo) / Ignore common white and black (probably background)
        if (r == g == b) and (r > 240 or r < 15):
            continue

        # Detectar verde (FIELD) / Detect green (FIELD)
        if g > r and g > b and suggestions['FIELD'] is None:
            suggestions['FIELD'] = color
        # Detectar azul (ROAD) / Detect blue (ROAD)
        elif b > r and b > g and suggestions['ROAD'] is None:
            suggestions['ROAD'] = color
        # Detectar rojo (CHURCH) / Detect red (CHURCH)
        elif r > g and r > b and r > 200 and suggestions['CHURCH'] is None:
            suggestions['CHURCH'] = color
        # Detectar naranja (CASTLE) / Detect orange (CASTLE)
        elif r > b and g > b and abs(r - g) < 100 and suggestions['CASTLE'] is None:
            suggestions['CASTLE'] = color
        # Detectar violeta (MEEPLE_1) / Detect violet (MEEPLE_1)
        elif r > g and b > g and suggestions['MEEPLE_1'] is None:
            suggestions['MEEPLE_1'] = color
        # Detectar negro/gris oscuro (MEEPLE_2) / Detect black/dark gray (MEEPLE_2)
        elif r < 50 and g < 50 and b < 50 and suggestions['MEEPLE_2'] is None:
            suggestions['MEEPLE_2'] = color

    for name, color in suggestions.items():
        if color:
            print(f"    '{name}': {color},")
        else:
            print(f"    '{name}': (0, 0, 0),  # NO DETECTADO - ajustar manualmente / NOT DETECTED - adjust manually")

    print("}\n")

    # Mostrar muestra visual de colores
    num_colors = min(10, len(sorted_colors))
    sample_height = 80
    sample_width = 600
    sample = np.zeros((sample_height * num_colors, sample_width, 3), dtype=np.uint8)

    for i, (color, count) in enumerate(sorted_colors[:num_colors]):
        y_start = i * sample_height
        y_end = (i + 1) * sample_height
        sample[y_start:y_end, :] = color

        # Añadir texto / Add text
        text = f"{i+1}. RGB{color} ({count/len(pixels)*100:.1f}%)"
        cv2.putText(
            sample, text, (10, y_start + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            (255-color[0], 255-color[1], 255-color[2]), 2
        )

    # Mostrar imagen original y colores detectados / Show original image and detected colors
    cv2.imshow('Imagen Original / Original Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow('Top 10 Colores Detectados / Top 10 Detected Colors', cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

    print("Presiona cualquier tecla en las ventanas para cerrar... / Press any key in the windows to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python auto_detect_colors.py <ruta_imagen> / Usage: python auto_detect_colors.py <image_path>")
        sys.exit(1)

    detect_colors(sys.argv[1])
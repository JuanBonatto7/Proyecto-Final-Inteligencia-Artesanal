"""
Pipeline para detectar tipo y rotación de losetas de Carcassonne.
Ejecuta tile_detector.py seguido de rotation_detector.py
"""

import subprocess
import sys
import os


def run_pipeline(image_path: str) -> tuple[str, int]:
    """Ejecuta la pipeline completa: detección de tipo + rotación"""

    if not os.path.exists(image_path):
        raise ValueError(f"No existe el archivo {image_path}")

    # Paso 1: Detectar tipo de loseta
    print("=== PASO 1: DETECCIÓN DE TIPO ===")
    try:
        result_type = subprocess.run([
            sys.executable, "tile_detector.py", image_path
        ], capture_output=True, text=True, timeout=60)

        if result_type.returncode != 0:
            print(f"Error en detección de tipo: {result_type.stderr}")
            raise RuntimeError("Falló la detección de tipo")

        # Extraer tipo del output
        output_lines = result_type.stdout.strip().split('\n')
        tile_type = None
        for line in output_lines:
            if "Resultado: Loseta tipo" in line:
                parts = line.split()
                tile_type = parts[3]  # "Resultado: Loseta tipo A"
                break

        if not tile_type:
            print("No se pudo extraer el tipo de loseta")
            raise RuntimeError("No se detectó tipo")

        print(f"Tipo detectado: {tile_type}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout en detección de tipo")

    # Paso 2: Detectar rotación
    print("\n=== PASO 2: DETECCIÓN DE ROTACIÓN ===")
    try:
        result_rotation = subprocess.run([
            sys.executable, "rotation_detector.py", tile_type, image_path
        ], capture_output=True, text=True, timeout=60)

        if result_rotation.returncode != 0:
            print(f"Error en detección de rotación: {result_rotation.stderr}")
            raise RuntimeError("Falló la detección de rotación")

        # Extraer rotación del output
        output_lines = result_rotation.stdout.strip().split('\n')
        rotation = None
        for line in output_lines:
            if "Resultado: Rotación" in line:
                parts = line.split()
                rot_str = parts[2].strip('°')  # "Resultado: Rotación 90°"
                try:
                    rotation = int(rot_str)
                except ValueError:
                    rotation = None
                break

        if rotation is None:
            print("No se pudo extraer la rotación")
            raise RuntimeError("No se detectó rotación")

        print(f"Rotación detectada: {rotation}°")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout en detección de rotación")

    return tile_type, rotation


def main():
    if len(sys.argv) < 2:
        print("Uso: python pipeline.py <ruta_imagen_loseta>")
        print("Ejemplo: python pipeline.py loseta.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        tile_type, rotation = run_pipeline(image_path)
        print(f"\n{'='*50}")
        print("RESULTADO FINAL:")
        print(f"{'='*50}")
        print(f"Loseta tipo: {tile_type}")
        print(f"Rotación: {rotation}°")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
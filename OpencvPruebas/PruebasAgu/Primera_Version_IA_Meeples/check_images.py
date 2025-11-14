#!/usr/bin/env python3
"""
Script de ejemplo para verificar que las imágenes están colocadas correctamente
"""

from pathlib import Path

def check_images():
    """Verifica que las imágenes estén colocadas en los directorios correctos"""

    print("🔍 VERIFICANDO IMÁGENES DEL PROYECTO")
    print("=" * 50)

    # Verificar imágenes reales del usuario
    real_dir = Path("real_test_images")
    if real_dir.exists():
        real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) + list(real_dir.glob("*.png"))
        print(f"📸 Imágenes reales en real_test_images/: {len(real_images)}")
        if real_images:
            print("   ✅ Encontradas:")
            for img in real_images[:5]:  # Mostrar primeras 5
                print(f"      - {img.name}")
            if len(real_images) > 5:
                print(f"      ... y {len(real_images) - 5} más")
        else:
            print("   ⚠️  No hay imágenes. Coloca tus fotos de Carcassonne aquí.")
    else:
        print("❌ Directorio 'real_test_images' no existe")

    print()

    # Verificar imágenes de losetas base
    tiles_dir = Path("tiles")
    if tiles_dir.exists():
        tiles_images = list(tiles_dir.glob("*.jpg")) + list(tiles_dir.glob("*.jpeg")) + list(tiles_dir.glob("*.png"))
        print(f"🏰 Imágenes de losetas en tiles/: {len(tiles_images)}")
    else:
        print("❌ Directorio 'tiles' no existe")

    print()

    # Verificar imágenes de prueba simuladas
    test_dir = Path("test_images")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
        print(f"🎨 Imágenes de prueba simuladas en test_images/: {len(test_images)}")
    else:
        print("❌ Directorio 'test_images' no existe")

    print()
    print("💡 INSTRUCCIONES:")
    print("   1. Coloca tus fotos reales de Carcassonne en 'real_test_images/'")
    print("   2. Ejecuta: python test_real_images.py")
    print("   3. Si no funciona bien, ajusta parámetros con: python tune_params.py")

if __name__ == "__main__":
    check_images()
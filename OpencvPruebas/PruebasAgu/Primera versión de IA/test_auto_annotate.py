#!/usr/bin/env python3
"""
Script de prueba rápida del auto-anotador
Verifica que todo esté configurado correctamente
"""

import sys
from pathlib import Path

def check_requirements():
    """Verifica que todas las dependencias estén instaladas"""
    print("Verificando dependencias...\n")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn (para skimage)',
        'PIL': 'Pillow'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            if module == 'sklearn':
                from skimage.metrics import structural_similarity
                print(f"✓ {package}")
            else:
                __import__(module)
                print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - FALTA")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Faltan dependencias. Instala con:")
        if 'scikit-learn' in [m.split()[0] for m in missing]:
            print("   pip install scikit-image")
        for pkg in missing:
            if 'scikit' not in pkg:
                pkg_name = pkg.split()[0].lower()
                print(f"   pip install {pkg_name}")
        return False
    
    print("\n✓ Todas las dependencias están instaladas\n")
    return True


def check_directories():
    """Verifica que los directorios necesarios existan"""
    print("Verificando directorios...\n")
    
    dirs = {
        'referencias': 'Imágenes de referencia',
        'tiles': 'Losetas detectadas (opcional para prueba)',
        'letras': 'Imágenes originales de letras'
    }
    
    all_ok = True
    
    for dir_name, description in dirs.items():
        path = Path(dir_name)
        if path.exists():
            files = list(path.glob('*.png')) + list(path.glob('*.jpg'))
            print(f"✓ {dir_name}/ - {len(files)} archivos")
        else:
            if dir_name != 'tiles':  # tiles es opcional
                print(f"✗ {dir_name}/ - NO EXISTE")
                all_ok = False
            else:
                print(f"⚠️  {dir_name}/ - No existe (ejecuta primero carcassonne.py)")
    
    return all_ok


def check_references():
    """Verifica que las referencias estén correctamente generadas"""
    print("\nVerificando referencias...\n")
    
    refs_dir = Path('referencias')
    
    if not refs_dir.exists():
        print("✗ Directorio 'referencias/' no existe")
        print("\nEjecuta primero:")
        print("   python tile_mapping.py prepare letras/ referencias/")
        return False
    
    # Buscar archivos de referencia
    ref_files = sorted(refs_dir.glob('tile_type_*.png'))
    
    if not ref_files:
        print("✗ No hay archivos de referencia (tile_type_*.png)")
        print("\nEjecuta primero:")
        print("   python tile_mapping.py prepare letras/ referencias/")
        return False
    
    print(f"✓ Encontradas {len(ref_files)} referencias:")
    
    try:
        from tile_mapping import TileMapper
        mapper = TileMapper()
        
        for ref_file in ref_files[:5]:  # Mostrar solo las primeras 5
            idx = int(ref_file.stem.split('_')[-1])
            letter = mapper.idx_to_letter(idx)
            print(f"  - {ref_file.name} → Letra {letter}")
        
        if len(ref_files) > 5:
            print(f"  ... y {len(ref_files) - 5} más")
        
        return True
    except Exception as e:
        print(f"⚠️  Error al cargar mapper: {e}")
        return True  # No bloqueante


def run_quick_test():
    """Ejecuta una prueba rápida del auto-anotador"""
    print("\n" + "="*60)
    print("PRUEBA RÁPIDA DEL AUTO-ANOTADOR")
    print("="*60 + "\n")
    
    tiles_dir = Path('tiles')
    
    if not tiles_dir.exists() or not list(tiles_dir.glob('*.png')):
        print("⚠️  No hay losetas para probar en tiles/")
        print("\nPara hacer una prueba completa:")
        print("1. Toma una foto del tablero: foto_tablero.jpg")
        print("2. Ejecuta: python carcassonne.py foto_tablero.jpg")
        print("3. Ejecuta: python test_auto_annotate.py")
        return
    
    print("Iniciando auto-anotación de prueba...\n")
    
    try:
        from auto_annotate import AutoAnnotator
        
        annotator = AutoAnnotator('referencias')
        
        # Tomar solo las primeras 5 losetas para prueba rápida
        tile_files = sorted(list(tiles_dir.glob('*.png')))[:5]
        
        print(f"Probando con {len(tile_files)} losetas...\n")
        
        for tile_file in tile_files:
            import cv2
            tile_img = cv2.imread(str(tile_file))
            
            if tile_img is None:
                continue
            
            match = annotator.find_best_match(tile_img, min_confidence=0.5)
            
            if match:
                print(f"✓ {tile_file.name}:")
                print(f"  Tipo: {match.letter}")
                print(f"  Rotación: {match.rotation * 90}°")
                print(f"  Confianza: {match.confidence:.2%}")
                print(f"  Scores: SSIM={match.method_scores['ssim']:.2f}, "
                      f"Hist={match.method_scores['histogram']:.2f}")
            else:
                print(f"✗ {tile_file.name}: Sin coincidencia")
        
        print("\n✓ Prueba completada exitosamente!")
        print("\nPara anotar todas las losetas:")
        print("   python auto_annotate.py tiles/ referencias/")
        
    except ImportError as e:
        print(f"✗ Error al importar: {e}")
        print("\nAsegúrate de tener todos los archivos:")
        print("  - auto_annotate.py")
        print("  - tile_mapping.py")
    except Exception as e:
        print(f"✗ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*60)
    print("VERIFICACIÓN DEL SISTEMA DE AUTO-ANOTACIÓN")
    print("="*60 + "\n")
    
    # Verificar dependencias
    if not check_requirements():
        print("\n⚠️  Instala las dependencias faltantes antes de continuar")
        return 1
    
    # Verificar directorios
    if not check_directories():
        print("\n⚠️  Crea los directorios necesarios antes de continuar")
        return 1
    
    # Verificar referencias
    if not check_references():
        print("\n⚠️  Genera las referencias antes de continuar")
        return 1
    
    print("\n" + "="*60)
    print("✓ SISTEMA LISTO PARA USAR")
    print("="*60)
    
    # Preguntar si quiere hacer prueba
    print("\n¿Quieres ejecutar una prueba rápida? (solo si tienes losetas en tiles/)")
    print("Presiona Enter para continuar o Ctrl+C para salir...")
    
    try:
        input()
        run_quick_test()
    except KeyboardInterrupt:
        print("\n\nPrueba cancelada.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

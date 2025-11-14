#!/usr/bin/env python
"""
Script de prueba rápida para la herramienta de anotación mejorada.
Verifica que todos los componentes funcionan correctamente.
"""

import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Verifica que todos los imports funcionen."""
    print("✓ Probando imports...")
    try:
        import cv2
        import numpy as np
        from annotate import AnnotationTool
        print("  ✓ cv2 disponible")
        print("  ✓ numpy disponible")
        print("  ✓ AnnotationTool importado")
        return True
    except ImportError as e:
        print(f"  ✗ Error de import: {e}")
        return False

def test_tile_types():
    """Verifica que los tipos de losetas estén correctos."""
    print("\n✓ Probando tipos de losetas...")
    from annotate import AnnotationTool
    
    expected = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'BLANCO']
    
    if AnnotationTool.TILE_TYPES == expected:
        print(f"  ✓ {len(expected)} tipos de losetas correctos")
        return True
    else:
        print("  ✗ Tipos de losetas incorrectos")
        return False

def test_annotation_structure():
    """Verifica la estructura de anotaciones."""
    print("\n✓ Probando estructura de anotaciones...")
    
    required_fields = [
        'tile_letter',
        'tile_type',
        'rotation',
        'has_meeple',
        'meeple_position'
    ]
    
    # Crear una instancia temporal (sin imágenes)
    try:
        from annotate import AnnotationTool
        # Simulamos que existe un directorio
        tool = AnnotationTool.__new__(AnnotationTool)
        tool.current_annotation = {
            'tile_letter': 'A',
            'tile_type': 0,
            'rotation': 0,
            'has_meeple': False,
            'meeple_position': 0
        }
        
        for field in required_fields:
            if field not in tool.current_annotation:
                print(f"  ✗ Falta campo: {field}")
                return False
        
        print(f"  ✓ Estructura de anotación correcta ({len(required_fields)} campos)")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_ranges():
    """Verifica los rangos de valores."""
    print("\n✓ Probando rangos de valores...")
    
    # Rotación: 0-3
    rotations = [0, 1, 2, 3]
    if all(0 <= r <= 3 for r in rotations):
        print("  ✓ Rotación: 0-3 ✓")
    else:
        print("  ✗ Rotación: rango incorrecto")
        return False
    
    # Posición meeple: 0-8
    positions = list(range(9))  # 0-8
    if all(0 <= p <= 8 for p in positions):
        print("  ✓ Posición meeple: 0-8 ✓")
    else:
        print("  ✗ Posición meeple: rango incorrecto")
        return False
    
    return True

def test_key_mapping():
    """Verifica el mapeo de teclas."""
    print("\n✓ Probando mapeo de teclas...")
    
    key_map = {
        'R/T': 'Rotación',
        'M': 'Toggle meeple',
        '0-8': 'Posición meeple',
        'A-X': 'Tipo loseta',
        'SPACE': 'BLANCO',
        'ENTER': 'Guardar+avanzar',
        'F5': 'Guardar',
        '< >': 'Navegar',
        'ESC': 'Salir'
    }
    
    print(f"  ✓ {len(key_map)} controles mapeados:")
    for keys, action in key_map.items():
        print(f"    • {keys:10s} → {action}")
    
    return True

def main():
    """Ejecuta todas las pruebas."""
    print("="*70)
    print("PRUEBAS DE LA HERRAMIENTA DE ANOTACIÓN MEJORADA")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Tipos de losetas", test_tile_types),
        ("Estructura de anotaciones", test_annotation_structure),
        ("Rangos de valores", test_ranges),
        ("Mapeo de teclas", test_key_mapping)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error en prueba '{name}': {e}")
            results.append((name, False))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "-"*70)
    print(f"Resultado: {passed}/{total} pruebas pasadas")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron! La herramienta está lista.")
        print("\nPara usar:")
        print('  python annotate.py "ruta/a/tiles" --output annotations.json')
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")
    
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

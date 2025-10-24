"""
Sistema de Detección de Tablero de Carcassonne
"""

import json
from config import *
from board_detector import BoardDetector


def print_matrix(matrix):
    """
    Imprime la matriz de forma legible
    """
    print("\n" + "="*60)
    print("MATRIZ DEL TABLERO")
    print("="*60)
    
    for i, row in enumerate(matrix):
        print(f"\nFila {i}:")
        for j, cell in enumerate(row):
            if cell:
                print(f"  [{j}] {cell['tile']} (rot: {cell['rotation']}, conf: {cell['confidence']})")
            else:
                print(f"  [{j}] VACÍO")
    
    print("\n" + "="*60 + "\n")


def save_matrix_json(matrix, filename="tablero_resultado.json"):
    """
    Guarda la matriz en formato JSON
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Matriz guardada en: {filename}")


def main():
    print("\n" + "="*60)
    print("SISTEMA DE DETECCIÓN - CARCASSONNE")
    print("="*60 + "\n")
    
    # Crear detector
    detector = BoardDetector()
    
    # Cargar imagen
    try:
        detector.load_image(INPUT_IMAGE)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Generar imagen de debug
    print("\n[DEBUG] Generando imagen de debug...")
    detector.generate_debug_image()
    print("[DEBUG] Verifica que el cuadrado de referencia esté alineado correctamente")
    print(f"[DEBUG] Revisa la imagen: {OUTPUT_DEBUG_IMAGE}\n")
    
    # Detectar tablero
    print("[INFO] Analizando tablero...\n")
    board_matrix = detector.create_board_matrix()
    
    # Mostrar resultado
    print_matrix(board_matrix)
    
    # Guardar resultado
    save_matrix_json(board_matrix)
    
    print("[INFO] Proceso completado!\n")


if __name__ == "__main__":
    main()
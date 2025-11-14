#!/usr/bin/env python3
"""
Demo script para el Detector de Meeples Azules/Negros
"""

from src.meeple_detector import MeepleDataset, create_data_transforms
from torch.utils.data import DataLoader
import json

def demo_dataset():
    """Demuestra cómo cargar el dataset"""
    print("=== DEMO: Carga de Dataset ===")

    # Crear transformaciones
    transform = create_data_transforms(augment=False)

    # Cargar dataset de entrenamiento
    try:
        dataset = MeepleDataset('data/train_annotations.json', transform=transform)
        print(f"✅ Dataset cargado exitosamente: {len(dataset)} muestras")

        # Mostrar algunas anotaciones
        print("\nPrimeras 3 anotaciones:")
        for i in range(min(3, len(dataset.annotations))):
            ann = dataset.annotations[i]
            print(f"  {ann.image_path}: meeple={'Sí' if ann.has_blue_or_black_meeple else 'No'}, pos={ann.meeple_position}")

    except FileNotFoundError:
        print("❌ No se encontraron archivos de anotaciones. Crea data/train_annotations.json primero.")

def demo_training():
    """Demuestra cómo sería el entrenamiento (sin ejecutarlo realmente)"""
    print("\n=== DEMO: Entrenamiento ===")
    print("Para entrenar el modelo, ejecuta:")
    print("  python src/train_meeple_detector.py")
    print("Esto entrenará el modelo con los datos en data/train_annotations.json y data/val_annotations.json")

def demo_prediction():
    """Demuestra cómo sería la predicción (sin ejecutarla realmente)"""
    print("\n=== DEMO: Predicción ===")
    print("Para hacer predicciones, ejecuta:")
    print("  python src/predict_meeple.py data/tiles/")
    print("O para una imagen específica:")
    print("  python src/predict_meeple.py data/tiles/A.png")

def show_grid():
    """Muestra la numeración de la cuadrícula 3x3"""
    print("\n=== CUADRÍCULA DE POSICIONES ===")
    print("La loseta se divide en 9 subespacios numerados así:")
    print("┌───┬───┬───┐")
    print("│ 0 │ 1 │ 2 │")
    print("├───┼───┼───┤")
    print("│ 3 │ 4 │ 5 │")
    print("├───┼───┼───┤")
    print("│ 6 │ 7 │ 8 │")
    print("└───┴───┴───┘")
    print("Posición -1 = No hay meeple")

if __name__ == "__main__":
    print("🚀 DEMO DEL SISTEMA DE DETECCIÓN DE MEEPLES AZULES/NEGROS")
    print("="*60)

    show_grid()
    demo_dataset()
    demo_training()
    demo_prediction()

    print("\n" + "="*60)
    print("✅ Demo completada. El sistema está listo para usar.")
    print("Recuerda: Para datos reales, necesitas imágenes de losetas con meeples azules/negros anotadas correctamente.")
"""
Script principal interactivo para el clasificador de Carcassonne
"""

import sys
from pathlib import Path

from config import DATASET_DIR, UNLABELED_DIR, MODELS_DIR
from data_utils import batch_generate_synthetic_data, print_dataset_info
from clustering import auto_organize_tiles, find_optimal_clusters
from train import train_model
from predict import load_model_and_mapping, predict_from_board


def print_menu():
    """Muestra el menú principal"""
    print("\n" + "="*70)
    print("CLASIFICADOR DE LOSETAS DE CARCASSONNE - Menú Principal")
    print("="*70)
    print("\n1. 🔍 Organizar losetas automáticamente (clustering)")
    print("2. 📈 Encontrar número óptimo de clusters")
    print("3. 🎨 Generar dataset sintético (data augmentation)")
    print("4. 📊 Mostrar información del dataset")
    print("5. 🚀 Entrenar modelo")
    print("6. 🎯 Realizar predicciones")
    print("7. ❌ Salir")
    print("\n" + "="*70)


def organize_tiles_menu():
    """Menú para organización automática"""
    print("\n" + "="*70)
    print("ORGANIZACIÓN AUTOMÁTICA DE LOSETAS")
    print("="*70)
    
    unlabeled = input(f"\nDirectorio con imágenes sin etiquetar [{UNLABELED_DIR}]: ").strip()
    if not unlabeled:
        unlabeled = UNLABELED_DIR
    
    output = input(f"Directorio de salida [dataset/clustered/]: ").strip()
    if not output:
        output = "dataset/clustered/"
    
    n_clusters = input("Número de clusters [24]: ").strip()
    n_clusters = int(n_clusters) if n_clusters else 24
    
    auto_organize_tiles(unlabeled, output, n_clusters, visualize=True)
    
    input("\nPresiona Enter para continuar...")


def find_clusters_menu():
    """Menú para encontrar número óptimo de clusters"""
    print("\n" + "="*70)
    print("BÚSQUEDA DE NÚMERO ÓPTIMO DE CLUSTERS")
    print("="*70)
    
    unlabeled = input(f"\nDirectorio con imágenes [{UNLABELED_DIR}]: ").strip()
    if not unlabeled:
        unlabeled = UNLABELED_DIR
    
    max_clusters = input("Número máximo de clusters a probar [30]: ").strip()
    max_clusters = int(max_clusters) if max_clusters else 30
    
    find_optimal_clusters(unlabeled, max_clusters)
    
    input("\nPresiona Enter para continuar...")


def generate_synthetic_menu():
    """Menú para generación de dataset sintético"""
    print("\n" + "="*70)
    print("GENERACIÓN DE DATASET SINTÉTICO")
    print("="*70)
    
    input_dir = input(f"\nDirectorio con imágenes originales [{DATASET_DIR}]: ").strip()
    if not input_dir:
        input_dir = DATASET_DIR
    
    output_dir = input("Directorio de salida [dataset/augmented/]: ").strip()
    if not output_dir:
        output_dir = "dataset/augmented/"
    
    num_variations = input("Variaciones por imagen [50]: ").strip()
    num_variations = int(num_variations) if num_variations else 50
    
    print(f"\n⚠️  Esto generará aproximadamente {num_variations} variaciones por imagen.")
    confirm = input("¿Continuar? (s/n): ").strip().lower()
    
    if confirm == 's':
        batch_generate_synthetic_data(input_dir, output_dir, num_variations)
    else:
        print("Operación cancelada.")
    
    input("\nPresiona Enter para continuar...")


def show_dataset_info_menu():
    """Menú para mostrar información del dataset"""
    print("\n" + "="*70)
    print("INFORMACIÓN DEL DATASET")
    print("="*70)
    
    dataset_dir = input(f"\nDirectorio del dataset [{DATASET_DIR}]: ").strip()
    if not dataset_dir:
        dataset_dir = DATASET_DIR
    
    if Path(dataset_dir).exists():
        print_dataset_info(dataset_dir)
    else:
        print(f"\n❌ Error: El directorio {dataset_dir} no existe")
    
    input("\nPresiona Enter para continuar...")


def train_menu():
    """Menú para entrenamiento"""
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO")
    print("="*70)
    
    dataset_dir = input(f"\nDirectorio del dataset [{DATASET_DIR}]: ").strip()
    if not dataset_dir:
        dataset_dir = DATASET_DIR
    
    if not Path(dataset_dir).exists():
        print(f"\n❌ Error: El directorio {dataset_dir} no existe")
        input("\nPresiona Enter para continuar...")
        return
    
    epochs = input("Número de épocas [50]: ").strip()
    epochs = int(epochs) if epochs else 50
    
    fine_tune = input("¿Realizar fine-tuning? (s/n) [s]: ").strip().lower()
    fine_tune = fine_tune != 'n'
    
    print("\n🚀 Iniciando entrenamiento...")
    print("⏳ Esto puede tomar varios minutos u horas dependiendo del dataset y hardware...")
    
    try:
        model, model_name = train_model(dataset_dir, epochs, fine_tune)
        print(f"\n✅ Modelo entrenado exitosamente: {model_name}")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
    
    input("\nPresiona Enter para continuar...")


def predict_menu():
    """Menú para predicciones"""
    print("\n" + "="*70)
    print("REALIZAR PREDICCIONES")
    print("="*70)
    
    # Listar modelos disponibles
    models_path = Path(MODELS_DIR)
    if models_path.exists():
        model_files = list(models_path.glob("*.h5"))
        if model_files:
            print("\nModelos disponibles:")
            for i, model_file in enumerate(model_files, 1):
                print(f"  {i}. {model_file.name}")
            
            choice = input(f"\nSelecciona modelo [1-{len(model_files)}] o escribe la ruta: ").strip()
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_files):
                    model_path = str(model_files[idx])
                else:
                    model_path = choice
            except ValueError:
                model_path = choice
        else:
            print("\n⚠️  No se encontraron modelos en el directorio")
            model_path = input("Ruta al modelo: ").strip()
    else:
        model_path = input("Ruta al modelo: ").strip()
    
    if not Path(model_path).exists():
        print(f"\n❌ Error: El modelo {model_path} no existe")
        input("\nPresiona Enter para continuar...")
        return
    
    # Directorio con imágenes
    tiles_dir = input("\nDirectorio con losetas a predecir: ").strip()
    
    if not Path(tiles_dir).exists():
        print(f"\n❌ Error: El directorio {tiles_dir} no existe")
        input("\nPresiona Enter para continuar...")
        return
    
    output_json = input("Archivo JSON de salida [predictions.json]: ").strip()
    if not output_json:
        output_json = "predictions.json"
    
    print("\n🔍 Realizando predicciones...")
    
    try:
        model, class_mapping = load_model_and_mapping(model_path)
        results = predict_from_board(model, tiles_dir, class_mapping, output_json)
        print(f"\n✅ Predicciones completadas. Total: {results['total_tiles']} losetas")
    except Exception as e:
        print(f"\n❌ Error durante la predicción: {e}")
    
    input("\nPresiona Enter para continuar...")


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🎲 CLASIFICADOR DE LOSETAS DE CARCASSONNE")
    print("="*70)
    print("\nSegunda Versión - Sistema Automatizado con Transfer Learning")
    print("="*70)
    
    while True:
        print_menu()
        choice = input("\nSelecciona una opción [1-7]: ").strip()
        
        if choice == '1':
            organize_tiles_menu()
        elif choice == '2':
            find_clusters_menu()
        elif choice == '3':
            generate_synthetic_menu()
        elif choice == '4':
            show_dataset_info_menu()
        elif choice == '5':
            train_menu()
        elif choice == '6':
            predict_menu()
        elif choice == '7':
            print("\n👋 ¡Hasta luego!")
            sys.exit(0)
        else:
            print("\n❌ Opción inválida. Por favor selecciona 1-7.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrumpido por el usuario. ¡Hasta luego!")
        sys.exit(0)
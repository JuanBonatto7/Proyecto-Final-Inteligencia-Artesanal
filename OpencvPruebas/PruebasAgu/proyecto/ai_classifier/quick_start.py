"""
Quick Start - Script de Inicio Rápido

Este script guía al usuario a través de todo el proceso:
1. Preparar datos
2. Entrenar modelo
3. Evaluar resultados
"""

import os
import sys
from pathlib import Path


def print_banner():
    """Imprime banner de inicio."""
    print("\n" + "="*70)
    print("🎮 SISTEMA DE IA PARA CLASIFICACIÓN DE LOSETAS DE CARCASSONNE")
    print("="*70 + "\n")


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print("📦 Verificando dependencias...")
    
    required_packages = [
        'torch',
        'torchvision',
        'cv2',
        'PIL',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NO ENCONTRADO")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Faltan paquetes: {', '.join(missing)}")
        print("Instala con: pip install -r requirements.txt")
        return False
    
    print("\n✓ Todas las dependencias instaladas\n")
    return True


def setup_directories():
    """Crea directorios necesarios."""
    print("📁 Creando directorios...")
    
    directories = [
        'models',
        'checkpoints',
        'logs',
        'data',
        'evaluation_results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ {directory}/")
    
    print()


def guided_workflow():
    """Workflow guiado paso a paso."""
    print_banner()
    
    print("Este script te guiará a través del proceso completo:\n")
    print("1️⃣  Preparar datos (anotar losetas)")
    print("2️⃣  Entrenar modelo")
    print("3️⃣  Evaluar resultados")
    print("4️⃣  Hacer predicciones\n")
    
    # Paso 1: Datos
    print("="*70)
    print("PASO 1: PREPARAR DATOS")
    print("="*70 + "\n")
    
    print("¿Ya tienes las losetas extraídas con el detector?")
    print("Si no, primero ejecuta:")
    print("  cd 'Reconocimiento de losetas con 8 referencias'")
    print("  python carcassonne.py foto_tablero.jpg\n")
    
    tiles_dir = input("Ruta al directorio con losetas (ej: tiles/): ").strip()
    
    if not tiles_dir or not os.path.exists(tiles_dir):
        print(f"⚠️ El directorio '{tiles_dir}' no existe")
        return
    
    # Contar imágenes
    from glob import glob
    images = glob(os.path.join(tiles_dir, '*.png')) + glob(os.path.join(tiles_dir, '*.jpg'))
    print(f"\n✓ Encontradas {len(images)} imágenes en {tiles_dir}\n")
    
    print("¿Quieres anotar las losetas ahora? (s/n): ", end='')
    if input().lower().startswith('s'):
        print("\n🏷️ Iniciando herramienta de anotación...")
        print("\nControles:")
        print("  A-Z: Tipo de loseta")
        print("  0-3: Rotación")
        print("  M: Toggle meeple")
        print("  0-8: Posición meeple")
        print("  ENTER: Siguiente")
        print("  ESC: Salir\n")
        
        annotations_file = input("Archivo de salida (annotations.json): ").strip()
        if not annotations_file:
            annotations_file = "annotations.json"
        
        os.system(f'python annotate.py "{tiles_dir}" --output {annotations_file}')
    else:
        annotations_file = input("Ruta al archivo de anotaciones existente: ").strip()
        if not os.path.exists(annotations_file):
            print(f"⚠️ El archivo '{annotations_file}' no existe")
            return
    
    # Dividir datos
    print("\n📊 Dividiendo datos en train/val...")
    
    from dataset import split_annotations
    
    train_ratio = input("Proporción para entrenamiento (0.8): ").strip()
    train_ratio = float(train_ratio) if train_ratio else 0.8
    
    train_file, val_file = split_annotations(
        annotations_file=annotations_file,
        train_ratio=train_ratio,
        output_dir='data'
    )
    
    print(f"\n✓ Datos preparados:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}\n")
    
    # Paso 2: Entrenamiento
    print("="*70)
    print("PASO 2: ENTRENAR MODELO")
    print("="*70 + "\n")
    
    print("Configuración de entrenamiento:")
    epochs = input("  Épocas (100): ").strip()
    epochs = int(epochs) if epochs else 100
    
    batch_size = input("  Batch size (32): ").strip()
    batch_size = int(batch_size) if batch_size else 32
    
    backbone = input("  Backbone (efficientnet_b0/resnet18/resnet34/resnet50): ").strip()
    backbone = backbone if backbone else 'efficientnet_b0'
    
    print(f"\n🚀 Iniciando entrenamiento...")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Backbone: {backbone}\n")
    
    cmd = f'python train.py --train {train_file} --val {val_file} --epochs {epochs} --batch-size {batch_size} --backbone {backbone}'
    os.system(cmd)
    
    # Paso 3: Evaluación
    print("\n" + "="*70)
    print("PASO 3: EVALUAR MODELO")
    print("="*70 + "\n")
    
    print("¿Quieres evaluar el modelo entrenado? (s/n): ", end='')
    if input().lower().startswith('s'):
        model_path = input("Ruta al modelo (models/best_model.pth): ").strip()
        if not model_path:
            model_path = "models/best_model.pth"
        
        if os.path.exists(model_path):
            print("\n📊 Evaluando modelo...")
            os.system(f'python evaluate.py {model_path} {val_file}')
        else:
            print(f"⚠️ Modelo no encontrado: {model_path}")
    
    # Paso 4: Predicciones
    print("\n" + "="*70)
    print("PASO 4: HACER PREDICCIONES")
    print("="*70 + "\n")
    
    print("¿Quieres clasificar nuevas losetas? (s/n): ", end='')
    if input().lower().startswith('s'):
        model_path = input("Ruta al modelo (models/best_model.pth): ").strip()
        if not model_path:
            model_path = "models/best_model.pth"
        
        test_dir = input("Directorio con losetas a clasificar: ").strip()
        
        if os.path.exists(model_path) and os.path.exists(test_dir):
            output_file = input("Archivo de salida (predictions.json): ").strip()
            if not output_file:
                output_file = "predictions.json"
            
            print(f"\n🔮 Clasificando losetas...")
            os.system(f'python inference.py {model_path} batch {test_dir} --output {output_file}')
        else:
            print("⚠️ Modelo o directorio no encontrado")
    
    # Finalización
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70 + "\n")
    
    print("Archivos generados:")
    if os.path.exists('models/best_model.pth'):
        print("  ✓ models/best_model.pth - Mejor modelo entrenado")
    if os.path.exists('logs/training_curves.png'):
        print("  ✓ logs/training_curves.png - Gráficas de entrenamiento")
    if os.path.exists('evaluation_results/metrics.json'):
        print("  ✓ evaluation_results/metrics.json - Métricas de evaluación")
    
    print("\n📚 Para más información, consulta el README.md\n")


def main():
    """Función principal."""
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Crear directorios
    setup_directories()
    
    # Workflow guiado
    try:
        guided_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

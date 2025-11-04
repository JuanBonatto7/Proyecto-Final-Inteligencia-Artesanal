#!/usr/bin/env python3
"""
Script rápido para entrenar con pseudo-labels
Maneja automáticamente los campos extras de clustering
"""

import sys
import json
from pathlib import Path

# Verificar archivos
train_file = "train_annotations.json"
val_file = "val_annotations.json"

if not Path(train_file).exists():
    print(f"✗ Error: {train_file} no existe")
    print("\nPrimero ejecuta:")
    print("  python data-augmentation.py split pseudo_labels.json")
    sys.exit(1)

if not Path(val_file).exists():
    print(f"✗ Error: {val_file} no existe")
    sys.exit(1)

print("="*60)
print("ENTRENAMIENTO CON PSEUDO-LABELS")
print("="*60)

# Cargar y verificar datos
with open(train_file, 'r') as f:
    train_data = json.load(f)

with open(val_file, 'r') as f:
    val_data = json.load(f)

print(f"\n✓ Train: {len(train_data)} muestras")
print(f"✓ Val: {len(val_data)} muestras")

# Verificar que hay suficientes datos
if len(train_data) < 10:
    print("\n⚠️  ADVERTENCIA: Muy pocas muestras de entrenamiento")
    print("   Recomendado: al menos 50 muestras")
    print("\nConsideración: Usa data augmentation")
    print("  python data-augmentation.py augment train_annotations.json data/augmented/ 10")

# Estadísticas de pseudo-labels
pseudo_count = sum(1 for item in train_data if item.get('pseudo_labeled', False))
auto_count = sum(1 for item in train_data if item.get('auto_annotated', False))

if pseudo_count > 0:
    print(f"\n✓ Pseudo-labeled: {pseudo_count}")
if auto_count > 0:
    print(f"✓ Auto-annotated: {auto_count}")

# Contar tipos
type_counts = {}
for item in train_data:
    t = item.get('tile_type', -1)
    type_counts[t] = type_counts.get(t, 0) + 1

print(f"\n✓ Tipos únicos: {len(type_counts)}")

# Detectar desbalance
if len(type_counts) > 0:
    counts = list(type_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    
    if max_count > 5 * min_count:
        print(f"\n⚠️  Dataset desbalanceado:")
        print(f"   Tipo más común: {max_count} muestras")
        print(f"   Tipo más raro: {min_count} muestras")
        print(f"\n   Considera usar data augmentation para balancear")

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60)

# Importar y entrenar
try:
    from train_model import train
    
    print("\nEsto puede tomar varios minutos...")
    print("(Presiona Ctrl+C para cancelar)\n")
    
    success = train(train_file, val_file, epochs=50, batch_size=32)
    
    if success:
        print("\n" + "="*60)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\nModelo guardado en: best_carcassonne_model.pth")
        print("\nPróximos pasos:")
        print("1. Evaluar modelo:")
        print("   python model-evaluation.py best_carcassonne_model.pth test_annotations.json")
        print("\n2. Usar en producción:")
        print("   python carcassonne-pipeline.py best_carcassonne_model.pth nueva_foto.jpg")
    else:
        print("\n⚠️  Entrenamiento con problemas, revisa los logs")
        
except KeyboardInterrupt:
    print("\n\n⚠️  Entrenamiento cancelado por el usuario")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

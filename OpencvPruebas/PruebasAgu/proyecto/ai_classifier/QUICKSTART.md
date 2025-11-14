# 🚀 Guía de Inicio Rápido - 5 Minutos

## Paso 1: Instalar Dependencias (1 minuto)

```bash
cd proyecto/ai_classifier
pip install -r requirements.txt
```

## Paso 2: Preparar Tus Datos (2 minutos)

### Opción A: Usar Script Interactivo

```bash
python quick_start.py
```

El script te guiará paso a paso por todo el proceso.

### Opción B: Manualmente

```bash
# 1. Anotar losetas
python annotate.py ../tiles --output annotations.json

# 2. Dividir en train/val
python -c "from dataset import split_annotations; split_annotations('annotations.json', 0.8, 'data')"
```

## Paso 3: Entrenar (2 minutos para configurar)

```bash
# Entrenamiento básico
python train.py --train data/train_annotations.json --val data/val_annotations.json

# Entrenamiento personalizado
python train.py \
    --train data/train_annotations.json \
    --val data/val_annotations.json \
    --epochs 50 \
    --batch-size 32 \
    --backbone efficientnet_b0
```

## Paso 4: Usar el Modelo

```bash
# Clasificar un directorio
python inference.py models/best_model.pth batch tiles/ --output predictions.json

# Clasificar una imagen
python inference.py models/best_model.pth single tile.png --visualize
```

## ⚡ Ejemplo Completo

```bash
# 1. Ir al directorio
cd proyecto/ai_classifier

# 2. Instalar
pip install -r requirements.txt

# 3. Anotar (interface gráfica)
python annotate.py ../tiles --output annotations.json

# 4. Entrenar
python train.py \
    --train data/train_annotations.json \
    --val data/val_annotations.json \
    --epochs 50

# 5. Evaluar
python evaluate.py models/best_model.pth data/val_annotations.json

# 6. Usar
python inference.py models/best_model.pth batch new_tiles/ --output results.json
```

## 💡 Tips Rápidos

### Para Mejores Resultados

1. **Anota al menos 200-300 losetas** de diferentes tableros
2. **Balance las clases**: Trata de tener similar cantidad de cada tipo
3. **Verifica las anotaciones**: Usa las flechas ←/→ para revisar

### Si el Modelo No Funciona Bien

```bash
# Prueba con más épocas
python train.py --train ... --val ... --epochs 100

# Prueba con modelo más grande
python train.py --train ... --val ... --backbone resnet50

# Reduce el learning rate
python train.py --train ... --val ... --lr 0.0001
```

### Si Te Quedas Sin Memoria GPU

```bash
# Reduce el batch size
python train.py --train ... --val ... --batch-size 16

# O usa CPU (más lento pero funciona)
# El script detecta automáticamente si hay GPU
```

## 📊 Monitorear Progreso

Durante el entrenamiento verás:

```
Época 1/50
Training: 100%|████████| 25/25 [00:15<00:00, loss: 2.1234, tile_acc: 0.456]
Validation: 100%|████████| 10/10 [00:03<00:00, loss: 1.9876, tile_acc: 0.523]

Resultados Época 1:
  Train Loss: 2.1234 | Val Loss: 1.9876
  Train Acc:  0.4560 | Val Acc:  0.5230
  🎉 ¡Nuevo mejor modelo!
```

## 🎯 Archivos Importantes

Después del entrenamiento:

- `models/best_model.pth` - **Tu modelo entrenado** (usar para predicciones)
- `logs/training_curves.png` - Gráficas del entrenamiento
- `logs/training_history.json` - Historial completo
- `evaluation_results/metrics.json` - Métricas detalladas

## 🆘 Ayuda

- Ver README.md completo para documentación detallada
- Cada script tiene `--help`:
  ```bash
  python train.py --help
  python inference.py --help
  python annotate.py --help
  ```

## 🎮 Integración Completa

```python
# Script para clasificar un tablero completo
import os

# 1. Extraer losetas (detector)
os.chdir('Reconocimiento de losetas con 8 referencias')
os.system('python carcassonne.py tablero.jpg')

# 2. Clasificar con IA
os.chdir('../ai_classifier')
from inference import classify_tiles_from_detector

results = classify_tiles_from_detector(
    detector_tiles_dir='../Reconocimiento de losetas con 8 referencias/tiles',
    model_path='models/best_model.pth',
    output_json='board_results.json'
)

print(f"✓ {len(results)} losetas clasificadas")
```

---

**¡Listo! Ya tienes tu sistema de IA funcionando 🚀**

Para más detalles, consulta el README.md completo.

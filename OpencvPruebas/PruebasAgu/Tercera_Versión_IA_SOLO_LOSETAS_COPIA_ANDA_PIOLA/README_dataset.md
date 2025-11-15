# Dataset Organizado de Carcassonne - Guía de Uso

## 📁 Estructura del Dataset

El dataset ahora está organizado en carpetas individuales para cada tipo de loseta:

```
referencias_organizadas/
├── A/           # Losetas tipo A
│   ├── A_ref_001.png
│   └── A_ref_002.png (cuando agregues más)
├── B/           # Losetas tipo B
│   └── B_ref_001.png
├── C/           # Losetas tipo C (con escudo)
│   └── C_ref_001.png
├── ...
├── X/           # Losetas tipo X
│   └── X_ref_001.png
└── Shield.png   # Imagen de referencia del escudo
```

## 🆕 Agregar Nuevas Imágenes de Referencia

### Método 1: Usando el Script Automático

1. Coloca tus nuevas imágenes en una carpeta temporal (ej: `nuevas_imagenes/`)
2. Nombra los archivos empezando con la letra correspondiente:
   - `A_tile_001.png` → irá a carpeta A/
   - `C_shield_variant.png` → irá a carpeta C/
   - `M_different_angle.png` → irá a carpeta M/

3. Ejecuta el script:
   ```bash
   python add_references.py nuevas_imagenes/
   ```

### Método 2: Copiado Manual

1. Copia las imágenes directamente a las carpetas correspondientes
2. Usa nombres descriptivos: `A_ref_002.png`, `A_ref_003.png`, etc.

## 🔄 Re-entrenar el Modelo CNN

Después de agregar nuevas imágenes, re-entrena el modelo:

```bash
python train_cnn_multi.py
```

Este script:
- Carga todas las imágenes de todas las carpetas
- Aplica data augmentation (rotación, flip, variación de color)
- Entrena por 100 epochs con batch size de 8
- Guarda el modelo como `carcassonne_cnn_multi_model.pth`

## 📊 Ver Estadísticas del Dataset

```bash
python -c "
from pathlib import Path
print('Dataset actual:')
total = 0
for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    folder = Path(f'referencias_organizadas/{letter}')
    count = len(list(folder.glob('*.png'))) if folder.exists() else 0
    print(f'{letter}: {count} imágenes')
    total += count
print(f'Total: {total} imágenes')
"
```

## 🎯 Beneficios de la Nueva Estructura

- **Escalabilidad**: Fácil agregar múltiples imágenes por tipo
- **Organización**: Cada tipo de loseta tiene su propia carpeta
- **Robustez**: Más datos = mejor generalización del modelo
- **Mantenimiento**: Fácil identificar y gestionar referencias

## 🔧 Archivos Actualizados

- `tile_detector.py`: Ahora carga múltiples imágenes por carpeta
- `train_cnn_multi.py`: Nuevo script de entrenamiento con data augmentation
- `add_references.py`: Script para agregar nuevas imágenes automáticamente
- `organize_references.py`: Script que creó la estructura inicial

¡Ahora puedes agregar todas tus imágenes prolijas y tener un dataset mucho más robusto! 🚀
# 📊 Análisis de Resultados: Auto-Anotación

## Resultados Obtenidos

```
✓ Alta confianza (≥85%):    0
○ Media confianza (75-85%): 0
? Baja confianza (65-75%):  19
✗ Fallidas (<65%):          72
Total anotadas: 19/91 (20.9%)
```

## 🔍 Diagnóstico

La baja tasa de éxito (solo 19/91 losetas anotadas) indica que **las losetas detectadas son muy diferentes a las imágenes de referencia**.

### Posibles Causas:

1. **Diferencias en condiciones de captura**
   - Las referencias en `letras/` fueron tomadas en condiciones diferentes
   - Iluminación distinta
   - Ángulo o perspectiva diferente
   - Calidad de imagen diferente

2. **Referencias de tipo equivocado**
   - Las referencias son de losetas individuales (sin contexto)
   - Las losetas detectadas son del tablero completo (con contexto)
   - Pueden tener sombras, reflejos, o distorsión

3. **Losetas realmente diferentes**
   - Las losetas en el tablero NO coinciden con las referencias
   - Diferentes tipos o versiones de losetas

## 💡 Soluciones

### Opción 1: Mejorar las Referencias (RECOMENDADO)

**Problema:** Tus referencias actuales no representan cómo se ven las losetas en el tablero.

**Solución:** Crear referencias a partir de losetas ya detectadas del tablero.

```bash
# 1. Seleccionar manualmente algunas losetas del tablero como nuevas referencias
# Copia manualmente 24 losetas diferentes de tiles/ a una carpeta temporal

# 2. Organízalas por tipo en una estructura tipo:
# nuevas_referencias/
#   ├── A/
#   │   ├── ejemplo1.png
#   │   └── ejemplo2.png
#   ├── B/
#   │   └── ejemplo1.png
#   ...

# 3. Crea un script para convertirlas en referencias
```

**Script de conversión:**

```python
# create_references_from_tiles.py
from pathlib import Path
import shutil

# Selecciona UNA loseta de cada tipo del tablero
# y nómbrala según la letra correspondiente
manual_selection = {
    'tile_006_r0_c2.png': 'A',
    'tile_015_r1_c0.png': 'B',
    'tile_021_r1_c6.png': 'C',
    # ... continúa con las 24 letras
}

referencias_dir = Path('referencias_from_board')
referencias_dir.mkdir(exist_ok=True)

for tile_file, letter in manual_selection.items():
    src = Path('tiles') / tile_file
    idx = TileMapper().letter_to_idx(letter)
    dst = referencias_dir / f'tile_type_{idx}.png'
    shutil.copy(src, dst)
    print(f"✓ {letter}: {tile_file} → {dst.name}")
```

### Opción 2: Anotación Semi-Automática

En lugar de auto-anotar todo, usa un enfoque híbrido:

1. **Anota manualmente 5-10 losetas de cada tipo** del tablero
2. **Usa esas como referencias** para auto-anotar el resto
3. **Itera:** Corrige errores y mejora referencias

**Workflow:**

```bash
# Paso 1: Anotación manual de muestra inicial
python annotation_tool_letters.py tiles/ referencias/ 
# Anota solo 50-100 losetas (10-15 minutos)

# Paso 2: Entrenar modelo básico
python data-augmentation.py split annotations_with_letters.json
python train_model.py train_annotations.json val_annotations.json

# Paso 3: Usar modelo para pre-anotar el resto
python use_model_to_preannotate.py best_carcassonne_model.pth tiles/

# Paso 4: Revisar y corregir solo las dudosas
```

### Opción 3: Anotación Manual Completa

Si las referencias no se pueden mejorar fácilmente:

```bash
# Anotación manual tradicional (2-3 horas para 91 losetas)
python annotation_tool_letters.py tiles/ referencias/
```

**Atajos para agilizar:**
- Usa teclas A-X para selección rápida
- Usa Espacio para siguiente loseta
- Usa Ctrl+S para guardar

### Opción 4: Transfer Learning (Avanzado)

Usa un modelo pre-entrenado que requiere MUCHAS menos muestras:

```python
# train_with_transfer_learning.py
import torch
import torchvision.models as models

# Cargar ResNet pre-entrenado
model = models.resnet18(pretrained=True)

# Reemplazar última capa para 24 clases
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 24)

# Congelar capas iniciales
for param in model.parameters():
    param.requires_grad = False
    
# Solo entrenar la última capa
model.fc.requires_grad = True

# Entrenar con TAN SOLO 10-20 ejemplos por clase
# Transfer learning aprovecha el conocimiento de ImageNet
```

## 🎯 Recomendación Final

Para tu caso específico, **recomiendo el Enfoque Híbrido**:

### Plan de Acción:

1. **[10 min] Inspección Visual**
   ```bash
   # Abre algunas losetas y referencias para compararlas
   # ¿Son realmente similares?
   ```

2. **[20 min] Crear Referencias Mejoradas**
   - Selecciona manualmente 1-2 losetas de cada tipo del tablero
   - Cópialas a una nueva carpeta `referencias_mejoradas/`
   - Nómbralas como `tile_type_0.png`, `tile_type_1.png`, etc.

3. **[5 min] Re-ejecutar Auto-Anotación**
   ```bash
   python auto_annotate.py tiles/ referencias_mejoradas/ --threshold 0.65
   ```

4. **[30 min] Anotación Manual de Faltantes**
   ```bash
   # Solo anota las que fallaron
   python annotation_tool_letters.py tiles/ referencias_mejoradas/
   # Carga review_list.txt para saber cuáles revisar
   ```

5. **[Automático] Combinar Resultados**
   ```bash
   python combine_annotations.py auto_annotations.json manual_annotations.json final_annotations.json
   ```

## 📈 Métricas de Éxito Esperadas

Con referencias mejoradas del mismo tablero:

- **Alta confianza (≥85%):** 60-80%
- **Media confianza (75-85%):** 15-25%
- **Baja confianza (65-75%):** 5-10%
- **Fallidas (<65%):** 0-5%

## 🛠️ Herramientas de Diagnóstico

### Comparar Referencia vs Loseta Detectada

```python
# compare_samples.py
import cv2
import numpy as np

# Cargar referencia
ref = cv2.imread('referencias/tile_type_0.png')

# Cargar loseta del tablero
tile = cv2.imread('tiles/tile_006_r0_c2.png')

# Mostrar lado a lado
combined = np.hstack([ref, tile])
cv2.imshow('Referencia vs Tablero', combined)
cv2.waitKey(0)
```

---

**Siguiente Paso:** ¿Quieres que te ayude a crear referencias mejoradas a partir de las losetas del tablero?

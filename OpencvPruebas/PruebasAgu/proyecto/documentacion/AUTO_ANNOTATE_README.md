# 🤖 Auto-Anotador Inteligente de Losetas

## ¿Qué hace?

El **auto-anotador** elimina la necesidad de etiquetar manualmente cada loseta detectada. En lugar de pasar horas identificando una por una, el sistema usa inteligencia artificial para:

1. **Comparar** cada loseta con tus imágenes de referencia
2. **Identificar** automáticamente el tipo (letra A-X)
3. **Detectar** la rotación (0°, 90°, 180°, 270°)
4. **Calcular** un nivel de confianza para cada predicción
5. **Generar** el archivo de anotaciones listo para entrenar

## 🚀 Uso Rápido

```bash
# Forma más simple
python auto_annotate.py tiles/ referencias/

# Con opciones avanzadas
python auto_annotate.py tiles/ referencias/ --threshold 0.70 --review --output mis_anotaciones.json
```

## 📊 ¿Cómo funciona?

El sistema usa **4 métricas diferentes** para comparar losetas:

### 1. **SSIM (Structural Similarity Index)** - 35% del peso
- Compara la estructura de la imagen
- Excelente para detectar formas similares
- Robusto ante cambios de iluminación

### 2. **Correlación de Histogramas** - 25% del peso
- Compara la distribución de colores
- Usa espacio de color HSV
- Identifica losetas por su paleta de colores

### 3. **Template Matching** - 30% del peso
- Busca coincidencias exactas de patrones
- Muy preciso para imágenes similares
- Detecta detalles finos

### 4. **MSE (Mean Squared Error)** - 10% del peso
- Mide la diferencia píxel a píxel
- Complementa las otras métricas
- Penaliza diferencias grandes

## 🎯 Niveles de Confianza

El sistema clasifica cada anotación:

| Confianza | Rango | Símbolo | Descripción |
|-----------|-------|---------|-------------|
| **ALTA** | ≥85% | ✓ | Muy confiable, usar directamente |
| **MEDIA** | 75-85% | ○ | Confiable, revisar si es crítico |
| **BAJA** | 65-75% | ? | Requiere revisión manual |
| **FALLIDA** | <65% | ✗ | No se encontró coincidencia |

## 📈 Ejemplo de Salida

```
[1/150] ✓ tile_000_r0_c0.png: A (rot 0°) - Confianza: 92.3% (ALTA)
[2/150] ✓ tile_001_r0_c1.png: B (rot 90°) - Confianza: 88.7% (ALTA)
[3/150] ○ tile_002_r0_c2.png: C (rot 180°) - Confianza: 79.2% (MEDIA)
[4/150] ? tile_003_r0_c3.png: D (rot 270°) - Confianza: 67.8% (BAJA)
[5/150] ✗ tile_004_r0_c4.png: Sin coincidencia confiable

============================================================
RESUMEN DE AUTO-ANOTACIÓN
============================================================
✓ Alta confianza (≥85%):    120
○ Media confianza (75-85%): 23
? Baja confianza (65-75%):  5
✗ Fallidas (<65%):          2
============================================================
Total anotadas: 148/150
```

## 🔧 Opciones del Comando

### `--threshold` o `-t`
Umbral mínimo de confianza (default: 0.65)

```bash
# Más estricto (solo anotaciones muy seguras)
python auto_annotate.py tiles/ referencias/ --threshold 0.80

# Más permisivo (acepta más casos)
python auto_annotate.py tiles/ referencias/ --threshold 0.60
```

### `--review` o `-r`
Marca losetas de confianza media/baja para revisión manual

```bash
python auto_annotate.py tiles/ referencias/ --review
```

Esto genera un archivo `review_list.txt` con las losetas que deberías revisar.

### `--output` o `-o`
Especifica el archivo de salida

```bash
python auto_annotate.py tiles/ referencias/ --output tablero_1_annotations.json
```

## 📁 Archivos Generados

### `auto_annotations.json`
Archivo principal con todas las anotaciones:

```json
[
  {
    "image_path": "tiles/tile_000_r0_c0.png",
    "tile_letter": "A",
    "tile_type": 0,
    "rotation": 0,
    "confidence": 0.923,
    "method_scores": {
      "ssim": 0.945,
      "histogram": 0.889,
      "template": 0.932,
      "mse": 0.901
    },
    "has_meeple": false,
    "meeple_position": -1,
    "meeple_color": "none",
    "auto_annotated": true
  }
]
```

### `review_list.txt` (si se usa `--review`)
Lista de archivos que requieren revisión manual:

```
# LOSETAS PARA REVISIÓN MANUAL
# Total: 7

tiles/tile_003_r0_c3.png
tiles/tile_012_r1_c2.png
tiles/tile_045_r4_c5.png
...
```

## 🔄 Workflow Completo

### 1. Preparar Referencias
```bash
python tile_mapping.py prepare letras/ referencias/
```

### 2. Detectar Losetas del Tablero
```bash
python carcassonne.py foto_tablero.jpg
```

### 3. Auto-Anotar (¡NUEVO!)
```bash
python auto_annotate.py tiles/ referencias/
```

### 4. Revisar Casos Dudosos (Opcional)
```bash
# Revisar solo las losetas marcadas
python annotation_tool_letters.py tiles/ referencias/
# Cargar: review_list.txt
```

### 5. Entrenar Modelo
```bash
python data-augmentation.py split auto_annotations.json
python train_model.py train_annotations.json val_annotations.json
```

## 💡 Consejos para Mejores Resultados

### ✅ Buenas Prácticas

1. **Imágenes de referencia de calidad**
   - Buena iluminación uniforme
   - Sin sombras o reflejos
   - Loseta centrada y bien enfocada

2. **Fotos del tablero consistentes**
   - Mismas condiciones de luz
   - Similar distancia/ángulo
   - Tablero plano sin deformaciones

3. **Ajustar el threshold según necesidad**
   - Dataset pequeño: threshold bajo (0.60-0.65)
   - Dataset grande: threshold alto (0.75-0.80)
   - Producción: threshold muy alto (0.85+)

### ❌ Evitar

1. **Mezclar condiciones de iluminación**
   - Referencias con luz natural ≠ Tablero con luz artificial
   
2. **Imágenes de baja calidad**
   - Desenfocadas, pixeladas, o muy comprimidas

3. **Rotaciones incorrectas en referencias**
   - Asegúrate que las referencias están en orientación correcta (0°)

## 🆚 Comparación con Anotación Manual

| Aspecto | Auto-Anotación | Manual |
|---------|----------------|--------|
| **Tiempo** | Segundos | Horas |
| **150 losetas** | ~30 segundos | ~2-3 horas |
| **Precisión** | 85-95% | 100% (humano perfecto) |
| **Errores** | Predecibles (baja confianza) | Impredecibles (fatiga) |
| **Escalabilidad** | Miles de losetas sin problema | Se vuelve tedioso |
| **Rotación** | Automática (4 orientaciones) | Manual por cada loseta |
| **Coste** | CPU/GPU | Tiempo humano |

## 🐛 Solución de Problemas

### Problema: "Sin coincidencia confiable" en muchas losetas

**Soluciones:**
1. Reducir el threshold: `--threshold 0.60`
2. Verificar que las referencias son correctas
3. Revisar que las losetas detectadas tienen buena calidad

### Problema: Muchas anotaciones incorrectas

**Soluciones:**
1. Aumentar el threshold: `--threshold 0.75`
2. Usar `--review` para revisar casos dudosos
3. Mejorar las imágenes de referencia

### Problema: Error "No se encontraron losetas"

**Soluciones:**
1. Verificar que el directorio `tiles/` contiene archivos `.png`
2. Ejecutar primero `carcassonne.py` para detectar losetas

### Problema: Error "No existe el directorio de referencias"

**Soluciones:**
1. Ejecutar primero: `python tile_mapping.py prepare letras/ referencias/`
2. Verificar que el directorio existe

## 🔬 Técnicas Avanzadas

### Personalizar Pesos de Métricas

Edita la función `aggregate_scores()` en `auto_annotate.py`:

```python
weights = {
    'ssim': 0.40,        # Aumentar si la estructura es importante
    'histogram': 0.20,   # Reducir si los colores varían mucho
    'template': 0.35,    # Aumentar para coincidencias exactas
    'mse': 0.05         # Peso bajo por defecto
}
```

### Añadir Métricas Adicionales

Puedes implementar tus propias métricas de comparación:

```python
@staticmethod
def compute_custom_metric(img1: np.ndarray, img2: np.ndarray) -> float:
    # Tu implementación aquí
    score = ...
    return score
```

## 📚 Referencias

- **SSIM**: Wang et al. (2004) - Image Quality Assessment
- **Template Matching**: OpenCV Documentation
- **HSV Color Space**: Smith (1978) - Color Gamut Transform Pairs

## 🎓 Casos de Uso

### 1. Dataset Inicial
```bash
# Primera vez - acepta más casos para tener datos
python auto_annotate.py tiles/ referencias/ --threshold 0.60
```

### 2. Dataset de Producción
```bash
# Solo casos muy seguros
python auto_annotate.py tiles/ referencias/ --threshold 0.85 --review
```

### 3. Múltiples Tableros
```bash
# Anotar varios tableros secuencialmente
python auto_annotate.py tablero1/tiles/ referencias/ -o tablero1.json
python auto_annotate.py tablero2/tiles/ referencias/ -o tablero2.json
python auto_annotate.py tablero3/tiles/ referencias/ -o tablero3.json

# Luego combinar todos los JSON
```

## ✨ Ventajas Clave

1. **Ahorro de tiempo masivo**: De horas a segundos
2. **Consistencia**: No hay fatiga ni errores humanos
3. **Escalabilidad**: Procesa miles de imágenes sin problemas
4. **Transparencia**: Reporta confianza de cada decisión
5. **Flexible**: Ajustable según tus necesidades
6. **Sin entrenamiento previo**: No requiere modelo pre-entrenado

---

## 🤝 Contribuir

Si encuentras formas de mejorar las métricas de comparación o añadir nuevas técnicas, ¡son bienvenidas!

## 📄 Licencia

Mismo que el proyecto principal.

---

**¡Disfruta de la anotación automática! 🎉**

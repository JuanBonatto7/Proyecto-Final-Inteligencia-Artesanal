# 📚 Guía: Entrenar IA con Múltiples Tableros

## 🎯 Resumen Ejecutivo

**Pregunta:** Tengo varios tableros para entrenar a la IA, ¿cuál sería el procedimiento con cada imagen?

**Respuesta rápida:** Procesar cada tablero → Combinar todas las losetas → Entrenar una sola vez con TODO el dataset.

### ⚠️ **IMPORTANTE: Detección Semi-Automática**

La detección de losetas **NO es completamente automática**. Para cada tablero debes:

1. **Seleccionar manualmente 8 losetas de referencia** (toma ~30 segundos)
   - Distribuidas uniformemente por el tablero
   - Preferiblemente: 4 en esquinas + 4 en centros de bordes
   
2. **El sistema detecta automáticamente todas las demás** (~60-90 losetas)
   - Usa interpolación basada en tus 8 selecciones
   - No necesitas seleccionar las 60-90 losetas una por una ✅

**⏱️ Tiempo real:** ~2 minutos por tablero (no 30 segundos como podría parecer)

---

## 🔄 Flujo de Trabajo Completo

### **Opción 1: Automático (RECOMENDADO) ⚡**

```powershell
# Un solo comando para procesar todos los tableros
python procesar_multiples_tableros.py --dir fotos_tableros/ --train
```

### **Opción 2: Paso a Paso (Control Total) 🎛️**

```powershell
# 1. Procesar cada tablero
python procesar_multiples_tableros.py tablero1.jpg tablero2.jpg tablero3.jpg

# 2. (Opcional) Revisar/corregir anotaciones
python annotation_tool_letters.py tablero_01/tiles/ referencias/

# 3. Entrenar con todo el dataset
python train_model.py train_annotations.json val_annotations.json
```

---

## 📋 Procedimiento Detallado

### **FASE 1: Preparación**

```powershell
# Organizar tus fotos de tableros
fotos_tableros/
├── partida_1.jpg
├── partida_2.jpg
├── partida_3.jpg
├── partida_4.jpg
└── partida_5.jpg
```

### **FASE 2: Procesamiento por Tablero**

#### ⚠️ **IMPORTANTE: Requiere interacción del usuario**

Para cada tablero, debes:

```
Tablero #1 (partida_1.jpg)
├── 1️⃣ Detectar losetas (MANUAL)
│   ├── Se abre ventana interactiva
│   ├── Seleccionas 8 losetas de referencia (distribuidas uniformemente)
│   ├── Presionas ENTER para confirmar
│   ├── Presionas 's' para guardar
│   └── Resultado: tablero_01/tiles/*.png
│
├── 2️⃣ Auto-anotar (AUTOMÁTICO)  → tablero_01/annotations_01.json
└── 3️⃣ Agregar metadatos (AUTOMÁTICO) → (tablero_id, imagen origen)

Tablero #2 (partida_2.jpg)
├── 1️⃣ Detectar losetas (MANUAL - repites la selección interactiva)
├── 2️⃣ Auto-anotar (AUTOMÁTICO)
└── 3️⃣ Agregar metadatos (AUTOMÁTICO)

... y así con cada tablero
```

**Pasos de detección interactiva:**
1. Se abre ventana con tu tablero
2. Haces clic y arrastras para seleccionar 8 losetas completas bien distribuidas:
   - **Recomendado:** 4 en esquinas + 4 en los centros de cada borde
   - O simplemente 8 losetas distribuidas uniformemente por todo el tablero
3. Presionas **ENTER** cuando termines las 8 selecciones
4. El sistema detecta automáticamente todas las demás losetas por interpolación
5. Presionas **'s'** para guardar las losetas individuales
6. Presionas **'q'** para continuar con el siguiente tablero

### **FASE 3: Consolidación**

```json
dataset_completo.json = {
  "tablero_01": [60 losetas],
  "tablero_02": [75 losetas],
  "tablero_03": [82 losetas],
  "tablero_04": [68 losetas],
  "tablero_05": [91 losetas]
}
// Total: 376 losetas para entrenar
```

### **FASE 4: División del Dataset**

```
376 losetas totales
├── 70% → train_annotations.json    (263 losetas)
├── 20% → val_annotations.json      (75 losetas)
└── 10% → test_annotations.json     (38 losetas)
```

### **FASE 5: Entrenamiento**

```powershell
python train_model.py train_annotations.json val_annotations.json

# La IA aprende de TODAS las losetas de TODOS los tableros
```

---

## 💡 Ventajas de Múltiples Tableros

### ✅ **Mejor Rendimiento**

| Característica | 1 Tablero | 3 Tableros | 5 Tableros |
|---------------|-----------|------------|------------|
| **Dataset**   | ~90 losetas | ~270 losetas | ~450 losetas |
| **Precisión esperada** | 65-75% | 80-88% | 88-95% |
| **Robustez** | Baja | Media | Alta |
| **Generalización** | Pobre | Buena | Excelente |

### 🎯 **Variabilidad Capturada**

- **Iluminación:** Diferentes condiciones de luz
- **Ángulos:** Pequeñas diferencias de perspectiva
- **Calidad:** Variaciones en nitidez y color
- **Distribución:** Más ejemplos de losetas raras

---

## 🚀 Comandos Prácticos

### **Escenario 1: Tengo 3 tableros en diferentes ubicaciones**

```powershell
python procesar_multiples_tableros.py `
  C:\fotos\tablero1.jpg `
  D:\imagenes\carcassonne_partida.jpg `
  C:\Desktop\foto_juego.png
```

**⏱️ Tiempo estimado:** 5-8 minutos
- ~2 min por tablero (seleccionar 8 losetas + auto-anotar)

### **Escenario 2: Tengo una carpeta con 10 fotos**

```powershell
python procesar_multiples_tableros.py --dir C:\fotos_partidas\
```

**⏱️ Tiempo estimado:** 15-20 minutos
- ~2 min por tablero × 10 tableros

### **Escenario 3: Quiero revisar manualmente las anotaciones**

```powershell
# 1. Procesar sin auto-anotación
python procesar_multiples_tableros.py --manual --dir fotos/

# 2. Anotar manualmente cada tablero
python annotation_tool_letters.py tablero_01/tiles/ referencias/
python annotation_tool_letters.py tablero_02/tiles/ referencias/

# 3. Combinar y entrenar
python procesar_multiples_tableros.py --combine --train
```

### **Escenario 4: Ya procesé tableros, solo quiero combinar**

```powershell
python procesar_multiples_tableros.py --combine --split --train
```

---

## 📊 Ejemplo Real Paso a Paso

### **Situación:** Tienes 5 fotos de diferentes partidas

```powershell
# 1. Procesar todos los tableros (10-15 minutos)
python procesar_multiples_tableros.py --dir fotos_partidas/

# Resultado en terminal:
# ==============================================================
# PROCESANDO TABLERO #1
# ==============================================================
# Imagen: fotos_partidas\partida_20231015.jpg
# 
# [1/3] Detectando losetas...
# 
# ⚠️  ATENCIÓN: El siguiente paso requiere tu interacción
#     1. Se abrirá una ventana con la imagen del tablero
#     2. Selecciona 8 losetas de referencia distribuidas uniformemente
#     3. Presiona ENTER cuando termines
#     4. Después presiona 's' para guardar las losetas
# 
# Presiona ENTER para continuar con la detección interactiva...
# 
# [Ventana interactiva se abre - seleccionas 8 losetas]
# [Presionas ENTER]
# [Presionas 's' para guardar]
# [Presionas 'q' para continuar]
# 
# ✓ 87 losetas detectadas
# 
# [2/3] Auto-anotando con IA...
# ✓ 73 losetas anotadas (83.9%)
# 
# [3/3] Agregando metadatos...
# ✓ Tablero #1 procesado exitosamente
# 
# ... (repite para tableros 2-5 - cada uno requiere selección interactiva)
# 
# ==============================================================
# REPORTE DE PROCESAMIENTO
# ==============================================================
# 
# Tableros procesados: 5
#   ✓ Exitosos: 5
#   ⏳ Pendientes de anotación: 0
#   ✗ Fallidos: 0
# 
# Total de losetas: 423
# Total anotadas: 361
# Tasa de anotación: 85.3%
```

```powershell
# 2. Revisar estructura generada
Get-ChildItem

# Resultado:
# tablero_01/
#   ├── tiles/
#   │   ├── tile_0.png
#   │   └── ...
#   └── annotations_01.json
# tablero_02/
# tablero_03/
# tablero_04/
# tablero_05/
# dataset_completo.json      ← TODAS las losetas
# train_annotations.json
# val_annotations.json
# test_annotations.json
# reporte_procesamiento.json
```

```powershell
# 3. (Opcional) Revisar anotaciones con errores
python annotation_tool_letters.py tablero_03/tiles/ referencias/
```

```powershell
# 4. Entrenar modelo
python train_model.py train_annotations.json val_annotations.json

# La IA aprende de 361 losetas de 5 tableros diferentes
```

```powershell
# 5. Evaluar
python model-evaluation.py best_carcassonne_model.pth test_annotations.json

# Resultado esperado:
# Precisión global: ~88-92% (¡excelente!)
# Sensibilidad promedio: 0.87
# Especificidad promedio: 0.99
```

---

## 🎓 Conceptos Clave

### **¿Por qué NO entrenar con cada tablero por separado?**

❌ **MAL:** Entrenar modelo → tablero1, entrenar modelo → tablero2, etc.

- La IA "olvida" lo aprendido anteriormente
- Resultados inconsistentes
- Mucho tiempo desperdiciado

✅ **BIEN:** Combinar TODO → Entrenar UNA sola vez

- La IA aprende de toda la variabilidad
- Mejor generalización
- Entrenamiento eficiente

### **Metadatos de Tablero**

Cada anotación incluye de qué tablero proviene:

```json
{
  "image_path": "tablero_03/tiles/tile_42.png",
  "tile_type": "A",
  "rotation": 90,
  "tablero_id": 3,
  "tablero_imagen": "fotos_partidas/partida_3.jpg"
}
```

**Utilidad:**
- Rastrear origen de errores
- Identificar tableros problemáticos
- Análisis de rendimiento por partida

---

## 🔧 Resolución de Problemas

### **Problema 1: Auto-anotación baja (<70%)**

```powershell
# Solución: Anotar manualmente el primer tablero y usar como referencias
python annotation_tool_letters.py tablero_01/tiles/ referencias/

# Luego copiar tiles anotadas a referencias/
# Y re-procesar los demás tableros
```

### **Problema 2: Un tablero tiene muchos errores**

```powershell
# Revisar y corregir solo ese tablero
python annotation_tool_letters.py tablero_03/tiles/ referencias/

# Recombinar
python procesar_multiples_tableros.py --combine --split
```

### **Problema 3: Quiero agregar más tableros después**

```powershell
# 1. Procesar nuevos tableros (comienzan desde ID 6)
python procesar_multiples_tableros.py tablero_nuevo1.jpg tablero_nuevo2.jpg

# 2. Recombinar TODO
python procesar_multiples_tableros.py --combine --split

# 3. Re-entrenar con dataset ampliado
python mejora_modelo.py retrain best_carcassonne_model.pth train_annotations.json val_annotations.json
```

---

## 📈 Estrategia Recomendada

### **Si tienes 2-3 tableros:**
```powershell
python procesar_multiples_tableros.py --dir fotos/ --train
```
Tiempo: ~10-15 minutos total

### **Si tienes 4-6 tableros:**
```powershell
# Procesar y revisar manualmente
python procesar_multiples_tableros.py --dir fotos/

# Revisar solo tableros con baja tasa de anotación
python annotation_tool_letters.py tablero_XX/tiles/ referencias/

# Entrenar
python train_model.py train_annotations.json val_annotations.json
```
Tiempo: ~20-30 minutos total

### **Si tienes 7+ tableros:**
```powershell
# Procesar primeros 3 tableros con cuidado
python procesar_multiples_tableros.py tablero1.jpg tablero2.jpg tablero3.jpg

# Revisar y corregir manualmente
# (estos serán tus referencias de calidad)

# Procesar los demás con auto-anotación
python procesar_multiples_tableros.py tablero4.jpg tablero5.jpg ...

# Entrenar
python train_model.py train_annotations.json val_annotations.json
```
Tiempo: ~40-60 minutos total

---

## 🎯 Flujo Visual Completo

```
📸 FOTOS DE TABLEROS
    ↓
┌───────────────────────────────┐
│ procesar_multiples_tableros   │
└───────────────────────────────┘
    ↓
    ├─→ Tablero 1 → 87 losetas → anotaciones_01.json
    ├─→ Tablero 2 → 75 losetas → anotaciones_02.json
    ├─→ Tablero 3 → 92 losetas → anotaciones_03.json
    └─→ Tablero 4 → 81 losetas → anotaciones_04.json
    ↓
┌───────────────────────────────┐
│ COMBINAR                      │
│ dataset_completo.json         │
│ (335 losetas totales)         │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ DIVIDIR                       │
│ 70% train / 20% val / 10% test│
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ ENTRENAR                      │
│ train_model.py                │
└───────────────────────────────┘
    ↓
🤖 MODELO ENTRENADO
   best_carcassonne_model.pth
   (Listo para usar)
```

---

## ✅ Checklist

Antes de entrenar, verifica:

- [ ] Todas las fotos de tableros están en una carpeta accesible
- [ ] El directorio `referencias/` tiene imágenes de losetas limpias
- [ ] Has ejecutado `procesar_multiples_tableros.py`
- [ ] Verificaste que la tasa de auto-anotación es >70%
- [ ] Revisaste manualmente al menos un tablero
- [ ] Existe `dataset_completo.json` con todas las losetas
- [ ] Existen `train_annotations.json`, `val_annotations.json`, `test_annotations.json`
- [ ] Estás listo para entrenar

---

## 🚀 Comando Rápido Final

```powershell
# TODO EN UNO 🎉
python procesar_multiples_tableros.py --dir fotos_tableros/ --train --epochs 150
```

**Esto hace:**
1. ✓ Detecta losetas en todos los tableros (REQUIERE tu selección interactiva de 8 losetas por tablero)
2. ✓ Auto-anota con IA
3. ✓ Combina todas las anotaciones
4. ✓ Divide en train/val/test
5. ✓ Entrena modelo por 150 epochs
6. ✓ Guarda `best_carcassonne_model.pth`

**⏱️ Tiempo total:**
- Detección interactiva: ~2 min × número de tableros (ej: 10 min para 5 tableros)
- Auto-anotación: ~1-2 min por tablero
- Entrenamiento: ~20-30 min (depende de tu GPU/CPU)
- **Total para 5 tableros: 40-50 minutos**

---

## 📞 Próximos Pasos

Después de entrenar con múltiples tableros:

1. **Evaluar:**
   ```powershell
   python model-evaluation.py best_carcassonne_model.pth test_annotations.json
   ```

2. **Usar en producción:**
   ```powershell
   python carcassonne-pipeline.py best_carcassonne_model.pth nuevo_tablero.jpg
   ```

3. **Mejora continua:**
   - Agregar más tableros periódicamente
   - Re-entrenar con dataset ampliado
   - Alcanzar >95% de precisión

---

## 📝 Nota Final: Nivel de Automatización

### ❓ ¿Es realmente "automático"?

**Respuesta honesta:** Semi-automático

**Lo que SÍ es automático (90% del trabajo):**
- ✅ Detección de las 60-90 losetas por interpolación (solo necesitas marcar 8)
- ✅ Auto-anotación con template matching
- ✅ Combinación de datasets de múltiples tableros
- ✅ División en train/val/test
- ✅ Data augmentation
- ✅ Entrenamiento del modelo
- ✅ Evaluación y métricas

**Lo que requiere tu intervención (10% del trabajo):**
- 👆 Seleccionar 8 losetas de referencia por tablero (~30-60 seg)
- 👆 Presionar 's' para guardar, 'q' para continuar
- 👆 Revisar opcionalmente los resultados

**Alternativas para 100% automatización:**
- Ver `DETECCION_BATCH.md` para métodos avanzados
- Reutilizar transformaciones entre tableros similares
- Detección por contornos (menos preciso)

**Veredicto:** El método actual es el mejor compromiso entre precisión y eficiencia. 2 minutos por tablero para obtener detección perfecta vale totalmente la pena vs. detección automática con errores.

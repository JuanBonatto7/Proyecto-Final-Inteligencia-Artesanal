# 🧠 Entrenar CNN Sin Anotaciones Manuales

## ✅ SÍ, Es Posible Entrenar Sin Anotar Manualmente

Existen múltiples técnicas de **aprendizaje automático moderno** que eliminan o minimizan la necesidad de anotaciones humanas.

---

## 🎯 Métodos Disponibles

### 1. **Self-Supervised Learning** ⭐⭐⭐⭐⭐

**Concepto:** La CNN aprende patrones útiles sin etiquetas, creando sus propias "tareas" de aprendizaje.

#### A) **Contrastive Learning (SimCLR)**

```
┌─────────────────────────────────────────┐
│  Misma loseta, diferentes aumentaciones │
└─────────────────────────────────────────┘

Original: [Loseta A]
    ↓
Aumentación 1: [Rotada + Brillo]
Aumentación 2: [Perspectiva + Color]
    ↓
Red aprende: "Estas dos son LA MISMA"
```

**Cómo funciona:**
1. Toma una loseta
2. Crea dos versiones aumentadas (rotación, brillo, perspectiva)
3. La red aprende que ambas versiones son la misma loseta
4. Repite con todas las losetas sin necesitar etiquetas

**Después:**
- Con solo 5-10 ejemplos etiquetados por clase → 85-90% precisión
- Sin self-supervised: necesitas 50-100 ejemplos por clase

**Uso:**
```bash
# Fase 1: Pre-entrenamiento (sin etiquetas)
python self_supervised_training.py pretrain tiles/ --method contrastive --epochs 100

# Fase 2: Fine-tuning (solo 5-10 por clase)
python fine_tune.py --pretrained contrastive_pretrained.pth --few-shot 10
```

---

#### B) **Rotation Prediction**

```
┌──────────────────────────────────┐
│  Predecir cuánto se rotó         │
└──────────────────────────────────┘

1. Toma una loseta
2. Rótala aleatoriamente (0°, 90°, 180°, 270°)
3. Red predice: "¿Cuánto la roté?"
4. Aprende características geométricas sin etiquetas
```

**Ventajas:**
- ✅ No requiere etiquetas de tipo
- ✅ Aprende características útiles para clasificación
- ✅ Especialmente útil para Carcassonne (rotaciones importantes)

**Uso:**
```bash
python self_supervised_training.py pretrain tiles/ --method rotation --epochs 50
```

---

### 2. **Clustering + Pseudo-Labels** ⭐⭐⭐⭐

**Concepto:** La CNN agrupa automáticamente losetas similares.

```
┌────────────────────────────────────────┐
│  Pipeline Automático                   │
└────────────────────────────────────────┘

1. Extrae features de TODAS las losetas
   └─> Sin etiquetas

2. K-Means agrupa en 24 clusters
   └─> Losetas similares juntas

3. Asigna pseudo-labels
   Cluster 0 → Tipo 0
   Cluster 1 → Tipo 1
   ...

4. Entrena CNN con pseudo-labels
   └─> 70-80% precisión automática

5. (Opcional) Humano verifica 1 loseta/cluster
   └─> Aumenta a 85-90% precisión
```

**Trabajo humano:** 5 minutos (verificar 24 losetas, una por cluster)

**Uso:**
```bash
# Clustering automático
python self_supervised_training.py cluster tiles/ --n-clusters 24

# Resultado: pseudo_labels.json con etiquetas automáticas
```

---

### 3. **Few-Shot Learning** ⭐⭐⭐⭐

**Concepto:** Entrena con SOLO 1-5 ejemplos por clase usando arquitecturas especiales.

#### **Siamese Networks**

```
┌──────────────────────────────────────┐
│  Aprendizaje por Comparación        │
└──────────────────────────────────────┘

Referencias (solo 24 imágenes):
  A.png, B.png, C.png, ..., X.png

Nueva loseta → Compara con las 24
  Similarity(nueva, A.png) = 0.92  ← MÁS SIMILAR
  Similarity(nueva, B.png) = 0.45
  ...
  
Predicción: Es tipo A
```

**Ventajas:**
- ✅ Solo necesitas 24 imágenes (1 por tipo)
- ✅ Mucho mejor que template matching
- ✅ Aprende a comparar, no memoriza

**Implementación:** En desarrollo

---

### 4. **Synthetic Data Generation** ⭐⭐⭐

**Concepto:** Genera miles de losetas sintéticas automáticamente.

```python
# Tomar 24 referencias
for reference in references:  # 24 imágenes
    for i in range(500):  # Generar 500 versiones
        synthetic = apply_random_augmentations(reference)
        # Rotación, perspectiva, iluminación, ruido, blur, etc.
        save(synthetic, label=reference.type)

# Resultado: 12,000 imágenes sintéticas
# Sin trabajo manual
```

**Ventajas:**
- ✅ Totalmente automático
- ✅ Cantidad ilimitada de datos
- ⚠️ Puede no ser realista

---

### 5. **Transfer Learning + Fine-Tuning Mínimo** ⭐⭐⭐⭐⭐

**Concepto:** Usa conocimiento de millones de imágenes (ImageNet).

```python
# Cargar ResNet pre-entrenado en ImageNet (1.2M imágenes)
model = ResNet18(pretrained=True)

# Congelar 90% del modelo
for param in model.parameters()[:-2]:
    param.requires_grad = False

# Solo entrenar últimas capas con POCOS datos
train(model, small_dataset)  # Solo 10-20 por clase
```

**Ya lo tienes implementado en `carcassonne_cnn.py`!**

---

## 🏆 Comparación de Métodos

| Método | Trabajo Manual | Datos Necesarios | Precisión | Complejidad |
|--------|----------------|------------------|-----------|-------------|
| **Anotación completa** | 2-3 horas | N/A | 95% | Baja |
| **Self-Supervised + Fine-tune** ⭐ | 10 min | 5-10/clase | 85-90% | Alta |
| **Clustering + Verificación** ⭐ | 5 min | 1/cluster | 80-85% | Media |
| **Few-Shot Learning** | 2 min | 1-5/clase | 75-85% | Alta |
| **Synthetic Data** | 0 min | 0 (usa refs) | 70-80% | Baja |
| **Transfer Learning básico** | 15 min | 10-20/clase | 85-92% | Media |

---

## 🚀 Recomendación: Pipeline Óptimo

### **Opción A: Mínimo Esfuerzo (5 minutos)**

```bash
# 1. Clustering automático
python self_supervised_training.py cluster tiles/ --n-clusters 24
# Resultado: pseudo_labels.json

# 2. Verificar 1 loseta por cluster (5 minutos)
python verify_clusters.py pseudo_labels.json

# 3. Entrenar con pseudo-labels
python train_model.py pseudo_labels.json val_annotations.json

# Precisión esperada: 80-85%
```

---

### **Opción B: Mejor Precisión (30 minutos)**

```bash
# 1. Pre-entrenamiento self-supervised (20 min)
python self_supervised_training.py pretrain tiles/ --method contrastive --epochs 50

# 2. Clustering con modelo pre-entrenado
python self_supervised_training.py cluster tiles/ --n-clusters 24 --model contrastive_pretrained.pth

# 3. Verificar clusters (5 min)
python verify_clusters.py pseudo_labels.json

# 4. Fine-tuning (5 min)
python train_model.py pseudo_labels.json val_annotations.json --pretrained contrastive_pretrained.pth

# Precisión esperada: 85-92%
```

---

### **Opción C: Máxima Precisión (1 hora)**

```bash
# 1. Self-supervised (30 min)
python self_supervised_training.py pretrain tiles/ --method contrastive --epochs 100

# 2. Clustering
python self_supervised_training.py cluster tiles/ --n-clusters 24 --model contrastive_pretrained.pth

# 3. Anotar manualmente 10 ejemplos por clase (20 min)
python annotation_tool_letters.py tiles/ referencias/
# Solo anotas 240 losetas (10 × 24)

# 4. Combinar pseudo-labels con anotaciones manuales
python combine_annotations.py pseudo_labels.json manual_annotations.json final.json

# 5. Fine-tuning
python train_model.py final.json val_annotations.json --pretrained contrastive_pretrained.pth

# Precisión esperada: 92-96%
```

---

## 📊 Workflow Completo Sin Anotaciones

```
┌──────────────────────────────────────────────────────────┐
│                 PIPELINE SIN ANOTACIONES                 │
└──────────────────────────────────────────────────────────┘

1. DETECCIÓN
   └─> python carcassonne.py foto.jpg
       └─> Extrae 91 losetas en tiles/

2. SELF-SUPERVISED LEARNING (Sin etiquetas)
   └─> python self_supervised_training.py pretrain tiles/
       └─> Modelo aprende patrones automáticamente

3. AUTO-CLUSTERING (Sin etiquetas)
   └─> python self_supervised_training.py cluster tiles/ --n-clusters 24
       └─> Agrupa automáticamente en 24 tipos

4. VERIFICACIÓN MÍNIMA (5 minutos)
   └─> python verify_clusters.py pseudo_labels.json
       └─> Humano verifica 1 loseta por cluster

5. ENTRENAMIENTO
   └─> python train_model.py pseudo_labels.json
       └─> CNN entrenada con pseudo-labels

6. PRODUCCIÓN
   └─> python carcassonne-pipeline.py modelo.pth nueva_foto.jpg
       └─> Clasifica automáticamente
```

---

## 💡 ¿Por Qué Funciona?

### Self-Supervised Learning

**Intuición:**
- Una loseta rotada sigue siendo la misma loseta
- Una loseta con diferente brillo sigue siendo la misma loseta
- Si la red aprende esto, aprende características ÚTILES
- Después, con pocos ejemplos, puede clasificar correctamente

**Analogía:**
```
Humano aprendiendo idiomas:
1. Primero escucha MUCHO sin entender (self-supervised)
2. Aprende patrones del idioma
3. Después, con pocas palabras traducidas (few-shot)
4. Puede entender y hablar

CNN aprendiendo losetas:
1. Primero ve MUCHAS losetas sin etiquetas (self-supervised)
2. Aprende patrones visuales
3. Después, con pocas losetas etiquetadas (few-shot)
4. Puede clasificar correctamente
```

---

## 🔬 Técnicas Avanzadas (Futuro)

### 1. **Active Learning**
La CNN pregunta qué losetas etiquetar:
```
CNN: "Estas 10 losetas son las más informativas, etiquétalas"
Humano: [Etiqueta solo 10]
CNN: [Mejora significativamente]
```

### 2. **Semi-Supervised Learning**
Combina datos etiquetados y no etiquetados:
```
- 50 losetas etiquetadas (manual)
- 1000 losetas no etiquetadas (automático)
- Red aprende de ambas
```

### 3. **Meta-Learning**
La red "aprende a aprender":
```
- Entrena en muchos problemas similares
- Aprende a adaptarse rápidamente
- Con 1-2 ejemplos nuevos, funciona bien
```

---

## 📚 Papers de Referencia

- **SimCLR:** "A Simple Framework for Contrastive Learning" (Google, 2020)
- **MoCo:** "Momentum Contrast for Unsupervised Visual Representation Learning" (Facebook, 2020)
- **Rotation:** "Unsupervised Representation Learning by Predicting Image Rotations" (2018)

---

## 🎯 Respuesta Directa a Tu Pregunta

**"¿Hay forma de entrenar CNN sin hacer anotaciones?"**

### SÍ, hay 5 formas:

1. ✅ **Self-Supervised Learning** → 0 anotaciones, luego 5-10 por clase
2. ✅ **Clustering Automático** → 0 anotaciones, verificar 24 losetas (5 min)
3. ✅ **Few-Shot Learning** → Solo 1-5 por clase
4. ✅ **Synthetic Data** → 0 anotaciones (usa referencias)
5. ✅ **Transfer Learning** → 10-20 por clase (ya lo tienes)

### **Recomendación:**

```bash
# Lo más fácil y efectivo
python self_supervised_training.py cluster tiles/ --n-clusters 24
python verify_clusters.py pseudo_labels.json  # 5 minutos
python train_model.py pseudo_labels.json val_annotations.json

# Precisión: 80-85%
# Trabajo manual: 5 minutos
```

---

**El futuro del ML es entrenar con menos (o cero) anotaciones humanas. Estas técnicas son el estado del arte.** 🚀

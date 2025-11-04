# 🎮 Usar el Modelo en Producción - Guía Completa

## 🎯 ¿Qué Significa "Producción"?

**"Producción"** = Usar tu modelo CNN entrenado en **situaciones reales** para clasificar losetas automáticamente.

---

## 📸 Escenario Real

### **Antes de tener el modelo:**
```
👤 Humano toma foto del tablero
👁️  Humano mira cada loseta
✍️  Humano anota tipo y rotación
⏱️  Tiempo: 10-15 minutos
😓 Tedioso y propenso a errores
```

### **Después de entrenar el modelo:**
```
👤 Humano toma foto del tablero
🤖 IA analiza automáticamente
📊 Resultados en 5 segundos
✅ Precisión: 85-90%
😎 Sin esfuerzo humano
```

---

## 🚀 Comando de Producción

```bash
python carcassonne-pipeline.py best_carcassonne_model.pth nueva_foto.jpg
```

### **Desglose del Comando:**

| Parte | ¿Qué es? | Explicación |
|-------|----------|-------------|
| `carcassonne-pipeline.py` | Script principal | Orquesta todo el proceso |
| `best_carcassonne_model.pth` | Modelo CNN | Tu "cerebro" entrenado |
| `nueva_foto.jpg` | Foto nueva | Cualquier foto del tablero |

---

## 🔄 ¿Qué Hace el Pipeline?

### **PASO 1: DETECTAR LOSETAS** 🔍

```
Foto completa del tablero
         ↓
Algoritmo de visión por computadora
         ↓
Lista de losetas individuales

Resultado: 87 losetas encontradas
```

**Técnicas usadas:**
- Detección de esquinas
- Segmentación de grid
- Extracción de cada loseta

### **PASO 2: CLASIFICAR CON CNN** 🧠

```
Para cada loseta:
  1. Preprocesar imagen (resize, normalizar)
  2. Pasar por la CNN entrenada
  3. Obtener predicción
  
Loseta 1 → CNN → Tipo A, Rotación 90°, Confianza 95%
Loseta 2 → CNN → Tipo C, Rotación 0°, Confianza 89%
Loseta 3 → CNN → Tipo B, Rotación 180°, Confianza 92%
...
```

### **PASO 3: GENERAR RESULTADOS** 📊

```
├─ resultado_visual.png    (Imagen con etiquetas)
├─ predicciones.json       (Datos estructurados)
├─ estadisticas.txt        (Resumen)
└─ mapa_tablero.txt        (Grid del tablero)
```

---

## 📁 Salidas Generadas

### **1. Imagen Anotada (`resultado_visual.png`)**

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│    ┌────────┐  ┌────────┐  ┌────────┐               │
│    │   A    │  │   C    │  │   B    │               │
│    │  90°   │  │   0°   │  │  180°  │               │
│    │ 95% ✓  │  │ 89% ✓  │  │ 92% ✓  │               │
│    └────────┘  └────────┘  └────────┘               │
│                                                        │
│    ┌────────┐  ┌────────┐  ┌────────┐               │
│    │   D    │  │   A    │  │   X    │               │
│    │ 270°   │  │  90°   │  │  180°  │               │
│    │ 87% ✓  │  │ 91% ✓  │  │ 78% ?  │               │
│    └────────┘  └────────┘  └────────┘               │
│                                                        │
│  Leyenda:                                             │
│  ✓ = Alta confianza (>85%)                           │
│  ? = Baja confianza (<85%)                           │
└────────────────────────────────────────────────────────┘
```

### **2. JSON con Predicciones (`predicciones.json`)**

```json
{
  "tablero": {
    "total_losetas": 87,
    "tipos_unicos": 18,
    "fecha_analisis": "2025-11-03 14:30:00"
  },
  "losetas": [
    {
      "id": 0,
      "posicion": {"fila": 0, "columna": 0},
      "tipo_letra": "A",
      "tipo_id": 0,
      "rotacion": 90,
      "rotacion_grados": 90,
      "confianza": 0.95,
      "tiene_ficha": false,
      "color_ficha": "none"
    },
    {
      "id": 1,
      "posicion": {"fila": 0, "columna": 1},
      "tipo_letra": "C",
      "tipo_id": 2,
      "rotacion": 0,
      "rotacion_grados": 0,
      "confianza": 0.89,
      "tiene_ficha": true,
      "color_ficha": "red",
      "posicion_ficha": 3
    }
  ],
  "estadisticas": {
    "distribucion_tipos": {
      "A": 12,
      "B": 8,
      "C": 15,
      "D": 10
    },
    "distribucion_rotaciones": {
      "0°": 23,
      "90°": 19,
      "180°": 21,
      "270°": 24
    },
    "fichas_detectadas": 4,
    "precision_promedio": 0.887
  }
}
```

### **3. Estadísticas en Texto (`estadisticas.txt`)**

```
============================================================
ANÁLISIS DEL TABLERO DE CARCASSONNE
============================================================
Archivo: partida_2025-11-03.jpg
Fecha: 2025-11-03 14:30:00

DETECCIÓN
✓ Losetas detectadas: 87
✓ Tiempo de procesamiento: 4.3 segundos

CLASIFICACIÓN
✓ Precisión promedio: 88.7%
✓ Alta confianza (>85%): 72 losetas (82.8%)
✓ Media confianza (70-85%): 12 losetas (13.8%)
✓ Baja confianza (<70%): 3 losetas (3.4%)

DISTRIBUCIÓN DE TIPOS
Tipo A: 12 losetas (13.8%)  ████████
Tipo B: 8 losetas (9.2%)    █████
Tipo C: 15 losetas (17.2%)  ██████████
Tipo D: 10 losetas (11.5%)  ██████
Tipo E: 7 losetas (8.0%)    ████
...

ROTACIONES
0° (Norte):   23 losetas (26.4%)
90° (Este):   19 losetas (21.8%)
180° (Sur):   21 losetas (24.1%)
270° (Oeste): 24 losetas (27.6%)

FICHAS DE JUGADORES
✓ Fichas detectadas: 4
  - Rojo: 2 fichas
  - Azul: 1 ficha
  - Verde: 1 ficha
  - Amarillo: 0 fichas

LOSETAS CON BAJA CONFIANZA (Revisar manualmente)
1. Loseta #45 (Fila 5, Col 3): Tipo X, 78% confianza
2. Loseta #67 (Fila 7, Col 5): Tipo M, 72% confianza
3. Loseta #81 (Fila 9, Col 1): Tipo Q, 69% confianza

============================================================
```

---

## 🎮 Casos de Uso Prácticos

### **1. Análisis de Estrategia Post-Partida**

```bash
# Después de jugar
python carcassonne-pipeline.py best_carcassonne_model.pth final_partida.jpg

# Obtienes análisis completo:
# - ¿Qué losetas se usaron más?
# - ¿Dónde colocó cada jugador sus fichas?
# - ¿Qué estrategias siguieron?
```

### **2. Digitalización de Partidas**

```bash
# Convertir partida física a formato digital
python carcassonne-pipeline.py best_carcassonne_model.pth partida_fisica.jpg

# Ahora puedes:
# - Guardar la partida en base de datos
# - Compartirla online
# - Reproducirla en simulador
```

### **3. Verificación de Jugadas Legales**

```bash
# ¿Esta loseta encaja aquí?
python carcassonne-pipeline.py best_carcassonne_model.pth --verify tablero.jpg

# El sistema verifica:
# - Compatibilidad de bordes
# - Validez de la posición
# - Sugerencias de jugadas
```

### **4. Sistema de Puntuación Automático**

```bash
# Calcular puntos al final
python carcassonne-pipeline.py best_carcassonne_model.pth --score tablero_final.jpg

# Resultado:
# Jugador Rojo: 87 puntos
#   - Ciudades completadas: 45 puntos
#   - Caminos: 12 puntos
#   - Monasterios: 9 puntos
#   - Prados: 21 puntos
```

### **5. Tutorial Interactivo**

```bash
# Para principiantes
python carcassonne-pipeline.py best_carcassonne_model.pth --tutorial jugada.jpg

# El sistema explica:
# - Qué tipo de loseta es
# - Cómo está rotada
# - Dónde puede colocarse
# - Qué puntos da
```

---

## 🔧 Opciones Avanzadas del Pipeline

```bash
# Básico
python carcassonne-pipeline.py model.pth foto.jpg

# Con opciones
python carcassonne-pipeline.py model.pth foto.jpg \
    --output resultados/           # Directorio de salida
    --format json                  # Formato de salida
    --confidence 0.80              # Umbral de confianza
    --visualize                    # Mostrar resultados visuales
    --save-tiles                   # Guardar losetas individuales
    --no-detect-meeples            # No detectar fichas
    --grid-size 15x15              # Tamaño máximo del grid
```

---

## 📊 Ejemplo Real Completo

### **Situación:**
Estás jugando Carcassonne. Quieres analizar el tablero actual.

### **1. Tomas Foto:**
```
📱 *Saca foto con el móvil*
💾 Guarda como: tablero_actual.jpg
```

### **2. Ejecutas Pipeline:**
```bash
cd proyecto/
python carcassonne-pipeline.py best_carcassonne_model.pth tablero_actual.jpg
```

### **3. Proceso (4-5 segundos):**
```
============================================================
PIPELINE DE RECONOCIMIENTO DE CARCASSONNE
============================================================

[1/4] Cargando imagen...
✓ Imagen cargada: 3024x4032 píxeles

[2/4] Selección de puntos de referencia...
✓ 8 puntos seleccionados

[3/4] Detectando losetas...
✓ 87 losetas detectadas

[4/4] Clasificando losetas con CNN...
  Procesando loseta 87/87... ✓

============================================================
✓ ANÁLISIS COMPLETADO
============================================================

Resultados guardados en:
  - resultado_visual.png
  - predicciones.json
  - estadisticas.txt

Resumen:
  87 losetas clasificadas
  Precisión promedio: 88.7%
  Tiempo total: 4.3 segundos
```

### **4. Abres Resultados:**
```
📂 resultado_visual.png    → Ves tablero con etiquetas
📂 predicciones.json       → Datos para análisis
📂 estadisticas.txt        → Resumen legible
```

---

## 🎯 Diferencia Clave: Entrenamiento vs Producción

### **ENTRENAMIENTO** (Lo que hiciste antes)
```
Objetivo: ENSEÑAR a la CNN
Input: Losetas + Etiquetas (91 ejemplos)
Proceso: Ajustar millones de parámetros
Tiempo: 15-30 minutos
Output: best_carcassonne_model.pth
Frecuencia: Una vez (o cuando quieras mejorar)
```

### **PRODUCCIÓN** (Lo que harás ahora)
```
Objetivo: USAR la CNN entrenada
Input: Foto nueva (nunca vista)
Proceso: Clasificación automática
Tiempo: 4-5 segundos
Output: Predicciones, imágenes, estadísticas
Frecuencia: Cada vez que quieras analizar una partida
```

---

## ✅ Verificar que Todo Está Listo

```bash
# ¿Tienes el modelo entrenado?
ls best_carcassonne_model.pth
# Debe existir (~50MB)

# ¿Tienes el pipeline?
ls carcassonne-pipeline.py
# Debe existir

# ¿Tienes una foto de prueba?
ls fotos/*.jpg
```

---

## 🚀 Primer Uso en Producción

```bash
# Si ya entrenaste el modelo
python carcassonne-pipeline.py best_carcassonne_model.pth <alguna_foto_del_tablero>.jpg

# Verás:
# 1. Detección automática de losetas
# 2. Clasificación con CNN
# 3. Resultados guardados
# 4. Estadísticas mostradas
```

---

## 💡 Resumen

**"Usar en producción"** significa:

✅ **Tomas una foto nueva** del tablero  
✅ **Ejecutas el pipeline** con tu modelo entrenado  
✅ **Obtienes resultados automáticos** en segundos  
✅ **Sin trabajo manual** - todo es automático  
✅ **Puedes repetir** cuantas veces quieras  

**Es el momento de la verdad:** Ver si tu modelo funciona en el mundo real.

---

¿Tienes el modelo entrenado y listo? ¿Quieres probarlo con una foto?

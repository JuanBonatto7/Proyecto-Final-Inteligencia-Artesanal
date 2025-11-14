# Guía de Anotación - Herramienta Mejorada

## 🎯 Cambios Principales

### ✅ Problemas Solucionados

1. **GUI cortada**: Panel lateral optimizado de 380px → 280px
2. **Actualización de UI**: Sistema de redibujado eficiente que actualiza la interfaz en tiempo real
3. **Posición meeple 0-8**: Sistema simplificado sin conflictos
4. **Navegación con flechas**: Soporte mejorado + alternativas (< y >)

### 🆕 Mejoras Implementadas

#### Controles Rediseñados
- **R/T**: Rotación incremental (antes: teclas 0-3)
- **M**: Toggle meeple (antes: TAB)
- **0-8**: Posición directa del meeple (rango completo)
- **< >**: Navegación alternativa si flechas fallan

#### Interfaz Gráfica
- Panel lateral más compacto (280px)
- Colores mejorados con mejor contraste
- Indicadores visuales más claros
- Información más organizada

## 📋 Controles Completos

### Tipo de Loseta
| Tecla | Acción |
|-------|--------|
| A-X | Seleccionar tipo de loseta (sin Ñ) |
| ESPACIO | Loseta BLANCO |

### Rotación
| Tecla | Acción |
|-------|--------|
| + o = | Rotar derecha (+90°) |
| - o _ | Rotar izquierda (-90°) |

### Meeple
| Tecla | Acción |
|-------|--------|
| TAB | Activar/desactivar meeple |
| 0-8 | Posición del meeple (solo si está activo) |

### Navegación y Guardado
| Tecla | Acción |
|-------|--------|
| ENTER | Guardar anotación actual y avanzar a la siguiente |
| F5 | Guardar progreso sin avanzar |
| ← → | Navegar entre imágenes (flechas) |
| < > | Navegación alternativa |
| ESC | Salir y guardar todo |

## 🚀 Uso

```bash
# Desde el directorio proyecto
python ai_classifier/annotate.py "ruta/a/tiles" --output annotations.json
```

## 💡 Flujo de Trabajo Recomendado

1. **Inicia la herramienta** con el comando anterior
2. **Define el tipo** presionando la letra correspondiente (A-X)
3. **Ajusta la rotación** con R (derecha) o T (izquierda)
4. **Si hay meeple**:
   - Presiona M para activarlo
   - Presiona 0-8 para seleccionar la posición
5. **Guarda** con ENTER (avanza) o F5 (sin avanzar)
6. **Navega** con flechas o < >

## 📐 Sistema de Posiciones del Meeple

```
┌─────────┐
│ 0  1  2 │
│ 3  4  5 │
│ 6  7  8 │
└─────────┘
```

- **0-2**: Posiciones superiores
- **3-5**: Posiciones centrales
- **6-8**: Posiciones inferiores
- **4**: Centro exacto

## ⚡ Atajos de Productividad

### Anotación Rápida sin Meeple
```
[Tipo] → +/- → ENTER
Ejemplo: D → + → + → ENTER
```

### Anotación con Meeple
```
[Tipo] → +/- → TAB → [0-8] → ENTER
Ejemplo: D → + → TAB → 4 → ENTER
```

### Corrección Rápida
- Si te equivocas, usa < para volver
- Modifica lo necesario
- Presiona ENTER para guardar y continuar

## 🔍 Indicadores Visuales

### Header
- **[OK]** en verde: Loseta ya anotada
- **[--]** en naranja: Loseta pendiente
- **Número**: Progreso (actual/total)

### Secciones
- **TIPO** (verde): Información del tipo de loseta
- **ROTACION** (azul): Grados de rotación
- **MEEPLE** (verde/rojo): Estado y posición del meeple
- **CONTROLES** (amarillo): Referencia rápida

## 🐛 Solución de Problemas

### Las flechas no funcionan
**Solución**: Usa < y > como alternativa para navegación

### La GUI se ve cortada
**Solución**: El panel ahora es de 280px. Si aún se ve mal, ajusta el tamaño de la ventana manualmente

### No puedo poner posición 0-3 del meeple
**Solución**: 
1. Primero activa el meeple con M
2. Luego presiona 0-8 directamente (ya no hay conflicto con rotación)

### La interfaz no se actualiza
**Solución**: Este problema está solucionado. La UI ahora se redibuja automáticamente con cada cambio

## 📊 Estado de Guardado

- **Auto-guardado**: Al presionar ENTER (guarda y avanza)
- **Guardado manual**: Al presionar F5 (guarda sin avanzar)
- **Guardado final**: Al salir con ESC (guarda todo automáticamente)

## 🎨 Mejoras de UX

1. **Feedback inmediato**: Cada acción muestra un mensaje en consola
2. **Redibujado eficiente**: Solo actualiza cuando hay cambios
3. **Navegación fluida**: Múltiples opciones para moverse entre imágenes
4. **Sin conflictos**: Cada control tiene su propia tecla única
5. **Panel compacto**: Más espacio para ver la imagen de la loseta

---

**Última actualización**: Noviembre 2025
**Versión**: 2.0 (Mejorada)

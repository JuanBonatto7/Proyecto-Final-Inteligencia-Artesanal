#!/usr/bin/env python3
"""
Script de DEBUG para detectar qué está fallando en la detección de meeples
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path


def debug_meeple_detection(image_path: str):
    """Analiza paso por paso la detección de meeples"""
    
    print(f"\n{'='*70}")
    print(f"DEBUG: Analizando {image_path}")
    print(f"{'='*70}\n")
    
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ERROR: No se pudo cargar la imagen")
        return
    
    h, w = image.shape[:2]
    print(f"✅ Imagen cargada: {w}x{h} píxeles")
    
    # Convertir a HSV y grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"\n{'='*70}")
    print("PASO 1: DETECCIÓN DE AZUL")
    print(f"{'='*70}")
    
    # Máscara azul
    blue_mask = cv2.inRange(hsv, np.array([80, 30, 40]), np.array([140, 255, 255]))
    blue_pixels = np.sum(blue_mask > 0)
    blue_ratio = blue_pixels / (w * h)
    
    print(f"Píxeles azules: {blue_pixels} ({blue_ratio*100:.2f}%)")
    
    if blue_pixels > 100:
        print("✅ Se detectaron píxeles azules - posible meeple azul")
        cv2.imwrite("debug_blue_mask.png", blue_mask)
        print("   Guardado: debug_blue_mask.png")
    else:
        print("❌ No se detectaron píxeles azules")
    
    print(f"\n{'='*70}")
    print("PASO 2: DETECCIÓN DE NEGRO (MÉTODO ORIGINAL)")
    print(f"{'='*70}")
    
    # Método original (threshold 60)
    _, black_mask_original = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    black_pixels_original = np.sum(black_mask_original > 0)
    black_ratio_original = black_pixels_original / (w * h)
    
    print(f"Threshold 60: {black_pixels_original} píxeles ({black_ratio_original*100:.2f}%)")
    
    if black_pixels_original > 100:
        print("⚠️  PROBLEMA: Threshold 60 detecta MUCHOS píxeles (incluye sombras)")
        cv2.imwrite("debug_black_threshold60.png", black_mask_original)
        print("   Guardado: debug_black_threshold60.png")
    
    print(f"\n{'='*70}")
    print("PASO 3: DETECCIÓN DE NEGRO (MÉTODO MEJORADO)")
    print(f"{'='*70}")
    
    # Método mejorado (threshold 50 + HSV)
    _, black_mask_gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    lower_black_hsv = np.array([0, 0, 0])
    upper_black_hsv = np.array([180, 100, 60])
    black_mask_hsv = cv2.inRange(hsv, lower_black_hsv, upper_black_hsv)
    
    black_mask_improved = cv2.bitwise_or(black_mask_gray, black_mask_hsv)
    
    # Limpiar ruido
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    black_mask_improved = cv2.morphologyEx(black_mask_improved, cv2.MORPH_OPEN, kernel_small, iterations=1)
    black_mask_improved = cv2.morphologyEx(black_mask_improved, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    black_pixels_improved = np.sum(black_mask_improved > 0)
    black_ratio_improved = black_pixels_improved / (w * h)
    
    print(f"Threshold 50 + HSV + Morfología: {black_pixels_improved} píxeles ({black_ratio_improved*100:.2f}%)")
    
    if black_pixels_improved > 100:
        print("✅ Se detectaron píxeles negros - posible meeple negro")
        cv2.imwrite("debug_black_improved.png", black_mask_improved)
        print("   Guardado: debug_black_improved.png")
    else:
        print("❌ No se detectaron píxeles negros suficientes")
    
    print(f"\n{'='*70}")
    print("PASO 4: ANÁLISIS DE ESTADÍSTICAS DE IMAGEN")
    print(f"{'='*70}")
    
    print(f"Intensidad promedio: {np.mean(gray):.1f}")
    print(f"Intensidad mínima: {np.min(gray)}")
    print(f"Intensidad máxima: {np.max(gray)}")
    print(f"Desviación estándar: {np.std(gray):.1f}")
    
    # Histograma
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    dark_pixels = np.sum(hist[:50])  # Píxeles muy oscuros
    very_dark_pixels = np.sum(hist[:30])
    
    print(f"\nPíxeles oscuros (<50): {int(dark_pixels)} ({dark_pixels/(w*h)*100:.2f}%)")
    print(f"Píxeles MUY oscuros (<30): {int(very_dark_pixels)} ({very_dark_pixels/(w*h)*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print("PASO 5: ANÁLISIS DE CONTORNOS")
    print(f"{'='*70}")
    
    # Analizar contornos en máscara mejorada
    contours, _ = cv2.findContours(black_mask_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total de contornos encontrados: {len(contours)}")
    
    if len(contours) > 0:
        print("\nTop 5 contornos más grandes:")
        
        # Ordenar por área
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for i, contour in enumerate(contours_sorted):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            
            area_ratio = area / (w * h)
            
            # Calcular intensidad promedio
            mask_contour = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], -1, 255, -1)
            pixels = gray[mask_contour > 0]
            
            if len(pixels) > 0:
                mean_intensity = np.mean(pixels)
                std_intensity = np.std(pixels)
            else:
                mean_intensity = 0
                std_intensity = 0
            
            print(f"\n  Contorno #{i+1}:")
            print(f"    Área: {area:.0f} px² ({area_ratio*100:.2f}% de imagen)")
            print(f"    Circularidad: {circularity:.3f}")
            print(f"    Aspect ratio: {aspect_ratio:.2f}")
            print(f"    Intensidad promedio: {mean_intensity:.1f}")
            print(f"    Desviación estándar: {std_intensity:.1f}")
            print(f"    BBox: ({x}, {y}, {bw}, {bh})")
            
            # Evaluar criterios
            is_valid = True
            reasons = []
            
            min_area = (w * h) * 0.04
            max_area = (w * h) * 0.55
            
            if area < min_area:
                is_valid = False
                reasons.append(f"área muy pequeña (< {min_area:.0f})")
            elif area > max_area:
                is_valid = False
                reasons.append(f"área muy grande (> {max_area:.0f})")
            
            if circularity < 0.4:
                is_valid = False
                reasons.append(f"poca circularidad (< 0.4)")
            
            if aspect_ratio < 0.5 or aspect_ratio > 1.6:
                is_valid = False
                reasons.append(f"aspect ratio fuera de rango (0.5-1.6)")
            
            if mean_intensity > 70:
                is_valid = False
                reasons.append(f"intensidad muy alta (> 70)")
            
            if std_intensity > 30:
                is_valid = False
                reasons.append(f"color no uniforme (std > 30)")
            
            if is_valid:
                print(f"    ✅ CANDIDATO VÁLIDO PARA MEEPLE")
            else:
                print(f"    ❌ DESCARTADO: {', '.join(reasons)}")
        
        # Visualizar contornos
        vis_image = image.copy()
        cv2.drawContours(vis_image, contours_sorted[:5], -1, (0, 255, 0), 2)
        cv2.imwrite("debug_contours.png", vis_image)
        print(f"\n   Guardado: debug_contours.png (top 5 contornos)")
    
    print(f"\n{'='*70}")
    print("RESUMEN Y RECOMENDACIONES")
    print(f"{'='*70}\n")
    
    if blue_pixels > 100:
        print("🔵 Meeple AZUL detectado probablemente OK")
    
    if black_pixels_improved > 100:
        if len(contours) > 0:
            print("⚫ Meeple NEGRO: Píxeles detectados, revisar contornos")
            print("   → Ver debug_black_improved.png y debug_contours.png")
        else:
            print("⚫ Meeple NEGRO: Píxeles detectados pero sin contornos válidos")
            print("   → Posiblemente ruido o sombras fragmentadas")
    else:
        print("⚫ NO hay meeple negro (pocos píxeles oscuros)")
    
    print("\nArchivos generados:")
    if os.path.exists("debug_blue_mask.png"):
        print("  - debug_blue_mask.png")
    if os.path.exists("debug_black_threshold60.png"):
        print("  - debug_black_threshold60.png")
    if os.path.exists("debug_black_improved.png"):
        print("  - debug_black_improved.png")
    if os.path.exists("debug_contours.png"):
        print("  - debug_contours.png")


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_meeple_debug.py <imagen>")
        print("Ejemplo: python test_meeple_debug.py tiles_complete/tile_018.png")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Archivo no encontrado: {image_path}")
        return
    
    debug_meeple_detection(image_path)
    print("\n✅ Análisis completo\n")


if __name__ == "__main__":
    main()
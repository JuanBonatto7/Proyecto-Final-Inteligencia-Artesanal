#!/usr/bin/env python3
"""
Herramienta de verificación rápida de clusters
Permite revisar y corregir pseudo-labels generadas por clustering automático

Uso:
    python verify_clusters.py pseudo_labels.json [--text-only]
"""

import json
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse


def visualize_clusters(annotations_file: str):
    """Muestra ejemplos de cada cluster para verificación"""
    
    print("="*60)
    print("VERIFICACIÓN DE CLUSTERS")
    print("="*60)
    print("\nEsta herramienta te permite verificar rápidamente")
    print("si el clustering automático funcionó correctamente.\n")
    
    # Cargar anotaciones
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Organizar por cluster
    clusters = {}
    for ann in annotations:
        cluster_id = ann['tile_type']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(ann['image_path'])
    
    print(f"Total de clusters: {len(clusters)}")
    print(f"Total de losetas: {len(annotations)}\n")
    
    # Mostrar estadísticas
    print("Distribución:")
    for cluster_id in sorted(clusters.keys()):
        print(f"  Cluster {cluster_id}: {len(clusters[cluster_id])} losetas")
    
    print("\n" + "="*60)
    print("VISUALIZACIÓN")
    print("="*60)
    print("\nPresiona cualquier tecla para ver siguiente cluster")
    print("Presiona 'q' para salir\n")
    
    # Mostrar ejemplos de cada cluster
    for cluster_id in sorted(clusters.keys()):
        paths = clusters[cluster_id][:6]  # Mostrar máximo 6 ejemplos
        
        images = []
        for path in paths:
            if Path(path).exists():
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (150, 150))
                    images.append(img)
        
        if not images:
            print(f"⚠️  Cluster {cluster_id}: Sin imágenes válidas")
            continue
        
        # Crear mosaico (asegurar que todas las filas tengan el mismo ancho)
        rows = []
        row_size = 3  # 3 imágenes por fila
        
        for i in range(0, len(images), row_size):
            row_images = images[i:i+row_size]
            
            # Rellenar con imágenes negras si la fila no está completa
            while len(row_images) < row_size:
                blank = np.zeros_like(images[0])
                row_images.append(blank)
            
            row = np.hstack(row_images)
            rows.append(row)
        
        if rows:
            mosaic = np.vstack(rows)
            
            # Añadir texto
            text = f"Cluster {cluster_id} ({len(clusters[cluster_id])} losetas)"
            cv2.putText(mosaic, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Verificacion de Clusters", mosaic)
            key = cv2.waitKey(0)
            
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("SIGUIENTE PASO")
    print("="*60)
    print("\n¿Los clusters tienen sentido?")
    print("\n✓ SÍ → Usa directamente para entrenar:")
    print("     python train_model.py pseudo_labels.json val_annotations.json")
    print("\n⚠️  NO → Opciones:")
    print("     1. Ajusta número de clusters")
    print("     2. Usa pre-entrenamiento self-supervised primero")
    print("     3. Anota manualmente algunos ejemplos problemáticos")


def show_cluster_stats(annotations_file: str):
    """Muestra solo estadísticas de clusters (modo texto)"""
    
    print("="*60)
    print("ESTADÍSTICAS DE CLUSTERS")
    print("="*60)
    
    # Cargar anotaciones
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Organizar por cluster
    clusters = {}
    for ann in annotations:
        cluster_id = ann['tile_type']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(ann['image_path'])
    
    print(f"\nTotal de clusters: {len(clusters)}")
    print(f"Total de losetas: {len(annotations)}")
    
    # Estadísticas por cluster
    print("\n" + "="*60)
    print("DISTRIBUCIÓN POR CLUSTER")
    print("="*60)
    
    for cluster_id in sorted(clusters.keys()):
        paths = clusters[cluster_id]
        print(f"\nCluster {cluster_id}: {len(paths)} losetas")
        
        # Mostrar primeras 3 losetas
        for i, path in enumerate(paths[:3], 1):
            filename = Path(path).name
            print(f"  {i}. {filename}")
        
        if len(paths) > 3:
            print(f"  ... y {len(paths) - 3} más")
    
    # Análisis de balance
    counts = [len(clusters[cid]) for cid in clusters]
    avg = sum(counts) / len(counts)
    min_count = min(counts)
    max_count = max(counts)
    
    print("\n" + "="*60)
    print("ANÁLISIS DE BALANCE")
    print("="*60)
    print(f"Promedio por cluster: {avg:.1f} losetas")
    print(f"Mínimo: {min_count} losetas")
    print(f"Máximo: {max_count} losetas")
    
    if max_count > 3 * min_count:
        print("\n⚠️  ADVERTENCIA: Clusters muy desbalanceados")
        print("   Considera ajustar el número de clusters")
    else:
        print("\n✓ Clusters razonablemente balanceados")
    
    print("\n" + "="*60)
    print("SIGUIENTE PASO")
    print("="*60)
    print("\nPara entrenar con estos pseudo-labels:")
    print("  python train_model.py pseudo_labels.json val_annotations.json")
    print("\nPara ver los clusters visualmente:")
    print("  python verify_clusters.py pseudo_labels.json --visual")


def main():
    parser = argparse.ArgumentParser(description='Verificar clusters generados automáticamente')
    parser.add_argument('annotations_file', help='Archivo de pseudo-labels JSON')
    parser.add_argument('--text-only', action='store_true', 
                       help='Solo mostrar estadísticas (sin ventanas)')
    parser.add_argument('--visual', action='store_true',
                       help='Mostrar visualización con ventanas OpenCV')
    
    args = parser.parse_args()
    
    if not Path(args.annotations_file).exists():
        print(f"✗ Error: {args.annotations_file} no existe")
        sys.exit(1)
    
    if args.visual:
        try:
            visualize_clusters(args.annotations_file)
        except Exception as e:
            print(f"\n⚠️  Error en visualización: {e}")
            print("Mostrando solo estadísticas...\n")
            show_cluster_stats(args.annotations_file)
    else:
        # Por defecto, modo texto (más seguro)
        show_cluster_stats(args.annotations_file)


if __name__ == "__main__":
    main()

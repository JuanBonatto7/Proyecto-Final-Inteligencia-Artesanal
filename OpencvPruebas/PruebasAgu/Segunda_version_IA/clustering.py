"""
Organización automática de losetas usando clustering
"""

import numpy as np
import shutil
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from config import N_CLUSTERS, CLUSTERING_RANDOM_STATE
from model import create_contrastive_model
from data_utils import extract_features_from_images


def auto_organize_tiles(unlabeled_dir, output_dir, n_clusters=N_CLUSTERS, visualize=True):
    """
    Organiza automáticamente losetas en carpetas por similitud
    
    Args:
        unlabeled_dir: Directorio con imágenes sin etiquetar
        output_dir: Directorio de salida con clusters
        n_clusters: Número de clusters (tipos de losetas esperados)
        visualize: Si True, genera visualización del clustering
    
    Returns:
        labels: Array con etiquetas de cluster para cada imagen
        image_paths: Lista de rutas a las imágenes procesadas
    """
    print("\n" + "="*60)
    print("CLUSTERING AUTOMÁTICO DE LOSETAS")
    print("="*60)
    
    unlabeled_path = Path(unlabeled_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Obtener todas las imágenes
    image_paths = list(unlabeled_path.glob('*.jpg')) + list(unlabeled_path.glob('*.png'))
    
    if len(image_paths) == 0:
        print(f"⚠ No se encontraron imágenes en {unlabeled_dir}")
        return None, None
    
    print(f"\n📁 Encontradas {len(image_paths)} imágenes")
    
    # Crear modelo para extracción de características
    print("🔧 Cargando modelo de extracción de características...")
    model = create_contrastive_model()
    
    # Extraer características
    print("🔍 Extrayendo características de las imágenes...")
    features = extract_features_from_images(image_paths, model)
    
    print(f"✓ Características extraídas: {features.shape}")
    
    # Realizar clustering
    print(f"\n🎯 Aplicando K-Means con {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=CLUSTERING_RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Organizar en carpetas
    print("\n📂 Organizando imágenes en carpetas...")
    for img_path, label in zip(image_paths, labels):
        dest_dir = output_path / f"cluster_{label:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, dest_dir / img_path.name)
    
    # Mostrar distribución
    unique, counts = np.unique(labels, return_counts=True)
    print("\n📊 Distribución de clusters:")
    print("-"*60)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id:02d}: {count:3d} imágenes")
    print("-"*60)
    
    # Visualización
    if visualize:
        visualize_clustering(features, labels, output_path)
    
    print(f"\n✓ Losetas organizadas en: {output_dir}")
    print("🔍 Revisa las carpetas y renómbralas según el tipo de loseta")
    print("="*60 + "\n")
    
    return labels, image_paths


def visualize_clustering(features, labels, output_dir):
    """
    Genera visualización 2D del clustering usando PCA
    
    Args:
        features: Array de características
        labels: Etiquetas de cluster
        output_dir: Directorio donde guardar la visualización
    """
    print("\n📈 Generando visualización del clustering...")
    
    # Reducir dimensionalidad con PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab20',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Visualización del Clustering de Losetas (PCA 2D)')
    plt.grid(True, alpha=0.3)
    
    # Guardar
    output_path = Path(output_dir) / 'clustering_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualización guardada en: {output_path}")


def analyze_cluster_quality(features, labels):
    """
    Analiza la calidad del clustering
    
    Args:
        features: Array de características
        labels: Etiquetas de cluster
    
    Returns:
        dict con métricas de calidad
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    metrics = {
        'silhouette_score': silhouette_score(features, labels),
        'calinski_harabasz': calinski_harabasz_score(features, labels),
        'davies_bouldin': davies_bouldin_score(features, labels)
    }
    
    print("\n📊 Métricas de Calidad del Clustering:")
    print("-"*60)
    print(f"  Silhouette Score (mayor es mejor): {metrics['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz (mayor es mejor): {metrics['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin (menor es mejor): {metrics['davies_bouldin']:.4f}")
    print("-"*60)
    
    return metrics


def find_optimal_clusters(unlabeled_dir, max_clusters=30):
    """
    Encuentra el número óptimo de clusters usando el método del codo
    
    Args:
        unlabeled_dir: Directorio con imágenes sin etiquetar
        max_clusters: Número máximo de clusters a probar
    
    Returns:
        Plot con curva de inercia
    """
    print("\n🔍 Buscando número óptimo de clusters...")
    
    unlabeled_path = Path(unlabeled_dir)
    image_paths = list(unlabeled_path.glob('*.jpg')) + list(unlabeled_path.glob('*.png'))
    
    # Extraer características
    model = create_contrastive_model()
    features = extract_features_from_images(image_paths, model)
    
    # Probar diferentes números de clusters
    inertias = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=CLUSTERING_RANDOM_STATE, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        print(f"  K={k}: inercia={kmeans.inertia_:.2f}")
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Determinar K Óptimo')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimal_clusters_elbow.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Gráfico guardado en: optimal_clusters_elbow.png")
    print("💡 Busca el 'codo' en la curva para determinar el K óptimo")
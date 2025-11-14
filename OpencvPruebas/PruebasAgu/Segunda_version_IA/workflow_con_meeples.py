"""
Flujo de trabajo completo para clasificar losetas ignorando meeples
"""

import sys
from pathlib import Path


def workflow_completo():
    """
    Ejecuta el flujo completo: limpiar meeples → clustering → organizar
    """
    print("\n" + "="*70)
    print("🎲 CLASIFICADOR DE CARCASSONNE - WORKFLOW CON MEEPLES")
    print("="*70)
    print("\nEste workflow procesará las imágenes para ignorar los meeples")
    print("y luego organizará las losetas automáticamente.\n")
    
    # Paso 1: Configuración
    print("📋 PASO 1: CONFIGURACIÓN")
    print("-" * 70)
    
    input_dir = input("Directorio con imágenes originales [dataset/unlabeled/]: ").strip()
    if not input_dir:
        input_dir = "dataset/unlabeled/"
    
    if not Path(input_dir).exists():
        print(f"❌ Error: {input_dir} no existe")
        return
    
    clean_dir = input("Directorio para imágenes limpias [dataset/unlabeled_clean/]: ").strip()
    if not clean_dir:
        clean_dir = "dataset/unlabeled_clean/"
    
    clustered_dir = input("Directorio para clusters [dataset/clustered/]: ").strip()
    if not clustered_dir:
        clustered_dir = "dataset/clustered/"
    
    n_clusters = input("Número de tipos de losetas [24]: ").strip()
    n_clusters = int(n_clusters) if n_clusters else 24
    
    # Paso 2: Limpiar meeples
    print("\n" + "="*70)
    print("🧹 PASO 2: LIMPIANDO MEEPLES CIRCULARES")
    print("="*70)
    print("Removiendo solo fichas redondas (azules y negras)...\n")
    
    try:
        from mask_meeples_circular import batch_remove_circular_meeples
        batch_remove_circular_meeples(input_dir, clean_dir, show_samples=3)
    except Exception as e:
        print(f"❌ Error al limpiar meeples: {e}")
        print("Asegúrate de tener instalado: pip install opencv-python")
        return
    
    print("\n✅ Imágenes limpias guardadas en:", clean_dir)
    input("\nPresiona Enter para continuar al clustering...")
    
    # Paso 3: Clustering
    print("\n" + "="*70)
    print("🎯 PASO 3: CLUSTERING AUTOMÁTICO")
    print("="*70)
    print(f"Organizando en {n_clusters} grupos...\n")
    
    try:
        from clustering import auto_organize_tiles
        labels, image_paths = auto_organize_tiles(
            unlabeled_dir=clean_dir,
            output_dir=clustered_dir,
            n_clusters=n_clusters,
            visualize=True
        )
    except Exception as e:
        print(f"❌ Error en clustering: {e}")
        return
    
    # Paso 4: Instrucciones finales
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETADO")
    print("="*70)
    print("\n📁 Los clusters están en:", clustered_dir)
    print("\n🔍 PRÓXIMOS PASOS:")
    print("-" * 70)
    print("1. Ve a la carpeta:", clustered_dir)
    print("2. Abre cada cluster_XX y mira las imágenes")
    print("3. Renombra las carpetas según el tipo de loseta:")
    print()
    print("   Ejemplo en Windows:")
    print(f"   ren {clustered_dir}\\cluster_00 ciudad_completa")
    print(f"   ren {clustered_dir}\\cluster_01 carretera_recta")
    print(f"   ren {clustered_dir}\\cluster_02 monasterio")
    print()
    print("   Ejemplo en Linux/Mac:")
    print(f"   mv {clustered_dir}/cluster_00 dataset/tiles/ciudad_completa")
    print(f"   mv {clustered_dir}/cluster_01 dataset/tiles/carretera_recta")
    print(f"   mv {clustered_dir}/cluster_02 dataset/tiles/monasterio")
    print()
    print("4. Una vez renombradas, ejecuta:")
    print("   python main.py")
    print("   → Opción 3: Generar dataset sintético")
    print("   → Opción 5: Entrenar modelo")
    print("="*70 + "\n")


def test_meeple_removal():
    """
    Prueba la remoción de meeples en imágenes de ejemplo
    """
    print("\n" + "="*70)
    print("🧪 TEST DE REMOCIÓN DE MEEPLES")
    print("="*70)
    
    input_dir = input("\nDirectorio con imágenes de prueba: ").strip()
    
    if not Path(input_dir).exists():
        print(f"❌ Error: {input_dir} no existe")
        return
    
    print("\n🔍 Mostrando visualización de las primeras 5 imágenes...")
    print("Cierra cada ventana para continuar con la siguiente.\n")
    
    from mask_meeples import test_single_image
    
    image_files = list(Path(input_dir).glob('*.jpg'))[:5]
    
    for img_file in image_files:
        test_single_image(img_file)
    
    print("\n✅ Test completado")


def adjust_detection():
    """
    Ajusta la detección de meeples interactivamente
    """
    print("\n" + "="*70)
    print("🔧 AJUSTAR DETECCIÓN DE MEEPLES")
    print("="*70)
    print("\nSi los meeples no se detectan correctamente, ajusta los rangos aquí.\n")
    
    image_path = input("Ruta a imagen de prueba: ").strip()
    
    if not Path(image_path).exists():
        print(f"❌ Error: {image_path} no existe")
        return
    
    print("\n📊 Mostrando detección actual...")
    print("Analiza qué color NO se está detectando bien.\n")
    
    from mask_meeples import adjust_meeple_detection
    adjust_meeple_detection(image_path)
    
    print("\n¿Quieres ajustar los rangos? (s/n): ", end="")
    if input().strip().lower() == 's':
        print("\nRANGOS EN HSV (Matiz, Saturación, Valor)")
        print("-" * 70)
        print("Azul actual: [90, 50, 50] a [130, 255, 255]")
        print("  - Matiz (H): 90-130 (azul en la rueda de color)")
        print("  - Saturación (S): 50-255 (qué tan intenso es el azul)")
        print("  - Valor (V): 50-255 (qué tan brillante es)")
        print()
        print("Negro actual: V < 50")
        print("  - Cualquier color con V < 50 se considera negro")
        print()
        
        # Ejemplo de cómo ajustar
        print("\n💡 Para ajustar, edita mask_meeples.py líneas 31-38:")
        print("   lower_blue = np.array([H_min, S_min, V_min])")
        print("   upper_blue = np.array([H_max, S_max, V_max])")
        print("   upper_black = np.array([180, 255, V_max])")


def main_menu():
    """
    Menú principal del workflow con meeples
    """
    while True:
        print("\n" + "="*70)
        print("🎲 CLASIFICADOR DE CARCASSONNE - MANEJO DE MEEPLES")
        print("="*70)
        print("\n1. 🔄 Ejecutar workflow completo (limpiar → clustering)")
        print("2. 🧪 Probar remoción de meeples")
        print("3. 🔧 Ajustar detección de meeples")
        print("4. 🧹 Solo limpiar meeples (sin clustering)")
        print("5. 🎯 Solo clustering (asume imágenes ya limpias)")
        print("6. ❌ Salir")
        print("\n" + "="*70)
        
        choice = input("\nSelecciona opción [1-6]: ").strip()
        
        if choice == '1':
            workflow_completo()
        
        elif choice == '2':
            test_meeple_removal()
        
        elif choice == '3':
            adjust_detection()
        
        elif choice == '4':
            input_dir = input("Directorio entrada: ").strip()
            output_dir = input("Directorio salida: ").strip()
            
            try:
                from mask_meeples import batch_remove_meeples
                batch_remove_meeples(input_dir, output_dir, show_samples=3)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '5':
            input_dir = input("Directorio con imágenes limpias: ").strip()
            output_dir = input("Directorio para clusters: ").strip()
            n = int(input("Número de clusters [24]: ").strip() or "24")
            
            try:
                from clustering import auto_organize_tiles
                auto_organize_tiles(input_dir, output_dir, n, visualize=True)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '6':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrumpido por el usuario. ¡Hasta luego!")
        sys.exit(0)
"""
Script de configuración e integración completa del proyecto Carcassonne
Automatiza todo el proceso desde las imágenes de letras hasta el entrenamiento
"""

import os
import sys
import json
import shutil
from pathlib import Path
import subprocess


class ProjectSetup:
    """Configurador automático del proyecto"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.letters_dir = self.project_dir / "letras"
        self.references_dir = self.project_dir / "referencias"
        self.tiles_dir = self.project_dir / "tiles"
        self.models_dir = self.project_dir / "models"
        self.data_dir = self.project_dir / "data"
        
    def check_requirements(self):
        """Verifica que existan los archivos necesarios"""
        print("="*60)
        print("VERIFICANDO REQUISITOS")
        print("="*60)
        
        required_files = [
            'tile_mapping.py',
            'carcassonne_cnn.py',
            'annotation_tool_letters.py',
            'carcassonne.py'
        ]
        
        missing = []
        for file in required_files:
            path = self.project_dir / file
            if path.exists():
                print(f"✓ {file}")
            else:
                print(f"✗ {file} - FALTANTE")
                missing.append(file)
        
        if missing:
            print(f"\n⚠️  Faltan archivos: {', '.join(missing)}")
            return False
        
        print("\n✓ Todos los archivos necesarios están presentes")
        return True
    
    def check_letter_files(self):
        """Verifica que existan las imágenes de letras"""
        print("\n" + "="*60)
        print("VERIFICANDO IMÁGENES DE LETRAS")
        print("="*60)
        
        if not self.letters_dir.exists():
            print(f"✗ Directorio '{self.letters_dir}' no existe")
            print(f"\nPor favor crea el directorio y coloca tus archivos:")
            print(f"  {self.letters_dir}/")
            print(f"    A.jpg")
            print(f"    B.jpg")
            print(f"    ...")
            print(f"    X.jpg")
            print(f"    blanco.jpg")
            return False
        
        expected_letters = list("ABCDEFGHIJKLMNOPQRSTUVWX") + ["blanco"]
        extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
        
        found = []
        missing = []
        
        for letter in expected_letters:
            file_found = False
            for ext in extensions:
                file_path = self.letters_dir / f"{letter}.{ext}"
                if file_path.exists():
                    found.append(letter)
                    file_found = True
                    break
            
            if not file_found:
                missing.append(letter)
        
        print(f"\nEncontradas: {len(found)}/{len(expected_letters)} losetas")
        
        if found:
            print("\n✓ Losetas encontradas:")
            for i in range(0, len(found), 10):
                print(f"  {', '.join(found[i:i+10])}")
        
        if missing:
            print(f"\n⚠️  Losetas faltantes:")
            for i in range(0, len(missing), 10):
                print(f"  {', '.join(missing[i:i+10])}")
            
            response = input("\n¿Continuar de todos modos? (s/n): ")
            return response.lower() == 's'
        
        return True
    
    def setup_directories(self):
        """Crea los directorios necesarios"""
        print("\n" + "="*60)
        print("CREANDO ESTRUCTURA DE DIRECTORIOS")
        print("="*60)
        
        dirs = [
            self.references_dir,
            self.tiles_dir,
            self.models_dir,
            self.data_dir,
        ]
        
        for dir_path in dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Creado: {dir_path}")
            else:
                print(f"  Existe: {dir_path}")
    
    def prepare_references(self):
        """Prepara las losetas de referencia"""
        print("\n" + "="*60)
        print("PREPARANDO LOSETAS DE REFERENCIA")
        print("="*60)
        
        cmd = [
            sys.executable,
            'tile_mapping.py',
            'prepare',
            str(self.letters_dir),
            str(self.references_dir)
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.project_dir)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al preparar referencias: {e}")
            return False
    
    def show_reference_map(self):
        """Muestra el mapeo de referencias"""
        print("\n" + "="*60)
        print("VERIFICANDO MAPEO DE REFERENCIAS")
        print("="*60)
        
        cmd = [
            sys.executable,
            'tile_mapping.py',
            'show',
            str(self.references_dir)
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.project_dir)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al mostrar mapeo: {e}")
            return False
    
    def create_workflow_guide(self):
        """Crea una guía de workflow"""
        guide = """
╔═══════════════════════════════════════════════════════════════╗
║         GUÍA DE WORKFLOW - PROYECTO CARCASSONNE              ║
╚═══════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────┐
│ FASE 1: DETECCIÓN DE LOSETAS                                 │
└───────────────────────────────────────────────────────────────┘

1. Tomar foto del tablero de Carcassonne

2. Ejecutar detector de losetas:
   python carcassonne.py <foto_tablero.jpg>
   
   - Selecciona 8 losetas de referencia bien distribuidas
   - El sistema detectará automáticamente todas las losetas
   - Se guardarán en el directorio 'tiles/'

┌───────────────────────────────────────────────────────────────┐
│ FASE 2: ANOTACIÓN DE LOSETAS                                 │
└───────────────────────────────────────────────────────────────┘

3. Anotar las losetas detectadas:
   python annotation_tool_letters.py tiles/ referencias/
   
   - Usa la interfaz gráfica para anotar cada loseta
   - Selecciona letra (A-X o blanco)
   - Selecciona rotación (0°, 90°, 180°, 270°)
   - Indica si tiene ficha y su posición
   - Exporta a 'annotations_with_letters.json'

4. Convertir anotaciones a formato numérico:
   python tile_mapping.py convert annotations_with_letters.json train_annotations.json

5. Dividir en train/val/test:
   python data-augmentation.py split train_annotations.json

┌───────────────────────────────────────────────────────────────┐
│ FASE 3: AUMENTACIÓN DE DATOS (OPCIONAL)                      │
└───────────────────────────────────────────────────────────────┘

6. Aumentar dataset (recomendado para mejor precisión):
   python data-augmentation.py augment train_annotations.json data/augmented/ 10
   python data-augmentation.py augment val_annotations.json data/augmented_val/ 5

┌───────────────────────────────────────────────────────────────┐
│ FASE 4: ENTRENAMIENTO DEL MODELO                             │
└───────────────────────────────────────────────────────────────┘

7. Entrenar el modelo CNN:
   python train_model.py train_annotations.json val_annotations.json
   
   - El mejor modelo se guardará como 'best_carcassonne_model.pth'
   - Se generarán gráficas de entrenamiento

┌───────────────────────────────────────────────────────────────┐
│ FASE 5: EVALUACIÓN DEL MODELO                                │
└───────────────────────────────────────────────────────────────┘

8. Evaluar el modelo:
   python model-evaluation.py best_carcassonne_model.pth test_annotations.json

┌───────────────────────────────────────────────────────────────┐
│ FASE 6: PIPELINE COMPLETO                                    │
└───────────────────────────────────────────────────────────────┘

9. Usar pipeline completo en nueva foto:
   python carcassonne-pipeline.py best_carcassonne_model.pth nueva_foto.jpg
   
   - Detecta automáticamente todas las losetas
   - Clasifica cada una con el modelo CNN
   - Genera reportes y visualizaciones


╔═══════════════════════════════════════════════════════════════╗
║                    ATAJOS DE TECLADO                          ║
╚═══════════════════════════════════════════════════════════════╝

Herramienta de anotación:
  A-X         : Selección rápida de tipo de loseta
  ←/→         : Anterior/Siguiente loseta
  Ctrl+S      : Guardar anotación
  Space       : Siguiente loseta

Detector de losetas:
  S           : Guardar losetas individuales
  R           : Guardar imagen resultado
  Q           : Salir


╔═══════════════════════════════════════════════════════════════╗
║                 ESTRUCTURA DE ARCHIVOS                        ║
╚═══════════════════════════════════════════════════════════════╝

proyecto/
├── letras/                          # Imágenes originales
│   ├── A.jpg
│   ├── B.jpg
│   └── ...
├── referencias/                     # Losetas de referencia convertidas
│   ├── tile_type_0.png (A)
│   ├── tile_type_1.png (B)
│   └── ...
├── tiles/                           # Losetas detectadas del tablero
│   ├── tile_000_r0_c0.png
│   └── ...
├── data/                            # Datasets aumentados (opcional)
├── models/                          # Modelos entrenados
│   └── best_carcassonne_model.pth
├── annotations_with_letters.json   # Anotaciones con letras
├── train_annotations.json          # Dataset de entrenamiento
├── val_annotations.json            # Dataset de validación
└── test_annotations.json           # Dataset de prueba


╔═══════════════════════════════════════════════════════════════╗
║                      CONSEJOS                                 ║
╚═══════════════════════════════════════════════════════════════╝

✓ Toma fotos con buena iluminación y sin reflejos
✓ El tablero debe estar lo más plano posible
✓ Anota al menos 50-100 losetas de cada tipo para buen entrenamiento
✓ Usa aumentación de datos si tienes pocas muestras
✓ Selecciona losetas de referencia en las esquinas y bordes
✓ Verifica el mapeo de referencias antes de entrenar
"""
        guide_file = self.project_dir / "documentacion" / "WORKFLOW.txt"
        # ensure the documentation directory exists
        (self.project_dir / "documentacion").mkdir(parents=True, exist_ok=True)
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide)
    
        print(f"\n✓ Guía de workflow creada en: {guide_file}")
        return guide_file
    
    def create_training_script(self):
        """Crea script simplificado para entrenamiento"""
        script = """#!/usr/bin/env python3
\"\"\"
Script de entrenamiento simplificado para Carcassonne CNN
\"\"\"

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from carcassonne_cnn import (
    CarcassonneTileDataset, 
    CarcassonneCNN, 
    CarcassonneTrainer,
    create_data_transforms
)

def train(train_file, val_file, epochs=50, batch_size=32):
    print("="*60)
    print("ENTRENAMIENTO DEL MODELO CARCASSONNE")
    print("="*60)
    
    # Verificar archivos
    if not Path(train_file).exists():
        print(f"✗ Error: No se encuentra {train_file}")
        return False
    
    if not Path(val_file).exists():
        print(f"✗ Error: No se encuentra {val_file}")
        return False
    
    print(f"\\n✓ Archivos de datos encontrados")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    
    # Cargar datasets
    print(f"\\nCargando datasets...")
    train_dataset = CarcassonneTileDataset(
        train_file,
        transform=create_data_transforms(augment=True)
    )
    val_dataset = CarcassonneTileDataset(
        val_file,
        transform=create_data_transforms(augment=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Crear modelo
    print(f"\\nCreando modelo...")
    model = CarcassonneCNN()
    
    # Entrenar
    trainer = CarcassonneTrainer(model)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Graficar historia
    trainer.plot_history()
    
    print("\\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Modelo guardado en: best_carcassonne_model.pth")
    print(f"Gráfica guardada en: training_history.png")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python train_model.py <train_annotations.json> <val_annotations.json> [epochs]")
        print("\\nEjemplo:")
        print("  python train_model.py train_annotations.json val_annotations.json 50")
        sys.exit(1)
    
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    train(train_file, val_file, epochs)
"""
        
        script_file = self.project_dir / "train_model.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script)
        
        # Hacer ejecutable en Unix
        try:
            os.chmod(script_file, 0o755)
        except:
            pass
        
        print(f"✓ Script de entrenamiento creado en: {script_file}")
        return script_file
    
    def run_full_setup(self):
        """Ejecuta la configuración completa"""
        print("\n")
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║   CONFIGURACIÓN AUTOMÁTICA DEL PROYECTO CARCASSONNE         ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        
        # 1. Verificar requisitos
        if not self.check_requirements():
            print("\n✗ Configuración abortada: Faltan archivos necesarios")
            return False
        
        # 2. Verificar imágenes de letras
        if not self.check_letter_files():
            print("\n✗ Configuración abortada: Faltan imágenes de letras")
            return False
        
        # 3. Crear directorios
        self.setup_directories()
        
        # 4. Preparar referencias
        if not self.prepare_references():
            print("\n✗ Error al preparar referencias")
            return False
        
        # 5. Mostrar mapeo
        self.show_reference_map()
        
        # 6. Crear guía
        self.create_workflow_guide()
        
        # 7. Crear script de entrenamiento
        self.create_training_script()
        
        # Resumen final
        print("\n")
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║              ✓ CONFIGURACIÓN COMPLETADA                      ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        print("Próximos pasos:")
        print()
        print("1. Lee la guía completa:")
        print("   cat documentacion/WORKFLOW.txt")
        print()
        print("2. Toma una foto del tablero y detecta losetas:")
        print("   python carcassonne.py foto_tablero.jpg")
        print()
        print("3. Anota las losetas detectadas:")
        print()
        print("4. Convierte anotaciones:")
        print("   python tile_mapping.py convert annotations_with_letters.json train_annotations.json")
        print()
        print("5. Divide en train/val/test:")
        print("   python data-augmentation.py split train_annotations.json")
        print()
        print("6. Entrena el modelo:")
        print("   python train_model.py train_annotations.json val_annotations.json 50")
        print()
        print("="*60)
        
        return True


def main():
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = "."
    
    setup = ProjectSetup(project_dir)
    success = setup.run_full_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

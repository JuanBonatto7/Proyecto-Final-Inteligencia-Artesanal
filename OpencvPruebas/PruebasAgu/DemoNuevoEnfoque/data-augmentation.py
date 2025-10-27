"""
Data augmentation y utilidades para aumentar el dataset de entrenamiento
"""

import cv2
import numpy as np
from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
from PIL import Image


class CarcassonneDataAugmenter:
    """Aumenta el dataset con transformaciones realistas"""
    
    def __init__(self):
        # Transformaciones realistas para fotos de tableros
        self.transform = A.Compose([
            # Variaciones de iluminación
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.8
            ),
            
            # Variaciones de color
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.7
            ),
            
            # Sombras y reflejos
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),
            
            # Blur (desenfoque de cámara)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Ruido
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            
            # Distorsiones geométricas leves
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=1.0),
                A.GridDistortion(distort_limit=0.1, p=1.0),
            ], p=0.2),
            
            # Cambios de perspectiva leves
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            
            # Compresión JPEG (común en fotos)
            A.ImageCompression(
                quality_lower=85,
                quality_upper=100,
                p=0.3
            ),
        ])
    
    def augment_single_tile(self, 
                           image: np.ndarray, 
                           num_augmentations: int = 10) -> List[np.ndarray]:
        """Genera múltiples versiones aumentadas de una loseta"""
        augmented = []
        
        for _ in range(num_augmentations):
            transformed = self.transform(image=image)
            augmented.append(transformed['image'])
        
        return augmented
    
    def augment_dataset(self, 
                       annotations_file: str,
                       output_dir: str,
                       augmentations_per_tile: int = 10):
        """Aumenta todo el dataset"""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        new_annotations = []
        
        print(f"Aumentando dataset: {augmentations_per_tile} versiones por loseta")
        
        for i, ann in enumerate(tqdm(annotations)):
            # Cargar imagen original
            img = cv2.imread(ann['image_path'])
            if img is None:
                print(f"Warning: No se pudo cargar {ann['image_path']}")
                continue
            
            # Agregar original
            original_filename = f"aug_{i:04d}_orig.png"
            original_path = output_path / original_filename
            cv2.imwrite(str(original_path), img)
            
            new_ann = ann.copy()
            new_ann['image_path'] = str(original_path)
            new_ann['augmentation_id'] = 0
            new_ann['source_image'] = ann['image_path']
            new_annotations.append(new_ann)
            
            # Generar versiones aumentadas
            augmented_images = self.augment_single_tile(img, augmentations_per_tile)
            
            for j, aug_img in enumerate(augmented_images):
                aug_filename = f"aug_{i:04d}_{j+1:02d}.png"
                aug_path = output_path / aug_filename
                cv2.imwrite(str(aug_path), aug_img)
                
                aug_ann = ann.copy()
                aug_ann['image_path'] = str(aug_path)
                aug_ann['augmentation_id'] = j + 1
                aug_ann['source_image'] = ann['image_path']
                new_annotations.append(aug_ann)
        
        # Guardar nuevas anotaciones
        output_annotations = output_path / 'augmented_annotations.json'
        with open(output_annotations, 'w') as f:
            json.dump(new_annotations, f, indent=2)
        
        print(f"\n✓ Dataset aumentado:")
        print(f"  Original: {len(annotations)} imágenes")
        print(f"  Aumentado: {len(new_annotations)} imágenes")
        print(f"  Factor: {len(new_annotations) / len(annotations):.1f}x")
        print(f"  Guardado en: {output_dir}")
        print(f"  Anotaciones: {output_annotations}")


class SyntheticTileGenerator:
    """Genera losetas sintéticas con variaciones de rotación"""
    
    def __init__(self, reference_tiles_dir: str):
        self.ref_dir = Path(reference_tiles_dir)
        self.reference_tiles = self.load_references()
    
    def load_references(self) -> Dict[int, np.ndarray]:
        """Carga losetas de referencia"""
        refs = {}
        for i in range(24):
            ref_path = self.ref_dir / f'tile_type_{i}.png'
            if ref_path.exists():
                refs[i] = cv2.imread(str(ref_path))
        return refs
    
    def rotate_tile(self, image: np.ndarray, rotation: int) -> np.ndarray:
        """Rota una loseta (0, 1, 2, 3 = 0°, 90°, 180°, 270°)"""
        if rotation == 0:
            return image
        elif rotation == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 3:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def add_meeple(self, 
                   image: np.ndarray, 
                   position: int,
                   color: str = 'blue') -> np.ndarray:
        """Añade una ficha de jugador en la posición especificada"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Posiciones en grid 3x3
        positions_3x3 = [
            (w//6, h//6),       # 0: superior-izq
            (w//2, h//6),       # 1: superior-centro
            (5*w//6, h//6),     # 2: superior-der
            (w//6, h//2),       # 3: centro-izq
            (w//2, h//2),       # 4: centro
            (5*w//6, h//2),     # 5: centro-der
            (w//6, 5*h//6),     # 6: inferior-izq
            (w//2, 5*h//6),     # 7: inferior-centro
            (5*w//6, 5*h//6),   # 8: inferior-der
        ]
        
        if position < 0 or position >= len(positions_3x3):
            return img
        
        # Colores de fichas
        colors_bgr = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'black': (0, 0, 0)
        }
        
        color_bgr = colors_bgr.get(color, (255, 0, 0))
        
        # Dibujar ficha (círculo)
        center = positions_3x3[position]
        radius = min(w, h) // 10
        
        # Sombra
        cv2.circle(img, (center[0]+2, center[1]+2), radius, (50, 50, 50), -1)
        # Ficha
        cv2.circle(img, center, radius, color_bgr, -1)
        # Borde
        cv2.circle(img, center, radius, (0, 0, 0), 2)
        
        return img
    
    def generate_synthetic_dataset(self,
                                   output_dir: str,
                                   samples_per_type: int = 50,
                                   meeple_probability: float = 0.3):
        """Genera dataset sintético completo"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        annotations = []
        sample_id = 0
        
        colors = ['blue', 'red', 'green', 'yellow', 'black']
        
        print(f"Generando dataset sintético...")
        print(f"  Tipos de losetas: {len(self.reference_tiles)}")
        print(f"  Muestras por tipo: {samples_per_type}")
        
        for tile_type, ref_img in tqdm(self.reference_tiles.items()):
            for _ in range(samples_per_type):
                # Rotación aleatoria
                rotation = random.randint(0, 3)
                rotated = self.rotate_tile(ref_img, rotation)
                
                # Añadir ficha aleatoriamente
                has_meeple = random.random() < meeple_probability
                
                if has_meeple:
                    meeple_pos = random.randint(0, 8)
                    meeple_color = random.choice(colors)
                    final_img = self.add_meeple(rotated, meeple_pos, meeple_color)
                else:
                    meeple_pos = -1
                    meeple_color = 'none'
                    final_img = rotated
                
                # Guardar imagen
                filename = f"synth_{sample_id:05d}.png"
                filepath = output_path / filename
                cv2.imwrite(str(filepath), final_img)
                
                # Crear anotación
                ann = {
                    'image_path': str(filepath),
                    'tile_type': tile_type,
                    'rotation': rotation,
                    'has_meeple': has_meeple,
                    'meeple_position': meeple_pos,
                    'meeple_color': meeple_color,
                    'synthetic': True
                }
                annotations.append(ann)
                
                sample_id += 1
        
        # Guardar anotaciones
        annotations_file = output_path / 'synthetic_annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"\n✓ Dataset sintético generado:")
        print(f"  Total de imágenes: {len(annotations)}")
        print(f"  Directorio: {output_dir}")
        print(f"  Anotaciones: {annotations_file}")


class DatasetSplitter:
    """Divide el dataset en train/validation/test"""
    
    @staticmethod
    def split_dataset(annotations_file: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     stratified: bool = True):
        """Divide dataset balanceadamente"""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        if stratified:
            # Estratificar por tipo de loseta
            by_type = {}
            for ann in annotations:
                tile_type = ann['tile_type']
                if tile_type not in by_type:
                    by_type[tile_type] = []
                by_type[tile_type].append(ann)
            
            train_anns = []
            val_anns = []
            test_anns = []
            
            for tile_type, anns in by_type.items():
                random.shuffle(anns)
                n = len(anns)
                
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_anns.extend(anns[:n_train])
                val_anns.extend(anns[n_train:n_train + n_val])
                test_anns.extend(anns[n_train + n_val:])
        else:
            random.shuffle(annotations)
            n = len(annotations)
            
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_anns = annotations[:n_train]
            val_anns = annotations[n_train:n_train + n_val]
            test_anns = annotations[n_train + n_val:]
        
        # Guardar splits
        base_path = Path(annotations_file).parent
        
        splits = {
            'train_annotations.json': train_anns,
            'val_annotations.json': val_anns,
            'test_annotations.json': test_anns
        }
        
        for filename, anns in splits.items():
            output_file = base_path / filename
            with open(output_file, 'w') as f:
                json.dump(anns, f, indent=2)
        
        print("✓ Dataset dividido:")
        print(f"  Train: {len(train_anns)} ({len(train_anns)/len(annotations)*100:.1f}%)")
        print(f"  Val:   {len(val_anns)} ({len(val_anns)/len(annotations)*100:.1f}%)")
        print(f"  Test:  {len(test_anns)} ({len(test_anns)/len(annotations)*100:.1f}%)")


class DatasetAnalyzer:
    """Analiza y muestra estadísticas del dataset"""
    
    @staticmethod
    def analyze_dataset(annotations_file: str):
        """Analiza el dataset"""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print("="*60)
        print("ANÁLISIS DEL DATASET")
        print("="*60)
        
        # Estadísticas básicas
        total = len(annotations)
        print(f"\nTotal de imágenes: {total}")
        
        # Distribución de tipos
        types_count = {}
        for ann in annotations:
            t = ann['tile_type']
            types_count[t] = types_count.get(t, 0) + 1
        
        print(f"\nTipos de losetas: {len(types_count)}")
        print("Distribución por tipo:")
        for t in sorted(types_count.keys()):
            count = types_count[t]
            print(f"  Tipo {t:2d}: {count:4d} ({count/total*100:5.1f}%)")
        
        # Distribución de rotaciones
        rotations_count = {0: 0, 1: 0, 2: 0, 3: 0}
        for ann in annotations:
            r = ann['rotation']
            rotations_count[r] = rotations_count.get(r, 0) + 1
        
        print("\nDistribución de rotaciones:")
        for r, count in rotations_count.items():
            print(f"  {r*90:3d}°: {count:4d} ({count/total*100:5.1f}%)")
        
        # Fichas de jugador
        with_meeple = sum(1 for ann in annotations if ann['has_meeple'])
        without_meeple = total - with_meeple
        
        print(f"\nFichas de jugador:")
        print(f"  Con ficha:  {with_meeple:4d} ({with_meeple/total*100:5.1f}%)")
        print(f"  Sin ficha:  {without_meeple:4d} ({without_meeple/total*100:5.1f}%)")
        
        # Balance del dataset
        max_count = max(types_count.values())
        min_count = min(types_count.values())
        balance_ratio = min_count / max_count
        
        print(f"\nBalance del dataset:")
        print(f"  Máximo por tipo: {max_count}")
        print(f"  Mínimo por tipo: {min_count}")
        print(f"  Ratio de balance: {balance_ratio:.2f}")
        
        if balance_ratio < 0.5:
            print("  ⚠️  Dataset desbalanceado - considera aumentar clases minoritarias")
        else:
            print("  ✓ Dataset balanceado")


def main():
    """Función principal"""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python data_utils.py augment <annotations.json> <output_dir> [num_aug]")
        print("  python data_utils.py synthetic <references_dir> <output_dir> [samples_per_type]")
        print("  python data_utils.py split <annotations.json> [train_ratio] [val_ratio]")
        print("  python data_utils.py analyze <annotations.json>")
        return
    
    command = sys.argv[1]
    
    if command == 'augment':
        if len(sys.argv) < 4:
            print("Error: Especifica annotations.json y output_dir")
            return
        
        augmenter = CarcassonneDataAugmenter()
        num_aug = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        augmenter.augment_dataset(sys.argv[2], sys.argv[3], num_aug)
    
    elif command == 'synthetic':
        if len(sys.argv) < 4:
            print("Error: Especifica references_dir y output_dir")
            return
        
        generator = SyntheticTileGenerator(sys.argv[2])
        samples = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        generator.generate_synthetic_dataset(sys.argv[3], samples)
    
    elif command == 'split':
        if len(sys.argv) < 3:
            print("Error: Especifica annotations.json")
            return
        
        train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        val_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
        test_ratio = 1.0 - train_ratio - val_ratio
        
        DatasetSplitter.split_dataset(sys.argv[2], train_ratio, val_ratio, test_ratio)
    
    elif command == 'analyze':
        if len(sys.argv) < 3:
            print("Error: Especifica annotations.json")
            return
        
        DatasetAnalyzer.analyze_dataset(sys.argv[2])
    
    else:
        print(f"Comando desconocido: {command}")


if __name__ == "__main__":
    main()

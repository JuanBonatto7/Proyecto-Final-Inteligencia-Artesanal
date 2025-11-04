import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from PIL import Image

# IMPORTAR EL MAPPER DE LETRAS
try:
    from tile_mapping import TileMapper
    MAPPER = TileMapper()
    NUM_TILE_TYPES = MAPPER.get_num_types()  # 25 tipos (A-X + blanco)
except ImportError:
    print("⚠️  Advertencia: No se encontró tile_mapping.py, usando 24 tipos por defecto")
    MAPPER = None
    NUM_TILE_TYPES = 24


@dataclass
class TileAnnotation:
    """Anotación de una loseta"""
    image_path: str
    tile_type: int  # 0-24 (25 tipos: A-X + blanco)
    rotation: int   # 0, 1, 2, 3 (0°, 90°, 180°, 270°)
    has_meeple: bool
    meeple_position: int  # 0-8, -1 si no hay ficha
    meeple_color: str  # 'red', 'blue', 'green', 'yellow', 'black', 'none'
    tile_letter: Optional[str] = None  # NUEVO: letra de la loseta (A-X o blanco)
    pseudo_labeled: bool = False  # NUEVO: si fue etiquetada automáticamente
    auto_annotated: bool = False  # NUEVO: si fue auto-anotada
    confidence: Optional[float] = None  # NUEVO: confianza de auto-anotación
    method_scores: Optional[Dict] = None  # NUEVO: scores de métodos de comparación


class CarcassonneTileDataset(Dataset):
    """Dataset para losetas de Carcassonne"""
    
    def __init__(self, annotations_file: str, transform=None):
        # Obtener directorio base para resolver rutas relativas
        annotations_path = Path(annotations_file)
        base_dir = annotations_path.parent
        
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.annotations = []
        skipped = 0
        for item in data:
            # Si tiene tile_letter, convertir a tile_type
            if 'tile_letter' in item and MAPPER is not None:
                if 'tile_type' not in item:
                    item['tile_type'] = MAPPER.letter_to_idx(item['tile_letter'])
            
            # Convertir ruta relativa a absoluta
            if not Path(item['image_path']).is_absolute():
                item['image_path'] = str(base_dir / item['image_path'])
            
            # Verificar que el archivo existe antes de añadirlo
            if Path(item['image_path']).exists():
                self.annotations.append(TileAnnotation(**item))
            else:
                skipped += 1
                print(f"⚠️  Archivo no encontrado, omitiendo: {item['image_path']}")
        
        self.transform = transform
        
        print(f"Dataset cargado: {len(self.annotations)} losetas")
        if skipped > 0:
            print(f"⚠️  {skipped} archivos no encontrados y omitidos")
        if MAPPER:
            print(f"Usando {NUM_TILE_TYPES} tipos de losetas (A-X + blanco)")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Cargar imagen
        image = Image.open(ann.image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Preparar labels
        labels = {
            'tile_type': torch.tensor(ann.tile_type, dtype=torch.long),
            'rotation': torch.tensor(ann.rotation, dtype=torch.long),
            'has_meeple': torch.tensor(int(ann.has_meeple), dtype=torch.float32),
            'meeple_position': torch.tensor(ann.meeple_position + 1, dtype=torch.long)  # 0-9 (0=no meeple)
        }
        
        return image, labels


class CarcassonneCNN(nn.Module):
    """Red neuronal multi-tarea para reconocimiento de losetas"""
    
    def __init__(self, num_tile_types=None, num_rotations=4, num_positions=10):
        super(CarcassonneCNN, self).__init__()
        
        # USAR AUTOMÁTICAMENTE 25 TIPOS SI HAY MAPPER
        if num_tile_types is None:
            num_tile_types = NUM_TILE_TYPES
        
        self.num_tile_types = num_tile_types
        
        # Backbone: ResNet18 pre-entrenado
        backbone = models.resnet18(pretrained=True)
        
        # Extraer feature extractor (sin la capa final)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Dimensión de features
        feature_dim = 512
        
        # Heads para cada tarea
        self.tile_type_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tile_types)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_rotations)
        )
        
        self.has_meeple_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.meeple_position_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_positions)
        )
        
        print(f"Modelo inicializado con {num_tile_types} tipos de losetas")
    
    def forward(self, x):
        # Extraer features
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Predicciones de cada tarea
        tile_type = self.tile_type_head(features)
        rotation = self.rotation_head(features)
        has_meeple = self.has_meeple_head(features)
        meeple_position = self.meeple_position_head(features)
        
        return {
            'tile_type': tile_type,
            'rotation': rotation,
            'has_meeple': has_meeple,
            'meeple_position': meeple_position
        }


class CarcassonneTrainer:
    """Entrenador del modelo"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.tile_type_criterion = nn.CrossEntropyLoss()
        self.rotation_criterion = nn.CrossEntropyLoss()
        self.has_meeple_criterion = nn.BCELoss()
        self.meeple_position_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'tile_type_acc': [],
            'rotation_acc': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calcular losses
            loss_tile = self.tile_type_criterion(
                outputs['tile_type'], 
                labels['tile_type'].to(self.device)
            )
            
            loss_rotation = self.rotation_criterion(
                outputs['rotation'],
                labels['rotation'].to(self.device)
            )
            
            loss_has_meeple = self.has_meeple_criterion(
                outputs['has_meeple'].squeeze(),
                labels['has_meeple'].to(self.device)
            )
            
            loss_position = self.meeple_position_criterion(
                outputs['meeple_position'],
                labels['meeple_position'].to(self.device)
            )
            
            # Loss total (ponderado)
            loss = (2.0 * loss_tile + 
                   1.5 * loss_rotation + 
                   1.0 * loss_has_meeple + 
                   1.0 * loss_position)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_tile = 0
        correct_rotation = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Calcular losses
                loss_tile = self.tile_type_criterion(
                    outputs['tile_type'],
                    labels['tile_type'].to(self.device)
                )
                loss_rotation = self.rotation_criterion(
                    outputs['rotation'],
                    labels['rotation'].to(self.device)
                )
                loss_has_meeple = self.has_meeple_criterion(
                    outputs['has_meeple'].squeeze(),
                    labels['has_meeple'].to(self.device)
                )
                loss_position = self.meeple_position_criterion(
                    outputs['meeple_position'],
                    labels['meeple_position'].to(self.device)
                )
                
                loss = (2.0 * loss_tile + 1.5 * loss_rotation + 
                       1.0 * loss_has_meeple + 1.0 * loss_position)
                
                total_loss += loss.item()
                
                # Calcular accuracy
                _, predicted_tile = torch.max(outputs['tile_type'], 1)
                _, predicted_rotation = torch.max(outputs['rotation'], 1)
                
                correct_tile += (predicted_tile == labels['tile_type'].to(self.device)).sum().item()
                correct_rotation += (predicted_rotation == labels['rotation'].to(self.device)).sum().item()
                total += labels['tile_type'].size(0)
        
        avg_loss = total_loss / len(val_loader)
        tile_acc = 100 * correct_tile / total
        rotation_acc = 100 * correct_rotation / total
        
        return avg_loss, tile_acc, rotation_acc
    
    def train(self, train_loader, val_loader, epochs=50):
        print("Iniciando entrenamiento...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, tile_acc, rotation_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['tile_type_acc'].append(tile_acc)
            self.history['rotation_acc'].append(rotation_acc)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Tile Accuracy: {tile_acc:.2f}%")
            print(f"  Rotation Accuracy: {rotation_acc:.2f}%")
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_carcassonne_model.pth')
                print("  ✓ Modelo guardado")
            
            print()
    
    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['tile_type_acc'], label='Tile Type Acc')
        axes[1].plot(self.history['rotation_acc'], label='Rotation Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training History - Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


class CarcassonnePredictor:
    """Predictor para nuevas losetas"""
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CarcassonneCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict:
        """Predice características de una loseta"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            # Obtener predicciones
            tile_type = torch.argmax(outputs['tile_type'], dim=1).item()
            rotation = torch.argmax(outputs['rotation'], dim=1).item()
            has_meeple = (outputs['has_meeple'].item() > 0.5)
            meeple_position = torch.argmax(outputs['meeple_position'], dim=1).item() - 1
            
            # Probabilidades
            tile_type_prob = torch.softmax(outputs['tile_type'], dim=1)[0][tile_type].item()
            rotation_prob = torch.softmax(outputs['rotation'], dim=1)[0][rotation].item()
            
            result = {
                'tile_type': tile_type,
                'tile_type_confidence': tile_type_prob,
                'rotation': rotation * 90,  # Convertir a grados
                'rotation_confidence': rotation_prob,
                'has_meeple': has_meeple,
                'meeple_confidence': outputs['has_meeple'].item(),
                'meeple_position': meeple_position if has_meeple else -1
            }
            
            # NUEVO: Agregar letra si tenemos mapper
            if MAPPER:
                result['tile_letter'] = MAPPER.idx_to_letter(tile_type)
            
            return result
    
    def predict_batch(self, tiles_dir: str) -> List[Dict]:
        """Predice un lote de losetas"""
        tiles_path = Path(tiles_dir)
        results = []
        
        for tile_file in sorted(tiles_path.glob('*.png')):
            result = self.predict(str(tile_file))
            result['filename'] = tile_file.name
            results.append(result)
            
            print(f"{tile_file.name}:")
            if MAPPER and 'tile_letter' in result:
                print(f"  Tipo: {result['tile_letter']} (índice {result['tile_type']}) ({result['tile_type_confidence']:.2%})")
            else:
                print(f"  Tipo: {result['tile_type']} ({result['tile_type_confidence']:.2%})")
            print(f"  Rotación: {result['rotation']}° ({result['rotation_confidence']:.2%})")
            print(f"  Tiene ficha: {'Sí' if result['has_meeple'] else 'No'}")
            if result['has_meeple']:
                print(f"  Posición ficha: {result['meeple_position']}")
            print()
        
        return results
    
    def visualize_prediction(self, image_path: str):
        """Visualiza una predicción"""
        result = self.predict(image_path)
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crear figura
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')
        
        # Texto con predicción
        if MAPPER and 'tile_letter' in result:
            text = f"Tipo: {result['tile_letter']} ({result['tile_type_confidence']:.1%})\n"
        else:
            text = f"Tipo: {result['tile_type']} ({result['tile_type_confidence']:.1%})\n"
        
        text += f"Rotación: {result['rotation']}° ({result['rotation_confidence']:.1%})\n"
        text += f"Ficha: {'Sí' if result['has_meeple'] else 'No'}"
        if result['has_meeple']:
            text += f" en pos. {result['meeple_position']}"
        
        ax.text(10, 30, text, color='white', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()


def create_data_transforms(augment=True):
    """Crea transformaciones para el dataset"""
    if augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_annotation_template(output_file='annotations_template.json'):
    """Crea plantilla de anotaciones para facilitar el etiquetado"""
    
    if MAPPER:
        # Plantilla con letras
        template = [
            {
                "image_path": "tiles/tile_000_r0_c0.png",
                "tile_letter": "A",  # Usar letras
                "rotation": 0,
                "has_meeple": False,
                "meeple_position": -1,
                "meeple_color": "none"
            }
        ]
        
        with open(output_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Plantilla creada en {output_file}")
        print("\nInstrucciones:")
        print("1. Duplica la estructura para cada loseta")
        print("2. tile_letter: A-X o 'blanco'")
        print("3. rotation: 0=0°, 1=90°, 2=180°, 3=270°")
        print("4. has_meeple: true/false")
        print("5. meeple_position: 0-8 (posición en la loseta), -1 si no hay")
        print("6. meeple_color: 'red', 'blue', 'green', 'yellow', 'black', 'none'")
        print("\nLas letras se convertirán automáticamente a índices durante el entrenamiento")
    else:
        # Plantilla tradicional con números
        template = [
            {
                "image_path": "tiles/tile_000_r0_c0.png",
                "tile_type": 0,
                "rotation": 0,
                "has_meeple": False,
                "meeple_position": -1,
                "meeple_color": "none"
            }
        ]
        
        with open(output_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Plantilla creada en {output_file}")
        print("\nInstrucciones:")
        print("1. Duplica la estructura para cada loseta")
        print("2. tile_type: 0-24 (según tu catálogo)")
        print("3. rotation: 0=0°, 1=90°, 2=180°, 3=270°")
        print("4. has_meeple: true/false")
        print("5. meeple_position: 0-8 (posición en la loseta), -1 si no hay")
        print("6. meeple_color: 'red', 'blue', 'green', 'yellow', 'black', 'none'")


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("SISTEMA DE RECONOCIMIENTO CNN PARA CARCASSONNE")
    print("="*60)
    
    if MAPPER:
        print(f"\n✓ Sistema de mapeo de letras activado")
        print(f"  Tipos de losetas: {NUM_TILE_TYPES} (A-X + blanco)")
    else:
        print(f"\n⚠️  Sistema tradicional (sin mapeo de letras)")
        print(f"  Tipos de losetas: {NUM_TILE_TYPES}")
    
    print("\nPasos siguientes:")
    print("1. Ejecuta create_annotation_template() para crear plantilla")
    print("2. Anota tus losetas en train_annotations.json y val_annotations.json")
    print("3. Entrena el modelo descomentando la sección de entrenamiento")
    print("4. Usa el predictor para reconocer nuevas losetas")
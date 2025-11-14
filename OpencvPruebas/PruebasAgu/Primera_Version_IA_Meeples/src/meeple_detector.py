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


@dataclass
class MeepleAnnotation:
    """Anotación de una loseta para detección de meeples azules/negros"""
    image_path: str
    has_blue_or_black_meeple: bool  # True si tiene meeple azul o negro
    meeple_position: int  # 0-8 si tiene, -1 si no tiene
    meeple_color: Optional[str] = None  # 'blue' o 'black' si tiene


class MeepleDataset(Dataset):
    """Dataset para detección de meeples azules/negros en losetas de Carcassonne"""

    def __init__(self, annotations_file: str, transform=None):
        # Obtener directorio base para resolver rutas relativas
        annotations_path = Path(annotations_file)
        base_dir = annotations_path.parent

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        self.annotations = []
        skipped = 0
        for item in data:
            # Convertir ruta relativa a absoluta
            if not Path(item['image_path']).is_absolute():
                item['image_path'] = str(base_dir / item['image_path'])

            # Verificar que el archivo existe antes de añadirlo
            if Path(item['image_path']).exists():
                self.annotations.append(MeepleAnnotation(**item))
            else:
                skipped += 1
                print(f"⚠️  Archivo no encontrado, omitiendo: {item['image_path']}")

        self.transform = transform

        print(f"Dataset cargado: {len(self.annotations)} losetas")
        if skipped > 0:
            print(f"⚠️  {skipped} archivos no encontrados y omitidos")

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
            'has_meeple': torch.tensor(int(ann.has_blue_or_black_meeple), dtype=torch.float32),
            'meeple_position': torch.tensor(ann.meeple_position + 1, dtype=torch.long)  # 0=no meeple, 1-9=positions 0-8
        }

        return image, labels


class MeepleCNN(nn.Module):
    """Red neuronal para detección de meeples azules/negros"""

    def __init__(self, num_positions=10):  # 0=no meeple, 1-9=positions 0-8
        super(MeepleCNN, self).__init__()

        # Backbone: ResNet18 pre-entrenado
        backbone = models.resnet18(pretrained=True)

        # Extraer feature extractor (sin la capa final)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Dimensión de features
        feature_dim = 512

        # Heads para cada tarea
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

        print("Modelo MeepleCNN inicializado")

    def forward(self, x):
        # Extraer features
        features = self.features(x)
        features = features.view(features.size(0), -1)

        # Predicciones
        has_meeple = self.has_meeple_head(features)
        meeple_position = self.meeple_position_head(features)

        return {
            'has_meeple': has_meeple,
            'meeple_position': meeple_position
        }


class MeepleTrainer:
    """Entrenador del modelo de meeples"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        # Loss functions
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
            'has_meeple_acc': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Calcular losses
            loss_has_meeple = self.has_meeple_criterion(
                outputs['has_meeple'].squeeze(),
                labels['has_meeple'].to(self.device)
            )

            loss_position = self.meeple_position_criterion(
                outputs['meeple_position'],
                labels['meeple_position'].to(self.device)
            )

            # Loss total (ponderado)
            loss = 1.0 * loss_has_meeple + 1.0 * loss_position

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
        correct_has_meeple = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)

                # Calcular losses
                loss_has_meeple = self.has_meeple_criterion(
                    outputs['has_meeple'].squeeze(),
                    labels['has_meeple'].to(self.device)
                )
                loss_position = self.meeple_position_criterion(
                    outputs['meeple_position'],
                    labels['meeple_position'].to(self.device)
                )

                loss = 1.0 * loss_has_meeple + 1.0 * loss_position
                total_loss += loss.item()

                # Calcular accuracy para has_meeple
                predicted_has_meeple = (outputs['has_meeple'].squeeze() > 0.5).float()
                correct_has_meeple += (predicted_has_meeple == labels['has_meeple'].to(self.device)).sum().item()
                total += labels['has_meeple'].size(0)

        avg_loss = total_loss / len(val_loader)
        has_meeple_acc = 100 * correct_has_meeple / total

        return avg_loss, has_meeple_acc

    def train(self, train_loader, val_loader, epochs=50):
        print("Iniciando entrenamiento del modelo de meeples...")
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, has_meeple_acc = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['has_meeple_acc'].append(has_meeple_acc)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Has Meeple Accuracy: {has_meeple_acc:.2f}%")

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_meeple_model.pth')
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
        axes[1].plot(self.history['has_meeple_acc'], label='Has Meeple Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training History - Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('output/training_history.png')
        plt.show()


class MeeplePredictor:
    """Predictor para detectar meeples en nuevas losetas"""

    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = MeepleCNN()
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
        """Predice si una loseta tiene meeple azul/negro y su posición"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

            # Obtener predicciones
            has_meeple = (outputs['has_meeple'].item() > 0.5)
            meeple_position = torch.argmax(outputs['meeple_position'], dim=1).item() - 1

            result = {
                'has_blue_or_black_meeple': has_meeple,
                'meeple_confidence': outputs['has_meeple'].item(),
                'meeple_position': meeple_position if has_meeple else -1
            }

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
            print(f"  Tiene meeple azul/negro: {'Sí' if result['has_blue_or_black_meeple'] else 'No'} ({result['meeple_confidence']:.2%})")
            if result['has_blue_or_black_meeple']:
                print(f"  Posición: {result['meeple_position']}")
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
        text = f"Meeple azul/negro: {'Sí' if result['has_blue_or_black_meeple'] else 'No'} ({result['meeple_confidence']:.1%})"
        if result['has_blue_or_black_meeple']:
            text += f"\nPosición: {result['meeple_position']}"

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


def create_annotation_template(output_file='data/annotations_template.json'):
    """Crea plantilla de anotaciones para facilitar el etiquetado"""

    template = [
        {
            "image_path": "data/tiles/tile_000.png",
            "has_blue_or_black_meeple": False,
            "meeple_position": -1,
            "meeple_color": None
        }
    ]

    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Plantilla creada en {output_file}")
    print("\nInstrucciones:")
    print("1. Duplica la estructura para cada loseta")
    print("2. has_blue_or_black_meeple: true si tiene meeple azul o negro, false si no")
    print("3. meeple_position: 0-8 (posición en la loseta dividida en 9 subespacios), -1 si no hay meeple")
    print("4. meeple_color: 'blue' o 'black' si tiene meeple, null si no")
    print("\nLos subespacios se numeran de izquierda a derecha, arriba a abajo:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("SISTEMA DE DETECCIÓN DE MEEPLES AZULES/NEGROS")
    print("="*60)
    
    # Crear plantilla de anotaciones
    create_annotation_template()
    
    print("\nPasos siguientes:")
    print("1. Coloca tus imágenes de losetas en data/tiles/")
    print("2. Anota tus losetas en data/train_annotations.json y data/val_annotations.json")
    print("3. Entrena el modelo descomentando la sección de entrenamiento")
    print("4. Usa el predictor para detectar meeples en nuevas losetas")
#!/usr/bin/env python3
"""
Detector de Meeples usando CNN (Redes Convolucionales)
Entrena un modelo CNN usando las anotaciones manuales como ground truth
"""

import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MeepleDataset(Dataset):
    """Dataset para entrenamiento de CNN con meeples"""

    def __init__(self, image_paths: List[str], annotations: Dict, patch_size: int = 64):
        self.image_paths = image_paths
        self.annotations = annotations
        self.patch_size = patch_size
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """Preparar muestras de entrenamiento desde anotaciones"""
        samples = []

        for img_path in self.image_paths:
            if img_path not in self.annotations:
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]

            # Para cada anotación, extraer patch alrededor del meeple
            for ann in self.annotations[img_path]:
                pixel_x, pixel_y = ann['pixel_coords']

                # Extraer patch centrado en el meeple
                half_size = self.patch_size // 2
                x1 = max(0, pixel_x - half_size)
                y1 = max(0, pixel_y - half_size)
                x2 = min(w, pixel_x + half_size)
                y2 = min(h, pixel_y + half_size)

                # Ajustar si el patch está en el borde
                if x2 - x1 < self.patch_size:
                    if x1 == 0:
                        x2 = min(w, x1 + self.patch_size)
                    else:
                        x1 = max(0, x2 - self.patch_size)

                if y2 - y1 < self.patch_size:
                    if y1 == 0:
                        y2 = min(h, y1 + self.patch_size)
                    else:
                        y1 = max(0, y2 - self.patch_size)

                patch = image[y1:y2, x1:x2]

                # Resize si es necesario
                if patch.shape[:2] != (self.patch_size, self.patch_size):
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))

                # Convertir a RGB y normalizar
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_normalized = patch_rgb.astype(np.float32) / 255.0

                # Label: 0=blue, 1=black
                label = 0 if ann['color'] == 'blue' else 1

                samples.append({
                    'image': patch_normalized,
                    'label': label,
                    'position': ann['position'],
                    'original_coords': (pixel_x, pixel_y)
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Convertir a tensor (C, H, W)
        image_tensor = torch.from_numpy(sample['image'].transpose(2, 0, 1))
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)

        return image_tensor, label_tensor, sample['position']

class MeepleCNN(nn.Module):
    """CNN simple para clasificación de meeples"""

    def __init__(self, num_classes: int = 2):
        super(MeepleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNNMeepleDetector:
    """Detector de meeples usando CNN"""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MeepleCNN().to(self.device)
        self.patch_size = 64

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"📂 Modelo CNN cargado desde: {model_path}")
        else:
            print("⚠️ Modelo CNN no encontrado, usarás solo OpenCV")

    def train(self, annotations_file: str, epochs: int = 20, batch_size: int = 8):
        """Entrenar el modelo CNN"""
        # Cargar anotaciones
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        if not annotations:
            print("❌ No hay anotaciones para entrenar")
            return

        # Preparar dataset
        image_paths = list(annotations.keys())
        dataset = MeepleDataset(image_paths, annotations, self.patch_size)

        if len(dataset) == 0:
            print("❌ No se pudieron extraer patches de entrenamiento")
            return

        # Split train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss y optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print(f"🚀 Entrenando CNN con {len(train_dataset)} muestras de entrenamiento")
        print(f"   Validación: {len(val_dataset)} muestras")
        print("=" * 50)

        best_val_acc = 0.0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_acc = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)

                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_acc = 100. * val_correct / val_total

            print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_meeple_cnn.pth')
                print(f"   💾 Mejor modelo guardado (acc: {val_acc:.2f}%)")

        print("
✅ Entrenamiento completado!"        print(f"Mejor accuracy de validación: {best_val_acc:.2f}%")

        # Mostrar métricas finales
        if val_labels:
            print("\n📊 Reporte de clasificación:")
            print(classification_report(val_labels, val_preds, target_names=['blue', 'black']))

    def predict_patch(self, patch: np.ndarray) -> str:
        """Predecir color de un patch usando CNN"""
        if not hasattr(self, 'model') or self.model is None:
            return 'unknown'

        # Preprocesar patch
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_resized = cv2.resize(patch_rgb, (self.patch_size, self.patch_size))
        patch_normalized = patch_resized.astype(np.float32) / 255.0

        # Convertir a tensor
        tensor = torch.from_numpy(patch_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # Predecir
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = outputs.max(1)
            class_idx = predicted.item()

        return 'blue' if class_idx == 0 else 'black'

def main():
    """Función principal"""
    print("🧠 Detector de Meeples con CNN")
    print("=" * 40)

    # Verificar si hay anotaciones
    annotations_file = 'manual_annotations.json'
    if not Path(annotations_file).exists():
        print(f"❌ No hay anotaciones. Ejecuta primero: python meeple_annotator.py")
        return

    # Verificar si hay GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")

    # Menú de opciones
    while True:
        print("\nOpciones:")
        print("1. Entrenar modelo CNN")
        print("2. Evaluar modelo existente")
        print("3. Comparar CNN vs OpenCV")
        print("4. Salir")

        choice = input("Elige una opción (1-4): ").strip()

        if choice == '1':
            # Entrenar
            epochs = int(input("Número de epochs (default 20): ") or "20")
            batch_size = int(input("Batch size (default 8): ") or "8")

            detector = CNNMeepleDetector()
            detector.train(annotations_file, epochs=epochs, batch_size=batch_size)

        elif choice == '2':
            # Evaluar
            model_path = 'best_meeple_cnn.pth'
            if not Path(model_path).exists():
                print(f"❌ Modelo no encontrado: {model_path}")
                continue

            detector = CNNMeepleDetector(model_path)

            # Cargar algunas imágenes de prueba
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)

            test_images = list(annotations.keys())[:5]  # Primeras 5

            print("🔍 Evaluando modelo en imágenes de prueba...")
            for img_path in test_images:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Extraer patches de meeples conocidos
                for ann in annotations[img_path]:
                    pixel_x, pixel_y = ann['pixel_coords']
                    half_size = 32
                    x1 = max(0, pixel_x - half_size)
                    y1 = max(0, pixel_y - half_size)
                    x2 = min(image.shape[1], pixel_x + half_size)
                    y2 = min(image.shape[0], pixel_y + half_size)

                    patch = image[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue

                    predicted_color = detector.predict_patch(patch)
                    true_color = ann['color']

                    status = "✅" if predicted_color == true_color else "❌"
                    print(f"  {Path(img_path).name}: GT={true_color}, Pred={predicted_color} {status}")

        elif choice == '3':
            print("🔄 Comparación CNN vs OpenCV próximamente...")

        elif choice == '4':
            break

        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()
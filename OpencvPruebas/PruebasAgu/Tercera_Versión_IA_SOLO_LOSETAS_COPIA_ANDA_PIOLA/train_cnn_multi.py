#!/usr/bin/env python3
"""
Script para entrenar el modelo CNN con múltiples imágenes por clase
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from carcassonne_cnn import CarcassonneCNN
from sklearn.model_selection import train_test_split

class CarcassonneMultiImageDataset(Dataset):
    """Dataset personalizado para múltiples imágenes por clase"""

    def __init__(self, reference_folder, transform=None):
        self.reference_folder = Path(reference_folder)
        self.transform = transform
        self.data = []

        # Mapear letras a índices (A-X = 0-23, BLANCO = 24)
        self.letter_to_idx = {chr(65 + i): i for i in range(24)}  # A=0, B=1, ..., X=23
        self.letter_to_idx['BLANCO'] = 24

        # Cargar todas las imágenes de cada carpeta (PNG y JPG)
        for letter in self.letter_to_idx.keys():
            letter_folder = self.reference_folder / letter
            if letter_folder.exists():
                # Buscar todas las imágenes PNG y JPG en la carpeta
                for img_file in letter_folder.glob("*.png"):
                    self.data.append((str(img_file), self.letter_to_idx[letter]))
                for img_file in letter_folder.glob("*.jpg"):
                    self.data.append((str(img_file), self.letter_to_idx[letter]))

        print(f"Dataset creado con {len(self.data)} imágenes de {len(self.letter_to_idx)} clases")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Si la imagen no se puede cargar, devolver una imagen negra
            image = np.zeros((64, 64, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model_multi():
    """Entrenar el modelo CNN con múltiples imágenes por clase"""

    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Transformaciones con data augmentation para ResNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # ResNet espera 224x224
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset y DataLoader con split train/val
    dataset = CarcassonneMultiImageDataset('referencias_organizadas', transform=transform)
    
    # Dividir en train y validation (80/20)
    train_data, val_data = train_test_split(dataset.data, test_size=0.2, random_state=42, stratify=[label for _, label in dataset.data])
    
    # Crear subsets
    train_dataset = torch.utils.data.Subset(dataset, [dataset.data.index(item) for item in train_data])
    val_dataset = torch.utils.data.Subset(dataset, [dataset.data.index(item) for item in val_data])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reducido para test rápido
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Modelo
    model = CarcassonneCNN(num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Aumentado para fine-tuning

    # Entrenamiento con validación
    num_epochs = 10  # Reducido para test rápido
    best_val_acc = 0.0
    patience = 15  # Aumentado para más paciencia
    patience_counter = 0
    
    print(f"Entrenando modelo con {len(train_dataset)} imágenes de train y {len(val_dataset)} de val...")

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'carcassonne_cnn_multi_model_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Cargar mejor modelo y guardar
    model.load_state_dict(torch.load('carcassonne_cnn_multi_model_best.pth'))
    torch.save(model.state_dict(), 'carcassonne_cnn_multi_model.pth')
    print(f"Modelo guardado como 'carcassonne_cnn_multi_model.pth' con val acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model_multi()
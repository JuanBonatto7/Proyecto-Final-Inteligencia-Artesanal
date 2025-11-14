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

class CarcassonneMultiImageDataset(Dataset):
    """Dataset personalizado para múltiples imágenes por clase"""

    def __init__(self, reference_folder, transform=None):
        self.reference_folder = Path(reference_folder)
        self.transform = transform
        self.data = []

        # Mapear letras a índices
        self.letter_to_idx = {chr(65 + i): i for i in range(24)}  # A=0, B=1, ..., X=23

        # Cargar todas las imágenes de cada carpeta
        for letter in self.letter_to_idx.keys():
            letter_folder = self.reference_folder / letter
            if letter_folder.exists():
                # Buscar todas las imágenes PNG en la carpeta
                for img_file in letter_folder.glob("*.png"):
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

    # Transformaciones con data augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),  # Rotación aleatoria
        transforms.RandomHorizontalFlip(),  # Flip horizontal
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variación de color
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset y DataLoader
    dataset = CarcassonneMultiImageDataset('referencias_organizadas', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Batch size mayor

    # Modelo
    model = CarcassonneCNN(num_classes=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento
    num_epochs = 100  # Más epochs para más datos
    print(f"Entrenando modelo con {len(dataset)} imágenes...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), 'carcassonne_cnn_multi_model.pth')
    print("Modelo guardado como 'carcassonne_cnn_multi_model.pth'")

if __name__ == "__main__":
    train_model_multi()
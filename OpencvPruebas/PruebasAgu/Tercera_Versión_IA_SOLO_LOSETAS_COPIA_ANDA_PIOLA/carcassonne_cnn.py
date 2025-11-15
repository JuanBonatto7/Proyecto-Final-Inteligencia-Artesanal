import torch
import torch.nn as nn
import torch.nn.functional as F

class CarcassonneCNN(nn.Module):
    """Modelo CNN simple para clasificar losetas de Carcassonne"""

    def __init__(self, num_classes=24):
        super(CarcassonneCNN, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Nueva capa

        # Capas fully connected
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Ajustado por la nueva capa
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Capas convolucionales con max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))  # Nueva capa
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(-1, 256 * 4 * 4)  # Ajustado

        # Capas fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
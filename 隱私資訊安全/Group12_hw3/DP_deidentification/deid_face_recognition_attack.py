"""
deid_face_recognition_attack.py

Description:
This script trains a ResNet18 convolutional neural network on original face images

Dependencies:
- torch
- torchvision
- PIL (via torchvision)
- numpy
- CUDA (if available)

Output:
- Saves the model to 'saved_models/original_model.pth'
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# ========== Setup ==========
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "original")
TEST_DIRS = {
    "Original": os.path.join(DATA_DIR, "original"),
    "Blurred": os.path.join(DATA_DIR, "blurred"),
    "Pixelated": os.path.join(DATA_DIR, "pixelated")
}
BATCH_SIZE = 32
NUM_EPOCHS = 15
MODEL_PATH = os.path.join("saved_models", "face_recognition_plain_resnet18.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Ensure Save Folder Exists ==========
os.makedirs("saved_models", exist_ok=True)

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ========== Training Function ==========
def train():
    print("Training model on original dataset...")
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(train_dataset.classes)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {total_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# ========== Main Execution ==========
if __name__ == "__main__":
    # Uncomment to train and save the model
    train()

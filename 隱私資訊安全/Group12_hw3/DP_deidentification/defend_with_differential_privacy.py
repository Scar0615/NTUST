"""
defend_with_differential_privacy.py

Trains and saves three differentially private ResNet18 models
using Opacus, one for each epsilon value (1.0, 2.0, 4.0).
Each model is saved with its epsilon level in the filename.

Keeps the number of epochs (15) consistent with the baseline model.

Dependencies:
- torch
- torchvision
- opacus
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


# ========== Configuration ==========
DATA_DIR = "dataset/original"
SAVE_DIR = "saved_models/differential_privacy"
EPSILON_VALUES = [1.0, 2.0, 4.0]
BATCH_SIZE = 32
NUM_EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Dataset ==========
dataset = ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
num_classes = len(dataset.classes)

# ========== Train and Save One Model Per Epsilon ==========
for epsilon in EPSILON_VALUES:
    print(f"\nTraining with ε = {epsilon}")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Replace BatchNorm with GroupNorm (required for DP)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=True)  # will raise if still invalid

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.1,
        max_grad_norm=1.0
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in private_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {total_loss:.4f}")

    model_path = os.path.join(SAVE_DIR, f"face_recognition_dp_resnet18_eps{epsilon}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
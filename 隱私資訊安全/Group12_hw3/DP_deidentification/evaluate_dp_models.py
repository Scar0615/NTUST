"""
evaluate_dp_models.py

Evaluates all DP-trained ResNet18 models saved in 'saved_models/differential_privacy/'
on original, blurred, and pixelated face datasets. Models must be named
with their epsilon values (e.g., face_recognition_dp_resnet18_eps1.0.pth).

Dependencies:
- torch
- torchvision
- opacus
- PIL
"""

import os
import re
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from opacus.validators import ModuleValidator

# ========== Configuration ==========
DATA_DIR = "dataset"
MODEL_DIR = "saved_models/differential_privacy"
TEST_FOLDERS = [
    "original",
    "pixelated_b4", "pixelated_b8", "pixelated_b16",
    "blurred_k15", "blurred_k45", "blurred_k99"
]
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Transforms ==========
# Preprocessing steps applied to all images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Evaluation Function ==========
def evaluate(model, test_path):
    """Evaluates a model on a given dataset folder and returns accuracy."""
    dataset = ImageFolder(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total > 0 else 0

# ========== Helper to Extract Epsilon from Filename ==========
def extract_epsilon(filename):
    match = re.search(r"eps([0-9]+(?:\.[0-9]+)?)\.pth", filename)
    if match:
        return float(match.group(1))
    else:
        return float('inf')  # push bad filenames to the end

# ========== Run Evaluation for All Models ==========
print("\nEvaluating all DP models on all test variants:")

# Determine number of output classes based on training set
ref_dataset = ImageFolder(os.path.join(DATA_DIR, "original"))
num_classes = len(ref_dataset.classes)

# Get all model files
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("face_recognition_dp_resnet18_eps")]
model_files.sort(key=extract_epsilon)

for model_file in model_files:
    eps = re.search(r"eps([0-9]+(?:\.[0-9]+)?)", model_file).group(1)
    print(f"\n--- Evaluation for ε = {eps} ---")
    model_path = os.path.join(MODEL_DIR, model_file)

    # Load model architecture and replace BatchNorm with GroupNorm
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = ModuleValidator.fix(model)

    # Load DP-trained model weights and remove '_module.' prefix
    state_dict = torch.load(model_path, map_location=DEVICE)
    state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Evaluate on all datasets
    for folder in TEST_FOLDERS:
        test_path = os.path.join(DATA_DIR, folder)
        acc = evaluate(model, test_path)
        print(f"{folder:<15}: {acc:.2f}%")

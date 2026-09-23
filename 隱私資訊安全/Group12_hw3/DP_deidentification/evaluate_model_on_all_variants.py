"""
evaluate_model_on_all_variants.py

Description:
Loads the baseline face recognition model and evaluates it on all
pixelated and blurred variants.

Assumes the model is saved as 'face_recognition_plain_resnet18.pth'
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

# ========== Config ==========
DATA_DIR = "dataset"
MODEL_PATH = os.path.join("saved_models", "face_recognition_plain_resnet18.pth")
TEST_FOLDERS = [
    "original",
    "pixelated_b4", "pixelated_b8", "pixelated_b16",
    "blurred_k15", "blurred_k45", "blurred_k99"
]
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Load Model ==========
def load_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ========== Transform ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Evaluate ==========
def evaluate_folder(folder_name, model):
    path = os.path.join(DATA_DIR, folder_name)
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


# ========== Run Evaluation ==========
print("Evaluating model on all de-identified datasets:\n")

# Get number of classes from original folder
ref_dataset = ImageFolder(os.path.join(DATA_DIR, "original"))
num_classes = len(ref_dataset.classes)

model = load_model(num_classes)

results = []
for folder in TEST_FOLDERS:
    acc = evaluate_folder(folder, model)
    results.append((folder, acc))

# Print nicely
print(f"{'Dataset':<20}Accuracy (%)")
print("-" * 35)
for folder, acc in results:
    print(f"{folder:<20}{acc:.2f}")

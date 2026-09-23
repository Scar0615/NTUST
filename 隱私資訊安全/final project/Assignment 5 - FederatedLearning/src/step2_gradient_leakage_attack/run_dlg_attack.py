# src/step2_gradient_leakage_attack/run_dlg_attack.py

# --- Deep Leakage from Gradients (DLG) Attack Implementation ---
# This script attempts to reconstruct a training image from shared gradients
# by performing optimization to match dummy input gradients with observed gradients.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ✅ Add parent folder to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import model and utilities
from src.step1_federated_learning.model import FederatedModel
from src.step2_gradient_leakage_attack.utils import gradient_matching_loss, save_comparison

# ✅ Total Variation Regularization to improve smoothness of the dummy image
def total_variation(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

def run_dlg_attack():
    # --- Path configuration ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "extracted_gradients"))
    gradient_path = os.path.join(base_dir, "gradients_client0.pt")
    original_img_path = os.path.join(base_dir, "original_input.pt")
    original_lbl_path = os.path.join(base_dir, "original_label.pt")
    model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "step1_federated_learning", "federated_model_final.pth"
    ))
    output_path = os.path.join(base_dir, "comparison.png")

    # --- Constants ---
    num_classes = 258                # Number of output classes in the model
    image_shape = (3, 224, 224)      # Input image shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load real gradients and true label ---
    real_grads = [g.to(device) for g in torch.load(gradient_path)]
    true_label = torch.load(original_lbl_path)[0].unsqueeze(0).to(device)

    # Initialize dummy image with random noise
    dummy_data = torch.randn((1, *image_shape), requires_grad=True, device=device)

    # --- Model setup ---
    model = FederatedModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()

    # Normalization constants (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # --- Optimization setup ---
    optimizer = torch.optim.Adam([dummy_data], lr=0.1)

    # --- DLG Optimization Loop ---
    for iteration in range(3000):
        optimizer.zero_grad()

        # Normalize dummy input before feeding to model
        normalized_input = (dummy_data - mean) / std
        pred = model(normalized_input)

        # Compute gradient w.r.t. model parameters using dummy input
        loss = nn.CrossEntropyLoss()(pred, true_label)
        dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Compute difference between dummy and real gradients
        grad_loss = gradient_matching_loss(dummy_grads, real_grads)

        # Add Total Variation penalty to encourage smooth image
        tv_loss = 1e-5 * total_variation(dummy_data)
        total_loss = grad_loss + tv_loss

        # Check for unstable gradients
        if torch.isnan(total_loss):
            print(f"[{iteration}] ❌ Loss became NaN. Stopping early.")
            break

        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([dummy_data], max_norm=10.0)
        optimizer.step()

        # Clamp pixel values to valid range [0, 1]
        with torch.no_grad():
            dummy_data.clamp_(0, 1)

        # Logging progress
        if iteration % 50 == 0 or iteration == 999:
            print(f"[{iteration}] Total loss: {total_loss.item():.6f} | Grad loss: {grad_loss.item():.6f} | TV: {tv_loss.item():.6f}")

    # ✅ Save reconstructed image tensor
    torch.save(dummy_data.detach(), os.path.join(base_dir, "reconstructed_image.pt"))

    # ✅ If original image exists, generate a side-by-side visual comparison
    if os.path.exists(original_img_path):
        original = torch.load(original_img_path)[0].detach().cpu()
        reconstructed = dummy_data.detach().cpu()[0]
        save_comparison(original, reconstructed, output_path)

    print("✅ Gradient leakage attack complete.")

if __name__ == "__main__":
    run_dlg_attack()

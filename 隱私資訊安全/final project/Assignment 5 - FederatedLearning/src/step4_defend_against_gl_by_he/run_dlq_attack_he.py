# src/step4_defend_against_gl_by_he/run_dlg_attack_he.py

# 🔓 Attempt to reconstruct original image from gradients under Homomorphic Encryption
# This demonstrates that even if gradients are captured, HE makes reconstruction difficult or ineffective

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to the path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.step1_federated_learning.model import FederatedModel
from src.step2_gradient_leakage_attack.utils import gradient_matching_loss, save_comparison

# --- Total Variation Loss ---
def total_variation(x):
    """
    Computes Total Variation (TV) loss for smoothness regularization.
    Encourages adjacent pixels to have similar values.
    """
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def run_dlg_attack_he():
    print("\n🔓 Attempting DLG attack on HE-protected gradients...")

    # --- Paths for input and output ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "extracted_gradients"))
    gradient_path = os.path.join(base_dir, "he_gradients_client0.pt")
    original_img_path = os.path.join(base_dir, "he_original_input.pt")
    original_lbl_path = os.path.join(base_dir, "he_original_label.pt")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "step1_federated_learning", "federated_model_final.pth"))
    output_path = os.path.join(base_dir, "he_comparison.png")

    # --- Constants ---
    num_classes = 258
    image_shape = (3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load encrypted (simulated) gradients and label ---
    real_grads = [g.to(device) for g in torch.load(gradient_path)]
    true_label = torch.load(original_lbl_path)[0].unsqueeze(0).to(device)

    # --- Initialize dummy input image for optimization ---
    dummy_data = torch.randn((1, *image_shape), requires_grad=True, device=device)

    # --- Load the global model ---
    model = FederatedModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()

    # --- Optimizer and normalization stats (ImageNet) ---
    optimizer = torch.optim.Adam([dummy_data], lr=0.05)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # --- Optimization loop to match dummy gradient with true gradient ---
    for iteration in range(3000):
        optimizer.zero_grad()

        # Normalize the dummy image before passing to the model
        normalized_input = (dummy_data - mean) / std
        pred = model(normalized_input)

        # Compute gradient matching loss using true label
        loss = nn.CrossEntropyLoss()(pred, true_label)
        dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_loss = gradient_matching_loss(dummy_grads, real_grads)

        # Add Total Variation loss for image smoothness
        tv_loss = 1e-5 * total_variation(dummy_data)
        total_loss = grad_loss + tv_loss

        total_loss.backward()
        optimizer.step()

        if iteration % 100 == 0 or iteration == 2999:
            print(f"[{iteration}] Total loss: {total_loss.item():.4f} | Grad loss: {grad_loss.item():.4f} | TV: {tv_loss.item():.4f}")

    # --- Save reconstructed image tensor ---
    torch.save(dummy_data.detach(), os.path.join(base_dir, "he_reconstructed_image.pt"))

    # --- Save visual comparison between original and reconstructed image ---
    if os.path.exists(original_img_path):
        original = torch.load(original_img_path)[0].detach().cpu()
        reconstructed = dummy_data.detach().cpu()[0]
        save_comparison(original, reconstructed, output_path)

    print("🔒 HE DLG attack complete. Image saved.")


# --- Entry point ---
if __name__ == "__main__":
    run_dlg_attack_he()

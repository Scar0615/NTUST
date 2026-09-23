"""
evaluate_image_utility.py

Compares one person's images across all blurred and pixelated variants
against the original using SSIM and MSE. Generates a bar chart for both.

Dependencies:
- numpy
- scikit-image
- matplotlib
- PIL
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage import img_as_float
from sklearn.metrics import mean_squared_error

# === Configuration ===
person_name = "aaron_taylor_johnson"
dataset_root = "dataset"
variants = [
    "pixelated_b4",
    "pixelated_b8",
    "pixelated_b16",
    "blurred_k15",
    "blurred_k45",
    "blurred_k99"
]

original_dir = os.path.join(dataset_root, "original", person_name)

# === Metric Function ===
def compute_metrics(img1, img2):
    img1 = img_as_float(rgb2gray(img1))
    img2 = img_as_float(rgb2gray(img2))
    return ssim(img1, img2, data_range=1.0), mean_squared_error(img1, img2)

# === Evaluation ===
ssim_scores = []
mse_scores = []
labels = []

print(f"\nUtility analysis for '{person_name}':\n")
print(f"{'Variant':<15}  SSIM      MSE")
print("-" * 35)

for variant in variants:
    proc_dir = os.path.join(dataset_root, variant, person_name)
    if not os.path.isdir(proc_dir):
        print(f"{variant:<15}  Folder not found")
        continue

    total_ssim = 0
    total_mse = 0
    count = 0

    for fname in os.listdir(original_dir):
        orig_path = os.path.join(original_dir, fname)
        proc_path = os.path.join(proc_dir, fname)
        if not os.path.exists(proc_path):
            continue

        orig_img = Image.open(orig_path).resize((224, 224)).convert('RGB')
        proc_img = Image.open(proc_path).resize((224, 224)).convert('RGB')

        try:
            ssim_val, mse_val = compute_metrics(np.array(orig_img), np.array(proc_img))
            total_ssim += ssim_val
            total_mse += mse_val
            count += 1
        except Exception as e:
            print(f"Error comparing {fname}: {e}")

    if count > 0:
        avg_ssim = total_ssim / count
        avg_mse = total_mse / count
        print(f"{variant:<15}  {avg_ssim:.4f}   {avg_mse:.4f}")
        labels.append(variant)
        ssim_scores.append(avg_ssim)
        mse_scores.append(avg_mse)
    else:
        print(f"{variant:<15}  No valid image pairs")

# === Generate Charts ===
if labels:
    # SSIM Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, ssim_scores, color="skyblue")
    plt.title(f"Average SSIM per Variant ({person_name})")
    plt.ylabel("SSIM")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig("utility_ssim_scores.png")
    print("\nSaved chart: utility_ssim_scores.png")

    # MSE Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mse_scores, color="salmon")
    plt.title(f"Average MSE per Variant ({person_name})")
    plt.ylabel("MSE")
    plt.grid(axis='y')
    plt.savefig("utility_mse_scores.png")
    print("Saved chart: utility_mse_scores.png")

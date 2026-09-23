# src/step3_defence_against_gla/compare_gradients_visual.py

# This script visualizes the effect of Differential Privacy (DP) noise on gradients
# by plotting grayscale images of gradients and comparing their distributions using histograms.

import os
import torch
import matplotlib.pyplot as plt

def compare_gradients(clean_path, noisy_path, image_save="gradient_noise_comparison.png", hist_save="gradient_histograms.png", max_layers=3):
    """
    Loads clean and noisy gradients from disk, and:
    - Visualizes them side-by-side in grayscale for visual inspection.
    - Plots histograms to show how the value distributions shift with added DP noise.
    """
    clean_grads = torch.load(clean_path)
    noisy_grads = torch.load(noisy_path)

    assert len(clean_grads) == len(noisy_grads), "Mismatch in gradient layers."

    # --- Grayscale visualization section ---
    fig_img, axes_img = plt.subplots(nrows=max_layers, ncols=2, figsize=(8, 3 * max_layers))
    if max_layers == 1:
        axes_img = [axes_img]  # ensure consistent indexing for single row

    for i in range(min(max_layers, len(clean_grads))):
        def flatten_for_plot(g):
            """
            Prepares gradients for grayscale plotting.
            Converts 4D conv layers or 2D FC layers to 2D images.
            """
            g = g.detach().cpu()
            if g.ndim == 4:  # Conv layer
                g = g[0, 0, :, :]  # take one filter channel
            elif g.ndim == 2:
                pass  # already suitable
            else:
                g = g.view(-1, 1)  # fallback for other shapes
            return g

        clean_img = flatten_for_plot(clean_grads[i])
        noisy_img = flatten_for_plot(noisy_grads[i])

        axes_img[i][0].imshow(clean_img, cmap="gray")
        axes_img[i][0].set_title(f"Clean Gradient (Layer {i})")
        axes_img[i][0].axis("off")

        axes_img[i][1].imshow(noisy_img, cmap="gray")
        axes_img[i][1].set_title(f"Noisy Gradient (Layer {i})")
        axes_img[i][1].axis("off")

    fig_img.tight_layout()
    fig_img.savefig(image_save)
    plt.close(fig_img)
    print(f"✅ Saved gradient image comparison to {image_save}")

    # --- Histogram comparison section ---
    fig_hist, axes_hist = plt.subplots(nrows=max_layers, figsize=(8, 3 * max_layers))
    if max_layers == 1:
        axes_hist = [axes_hist]

    # Print summary statistics for each layer
    for i in range(min(max_layers, len(clean_grads))):
        clean = clean_grads[i].detach().cpu()
        noisy = noisy_grads[i].detach().cpu()
        print(f"[Layer {i}] Clean: min={clean.min():.4f}, max={clean.max():.4f}, mean={clean.mean():.4f}")
        print(f"[Layer {i}] Noisy: min={noisy.min():.4f}, max={noisy.max():.4f}, mean={noisy.mean():.4f}")

    # Plot histograms showing value distribution of gradients
    for i in range(min(max_layers, len(clean_grads))):
        clean = clean_grads[i].detach().cpu().flatten().numpy()
        noisy = noisy_grads[i].detach().cpu().flatten().numpy()

        axes_hist[i].hist(clean, bins=100, alpha=0.6, label="Clean", color='blue', density=True)
        axes_hist[i].set_yscale("log")  # log scale to visualize small differences
        axes_hist[i].hist(noisy, bins=100, alpha=0.6, label="Noisy", color='orange', density=True)
        axes_hist[i].set_title(f"Gradient Value Distribution (Layer {i})")
        axes_hist[i].legend()
        axes_hist[i].set_xlim(-0.1, 0.1)

    fig_hist.tight_layout()
    fig_hist.savefig(hist_save)
    plt.close(fig_hist)
    print(f"✅ Saved histogram comparison to {hist_save}")

if __name__ == "__main__":
    # Define file paths for clean and noisy gradients
    clean_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "step2_gradient_leakage_attack", "extracted_gradients", "gradients_client0.pt"
    ))
    noisy_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "extracted_gradients", "dp_gradients_client0.pt"
    ))
    img_save = os.path.join(os.path.dirname(__file__), "gradient_noise_comparison.png")
    hist_save = os.path.join(os.path.dirname(__file__), "gradient_histograms.png")

    # Run visualization
    compare_gradients(clean_path, noisy_path, img_save, hist_save, max_layers=3)

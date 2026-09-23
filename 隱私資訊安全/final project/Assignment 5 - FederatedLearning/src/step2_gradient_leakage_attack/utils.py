# src/step2_gradient_leakage_attack/utils.py

import torch


def gradient_matching_loss(dummy_grads, real_grads):
    """
    Computes the squared L2 loss between gradients of the dummy image
    and the real image used in training. This is the core metric used
    during the optimization process in DLG to align gradients.

    Args:
        dummy_grads (list of torch.Tensor): Gradients from dummy image.
        real_grads (list of torch.Tensor): True gradients from original image.

    Returns:
        loss (torch.Tensor): Scalar value representing gradient difference.
    """
    loss = 0
    for dg, rg in zip(dummy_grads, real_grads):
        loss += ((dg - rg) ** 2).sum()
    return loss


def save_comparison(original, reconstructed, output_path):
    """
    Creates a side-by-side image comparing the original and reconstructed input.
    Used to visually assess the success of the gradient leakage attack.

    Args:
        original (torch.Tensor): Ground-truth input image tensor (C x H x W).
        reconstructed (torch.Tensor): Reconstructed image tensor (C x H x W).
        output_path (str): File path to save the comparison image.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ✅ Handle batched input (use the first image in batch)
    if original.dim() == 4:
        original = original[0]
    if reconstructed.dim() == 4:
        reconstructed = reconstructed[0]

    # Convert from (C, H, W) to (H, W, C) for visualization
    original = original.permute(1, 2, 0).detach().cpu().numpy()
    reconstructed = reconstructed.permute(1, 2, 0).detach().cpu().numpy()

    # ✅ Undo normalization (ImageNet mean and std) to get valid RGB values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original = np.clip((original * std + mean), 0, 1)
    reconstructed = np.clip((reconstructed * std + mean), 0, 1)

    # ✅ Display original vs reconstructed side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

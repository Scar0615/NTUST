"""
generate_deidentified_images.py

Description:
Processes class-organized image datasets from 'dataset/original/',
applying different levels of pixelation and Gaussian blur. Outputs
preserve class (person) subfolders.

Dependencies:
- opencv-python
"""

import cv2
import os
from pathlib import Path

# ========== Parameters ==========
INPUT_DIR = Path("dataset/original")
PIXELATION_LEVELS = [4, 8, 16]
BLUR_LEVELS = [15, 45, 99]

# ========== Processing Functions ==========
def apply_pixelation(image, b):
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w // b, h // b), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_blur(image, k):
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(image, (k, k), 0)

def process_and_save(input_root, output_root, process_func):
    for class_dir in input_root.iterdir():
        if not class_dir.is_dir():
            continue

        output_class_dir = output_root / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipped: {img_path.name}")
                continue

            processed = process_func(img)
            output_path = output_class_dir / img_path.name
            cv2.imwrite(str(output_path), processed)

# ========== Generate Pixelated Versions ==========
for b in PIXELATION_LEVELS:
    out_dir = Path(f"dataset/pixelated_b{b}")
    print(f"Generating pixelated images: b = {b}")
    process_and_save(INPUT_DIR, out_dir, lambda img: apply_pixelation(img, b))

# ========== Generate Blurred Versions ==========
for k in BLUR_LEVELS:
    out_dir = Path(f"dataset/blurred_k{k}")
    print(f"Generating blurred images: k = {k}")
    process_and_save(INPUT_DIR, out_dir, lambda img: apply_blur(img, k))

print("All class-based de-identified datasets generated.")

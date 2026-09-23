import os
import numpy as np
from PIL import Image
import cv2  # OpenCV

def add_laplace_noise(image_array, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=image_array.shape)
    noisy_image = image_array + noise
    return np.clip(noisy_image, 0, 255)

def dp_box_blur_image(image_path, blur_kernel=5, epsilon=10.0):
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image, dtype=np.float32)

    # Step 1: Add Laplace noise
    sensitivity = 255.0
    noisy_array = add_laplace_noise(img_array, sensitivity, epsilon)

    # Step 2: Apply box blur using OpenCV
    noisy_uint8 = noisy_array.astype(np.uint8)
    blurred_array = cv2.blur(noisy_uint8, ksize=(blur_kernel, blur_kernel))

    # Convert back to PIL image
    return Image.fromarray(blurred_array)

def dp_box_blur_folder(input_root, output_root, blur_kernel=5, epsilon=10.0, image_exts={'.jpg', '.jpeg', '.png'}):
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not any(filename.lower().endswith(ext) for ext in image_exts):
                continue

            input_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                result = dp_box_blur_image(input_path, blur_kernel, epsilon)
                result.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed on {input_path}: {e}")

# Example usage
if __name__ == "__main__":
    input_folder = "dataset/original"
    output_folder = "dataset/DP_blur"
    dp_box_blur_folder(input_folder, output_folder, blur_kernel=15, epsilon=1.0)

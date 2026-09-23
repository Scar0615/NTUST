import os
import numpy as np
from PIL import Image

def add_laplace_noise(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    return value + np.random.laplace(0, scale)

def dp_pix_image(image_path, block_size=8, epsilon=10.0):
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    h, w, c = img_array.shape

    output = np.zeros_like(img_array)
    sensitivity = 255.0

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_array[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            avg_color = block.mean(axis=(0, 1))
            noisy_color = [
                np.clip(add_laplace_noise(avg_color[i], sensitivity, epsilon), 0, 255)
                for i in range(3)
            ]
            output[y:y+block_size, x:x+block_size] = noisy_color

    return Image.fromarray(output.astype(np.uint8))

def dp_pix_folder(input_root, output_root, block_size=8, epsilon=10.0, image_exts={'.jpg', '.jpeg', '.png'}):
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not any(filename.lower().endswith(ext) for ext in image_exts):
                continue

            input_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                result = dp_pix_image(input_path, block_size, epsilon)
                result.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed on {input_path}: {e}")

# Example usage
if __name__ == "__main__":
    input_folder = "dataset/original"
    output_folder = "dataset/DP-pix"
    dp_pix_folder(input_folder, output_folder, block_size=4, epsilon=1.0)

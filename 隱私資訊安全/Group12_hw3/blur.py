import cv2
import os

def pixelate_image(img, pixel_size):
    """
    Pixelise une image en la réduisant puis réagrandissant.
    """
    height, width = img.shape[:2]
    temp = cv2.resize(img, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def blur_image(img, blur_strength):
    """
    Applique un flou gaussien à l'image.
    """
    return cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

def process_dataset(input_folder, output_folder, pixel_size=16, blur_strength=9, apply_pixelate=True, apply_blur=True):
    """
    Applique la pixelisation et/ou le flou à toutes les images d'un dossier.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erreur chargement image : {img_path}")
                    continue

                # Application des effets
                processed = img.copy()
                if apply_pixelate:
                    processed = pixelate_image(processed, pixel_size)
                if apply_blur:
                    processed = blur_image(processed, blur_strength)

                # Enregistrement avec structure identique
                rel_path = os.path.relpath(root, input_folder)
                out_dir = os.path.join(output_folder, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, file)
                cv2.imwrite(out_path, processed)
                print(f"Image traitée : {out_path}")

# === Exemple d'utilisation ===
input_folder = "C:/TAIWAN/HW3/pictures/neutral"
output_folder = "C:/TAIWAN/HW3/pictures/blur_9"

# Paramètres
pixel_size = 48         # Plus petit = moins pixelisé
blur_strength = 9   # Doit être impair (ex: 3, 5, 7, 9)

# Active ou désactive les traitements
apply_pixelate = False
apply_blur = True

process_dataset(input_folder, output_folder, pixel_size, blur_strength, apply_pixelate, apply_blur)

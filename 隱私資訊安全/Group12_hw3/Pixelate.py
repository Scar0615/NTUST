import cv2
import os

def pixelate_image(img, pixel_size):
    """
    Pixelise une image en réduisant puis réagrandissant sa taille.
    """
    height, width = img.shape[:2]
    temp = cv2.resize(img, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def process_dataset(input_folder, output_folder, pixel_size=16):
    """
    Pixelise toutes les images dans un dossier (et sous-dossiers).
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

                pixelated = pixelate_image(img, pixel_size)

                # Création du chemin de sortie avec la même structure
                rel_path = os.path.relpath(root, input_folder)
                out_dir = os.path.join(output_folder, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, file)
                cv2.imwrite(out_path, pixelated)
                print(f"Image pixelisée : {out_path}")

# Exemple d'utilisation
input_folder = "C:/TAIWAN/HW3/pictures/neutral"
output_folder = "C:/TAIWAN/HW3/pictures/pixelated_24"
pixel_size = 24  # Taille réduite pour la pixelisation

process_dataset(input_folder, output_folder, pixel_size)

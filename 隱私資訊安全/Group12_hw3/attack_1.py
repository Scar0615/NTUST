import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === Paramètres ===
IMG_SIZE = 48
DATA_PATH_ORIG = "C:/TAIWAN/HW3/pictures/neutral"
DATA_PATH_MODIF = "C:/TAIWAN/HW3/pictures/pixelated_16"

def load_images(folder, img_size):
    images = []
    for file in sorted(os.listdir(folder)):  # sorted to align images
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            images.append(img)
    return np.array(images)

# Chargement des données
x_orig = load_images(DATA_PATH_ORIG, IMG_SIZE)
x_modif = load_images(DATA_PATH_MODIF, IMG_SIZE)

print(f"Nombre d'images chargées : {len(x_orig)}")

# === Création du modèle autoencodeur convolutionnel ===
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

autoencoder.summary()

# === Entraînement ===
autoencoder.fit(x_modif, x_orig, epochs=50, batch_size=16, shuffle=True, validation_split=0.1)

# === Visualisation des résultats ===
decoded_imgs = autoencoder.predict(x_modif[:5])

for i in range(5):
    plt.figure(figsize=(10,3))
    # Image modifiée
    plt.subplot(1, 3, 1)
    plt.imshow(x_modif[i])
    plt.title("Modifiée")
    # Reconstruction
    plt.subplot(1, 3, 2)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstruction")
    # Image originale
    plt.subplot(1, 3, 3)
    plt.imshow(x_orig[i])
    plt.title("Originale")
    plt.show()

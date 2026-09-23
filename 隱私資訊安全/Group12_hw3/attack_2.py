import os
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

IMG_SIZE = 48
MOD_DIR = "C:/TAIWAN/HW3/pictures/blur_5"
ORIG_DIR = "C:/TAIWAN/HW3/pictures/neutral"

def load_images(folder, size):
    images = []
    files = sorted(os.listdir(folder))
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.resize(img, (size, size))
            img = img.astype(np.float32) / 255.0
            images.append(img)
    return np.array(images), files

# === Chargement des images ===
x_modif, modif_files = load_images(MOD_DIR, IMG_SIZE)
x_orig, orig_files = load_images(ORIG_DIR, IMG_SIZE)

# Assure-toi que les fichiers sont triés dans le même ordre
assert len(x_modif) == len(x_orig), "Les dossiers ne contiennent pas le même nombre d'images"

# Les labels sont simplement les indices (0 à N-1)
y_labels = np.arange(len(x_modif))

# One-hot encoding
y_onehot = keras.utils.to_categorical(y_labels, num_classes=len(x_modif))

# === Modèle CNN de classification ===
Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
MaxPool2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Model = keras.models.Model

inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Conv2D(32, (3, 3), activation="relu")(inp)
x = MaxPool2D()(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
out = Dense(len(x_modif), activation="softmax")(x)

model = Model(inp, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === Entraînement ===
model.fit(x_modif, y_onehot, epochs=50, batch_size=4, validation_split=0.1)

# === Prédiction sur quelques exemples ===
preds = model.predict(x_modif[:5])
for i, pred in enumerate(preds):
    predicted_index = np.argmax(pred)
    actual_index = i
    print(f"Image modifiée {modif_files[i]} → prédite comme {orig_files[predicted_index]}")

# === Évaluation du modèle sur tout l'ensemble
preds = model.predict(x_modif)
predicted_labels = np.argmax(preds, axis=1)
true_labels = np.arange(len(x_modif))

# Calcul du taux de bonnes classifications
accuracy = np.mean(predicted_labels == true_labels)
print(f"Taux de bonne classification : {accuracy * 100:.2f}%")



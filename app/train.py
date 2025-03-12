#!/usr/bin/env python3
"""
train.py
Entraîne un modèle CNN pour classifier les feuilles dans Leaffliction.
Génère un modèle H5, un ZIP, et un signature.txt.
"""

import os
import zipfile
import hashlib
import argparse
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from colorama import Fore, init

init(autoreset=True)


def generate_sha1(file_path):
    """
    Génère un hash SHA1 d'un fichier
    """
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def create_zip_and_signature(model_file, zip_output_file, signature_file):
    """
    Zip le modèle et crée le fichier signature SHA1
    """
    print(f"{Fore.YELLOW}➡️ Zippage du modèle : {model_file} ➜ "
          f"{zip_output_file}")

    with zipfile.ZipFile(zip_output_file, 'w') as zipf:
        zipf.write(model_file, os.path.basename(model_file))

    sha1_hash = generate_sha1(zip_output_file)

    with open(signature_file, 'w') as f:
        f.write(f"{sha1_hash}  {os.path.basename(zip_output_file)}\n")

    print(f"{Fore.GREEN}✅ ZIP généré : {zip_output_file}")
    print(f"{Fore.GREEN}✅ Signature SHA1 sauvegardée : "
          f"{signature_file} ({sha1_hash})")


def plot_history(history, output_dir):
    """
    Génère et sauvegarde les courbes d'apprentissage.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_file)
    plt.close()

    print(f"{Fore.CYAN}📈 Courbes d'apprentissage sauvegardées ➜ {plot_file}")


def train_model(balanced_dir,
                output_model_dir,
                zip_output_file,
                signature_file,
                img_size=(224, 224),
                batch_size=32,
                epochs=30,
                validation_split=0.2):

    os.makedirs(output_model_dir, exist_ok=True)
    model_file = os.path.join(output_model_dir, 'leaffliction_model.h5')

    # === DATA GENERATOR ===
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )

    print(f"{Fore.YELLOW}➡️ Chargement du dataset depuis : {balanced_dir}")

    train_generator = datagen.flow_from_directory(
        balanced_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        balanced_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # === CALCUL DES POIDS DES CLASSES ===
    # Récupérer les indices de classe de toutes les images d'entraînement
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())

    # Pour calculer les poids automatiquement à partir du nombre d'échantillons
    # On récupère les étiquettes des échantillons d'entraînement
    train_labels = train_generator.classes

    # Calcul des poids des classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels)
    class_weight_dict = {
        i: class_weights[i] for i in range(len(class_weights))
    }

    # Affichage des poids pour vérification
    print(f"{Fore.CYAN}➡️ Poids des classes :")
    for class_name, idx in class_indices.items():
        print(f"   {class_name}: {class_weight_dict[idx]:.2f}")

    # Accentuer manuellement les poids pour les classes problématiques
    # Augmenter le poids des classes Grape_healthy
    # et Grape_Black_rot qui sont mal classifiées
    if "Grape_healthy" in class_names:
        healthy_idx = class_indices["Grape_healthy"]
        class_weight_dict[healthy_idx] *= 1.5  # Donner 50% plus de poids
        print(f"{Fore.GREEN}✅ Poids de Grape_healthy augmenté à "
              f"{class_weight_dict[healthy_idx]:.2f}")

    if "Grape_Black_rot" in class_names:
        black_rot_idx = class_indices["Grape_Black_rot"]
        class_weight_dict[black_rot_idx] *= 1.2  # Donner 20% plus de poids
        print(f"{Fore.GREEN}✅ Poids de Grape_Black_rot augmenté à "
              f"{class_weight_dict[black_rot_idx]:.2f}")

    if "Grape_spot" in class_names:
        spot_idx = class_indices["Grape_spot"]
        class_weight_dict[spot_idx] *= 1.5  # Augmenter le poids de 50%
        print(f"{Fore.GREEN}✅ Poids de Grape_spot augmenté à "
              f"{class_weight_dict[spot_idx]:.2f}")

    # === MODEL CNN ===
    num_classes = train_generator.num_classes

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # === CALLBACKS ===
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True),
        ModelCheckpoint(
            model_file,
            monitor='val_accuracy',
            save_best_only=True)
    ]

    # === TRAIN ===
    print(
        f"{Fore.YELLOW}➡️ Démarrage de l'entraînement avec class weighting...")

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # === EVALUATION ===
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"\n{Fore.GREEN}✅ Validation accuracy: {val_accuracy * 100:.2f}%")

    plot_history(history, output_model_dir)

    # === ZIP + SIGNATURE ===
    create_zip_and_signature(model_file, zip_output_file, signature_file)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Entraînement du modèle Leaffliction "
            "(classification de feuilles)"
        ))
    parser.add_argument(
        "--balanced_dir", type=str, default="./output/balanced",
        help="Chemin du dataset équilibré (par défaut: ./output/balanced)")
    parser.add_argument(
        "--output_model_dir", type=str, default="./output/models",
        help="Dossier de sortie du modèle (par défaut: ./output/models)")
    parser.add_argument(
        "--zip_output_file", type=str, default="./dataset.zip",
        help="Nom du fichier zip généré (par défaut: ./dataset.zip)")
    parser.add_argument(
        "--signature_file", type=str, default="./signature.txt",
        help="Nom du fichier signature généré (par défaut: ./signature.txt)")
    parser.add_argument(
        "--img_size", type=int, nargs=2, default=(224, 224),
        help="Dimensions des images (par défaut: 224 224)")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Taille des batchs (par défaut: 32)")
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Nombre d'époques d'entraînement (par défaut: 30)")
    parser.add_argument(
        "--validation_split", type=float, default=0.2,
        help="Fraction du dataset pour la validation (par défaut: 0.2)")

    args = parser.parse_args()

    train_model(args.balanced_dir, args.output_model_dir,
                args.zip_output_file, args.signature_file,
                img_size=tuple(args.img_size),
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_split=args.validation_split)


if __name__ == "__main__":
    main()

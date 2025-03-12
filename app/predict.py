#!/usr/bin/env python3
"""
predict.py
Prédit la classe d'une image ou d'un dossier d'images.
Affiche les résultats et les sauvegarde dans output/predictions/.
"""

import os
import argparse
from colorama import init, Fore
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re

init(autoreset=True)

# Paramètres
IMAGE_SIZE = (224, 224)
OUTPUT_PREDICTIONS = "./output/predictions"
MODEL_PATH = "./output/models/leaffliction_model.h5"
CLASSES_DIR = "./output/balanced"


def load_and_prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire : {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, IMAGE_SIZE)
    normalized = resized.astype('float32') / 255.0
    expanded = np.expand_dims(normalized, axis=0)
    return expanded


def predict_image(model, img_tensor):
    prediction = model.predict(img_tensor, verbose=0)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    return class_idx, confidence


def load_class_labels(classes_dir):
    return sorted([
        d for d in os.listdir(classes_dir)
        if os.path.isdir(os.path.join(classes_dir, d))
    ])


def predict_image_top_k(model, img_tensor, class_labels, k=2):
    prediction = model.predict(img_tensor, verbose=0)[0]
    # Récupère les indices triés des plus grandes probabilités
    top_indices = prediction.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        label = class_labels[idx]
        confidence = prediction[idx]
        results.append((label, confidence))

    return results


def predict_on_single_image(model, class_labels, image_path):
    try:
        img_tensor = load_and_prepare_image(image_path)
        idx, confidence = predict_image(model, img_tensor)
        predicted_label = class_labels[idx]
        print(f"{Fore.CYAN}{os.path.basename(image_path)} ➜ "
              f"{Fore.GREEN}{predicted_label} ({confidence*100:.2f}%)")
        return predicted_label, confidence
    except Exception as e:
        print(f"{Fore.RED}Erreur sur {image_path}: {e}")
        return None, None


def extract_true_label(filename):
    filename_no_ext = os.path.splitext(filename)[0]

    match = re.match(r'^(.*?)(?:_\d+)?$', filename_no_ext)

    if match:
        return match.group(1)
    else:
        return filename_no_ext


def is_image_file(filename):
    """
    Vérifie si le fichier est une image valide
    """
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))


def predict_on_folder(model,
                      class_labels,
                      folder_path,
                      output_dir,
                      confusion=False):
    """
    Prédit les classes d'un dossier d'images, enregistre les résultats,
    et génère une matrice de confusion si demandé.

    :param model: Le modèle Keras entraîné.
    :param class_labels: Liste des classes apprises par le modèle.
    :param folder_path: Dossier contenant les images à prédire.
    :param output_dir: Dossier où sauvegarder les résultats.
    :param confusion: Booléen pour activer la matrice de confusion.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 📄 Fichier pour écrire les prédictions
    output_file = os.path.join(
        output_dir, f"predictions_{os.path.basename(folder_path)}.txt"
    )

    y_true = []
    y_pred = []

    print(f"{Fore.BLUE}➡️ Dossier à prédire : {folder_path}")

    with open(output_file, 'w') as f:
        for root, _, files in os.walk(folder_path):
            for file in sorted(files):  # Tri pour l'ordre
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(root, file)

                # Méthode améliorée pour extraire la classe réelle
                filename_no_ext = os.path.splitext(file)[0]

                # Option 1: Essayer de trouver une correspondance exacte
                true_label = None
                for label in class_labels:
                    # Vérifie différents formats possibles
                    is_exact = filename_no_ext == label
                    starts_with = filename_no_ext.startswith(f"{label}_")
                    if is_exact or starts_with:
                        true_label = label
                        break

                # Si aucune correspondance exacte,
                # alors chercher si le label fait partie du nom
                if true_label is None:
                    # Trie les labels du plus long au plus court
                    # pour éviter les sous-chaînes
                    sorted_labels = sorted(
                        class_labels, key=len, reverse=True
                    )
                    for label in sorted_labels:
                        if label in filename_no_ext:
                            true_label = label
                            break

                if true_label is None:
                    error_msg = (
                        f"{Fore.RED}❌ Classe inconnue pour {file} ! "
                        f"Impossible de déterminer l'étiquette réelle.")
                    print(error_msg)
                    continue

                y_true.append(true_label)

                # 📸 Faire la prédiction
                try:
                    img_tensor = load_and_prepare_image(image_path)

                    # Afficher les top 2 prédictions pour le diagnostic
                    top_predictions = predict_image_top_k(
                        model, img_tensor, class_labels, k=2
                    )
                    predicted_label, confidence = top_predictions[0]

                    second_best = ""
                    if len(top_predictions) > 1:
                        second_label, second_conf = top_predictions[1]
                        second_best = (
                            f" (2nd: {second_label}: "
                            f"{second_conf*100:.2f}%)"
                        )

                    color = (Fore.GREEN if predicted_label == true_label
                             else Fore.RED)
                    print(f"{Fore.CYAN}{file} ➜ "
                          f"{color}"
                          f"{predicted_label} ({confidence*100:.2f}%)"
                          f"{second_best}")

                    y_pred.append(predicted_label)
                    f.write(f"{file}: {predicted_label} "
                            f"({confidence*100:.2f}%)\n")

                except Exception as e:
                    print(f"{Fore.RED}Erreur sur {image_path}: {e}")

    print(f"{Fore.YELLOW}✅ Résultats enregistrés dans : {output_file}")

    # Créer un rapport détaillé des erreurs
    errors_file = os.path.join(
        output_dir,
        f"errors_{os.path.basename(folder_path)}.txt")

    with open(errors_file, 'w') as f:
        f.write("Erreurs de prédiction :\n")
        f.write("=====================\n\n")
        image_files = [
            f for f in os.listdir(folder_path) if is_image_file(f)
        ]
        for true, pred, file in zip(y_true, y_pred, image_files):
            if true != pred:
                f.write(f"Fichier: {file}\n")
                f.write(f"   Classe réelle: {true}\n")
                f.write(f"   Prédiction:    {pred}\n\n")

    error_msg = (
        f"{Fore.YELLOW}✅ Rapport d'erreurs enregistré dans : {errors_file}"
    )
    print(error_msg)

    if confusion:
        print(f"{Fore.YELLOW}➡️ Génération de la matrice de confusion...")

        # ✅ On vérifie les classes réellement présentes
        unique_classes = sorted(set(y_true + y_pred))

        class_msg = (
            f"{Fore.BLUE}➡️ Classes présentes dans ce jeu de données : "
            f"{unique_classes}"
        )
        print(class_msg)

        if not unique_classes:
            error_msg = (
                f"{Fore.RED}❌ Aucune classe détectée pour générer la "
                "matrice de confusion."
            )
            print(error_msg)
            return

        # ➡️ Calcul de la matrice
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

        # ➡️ Affichage de la matrice
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=unique_classes
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

        # ➡️ Sauvegarde de la matrice en PNG
        output_conf = os.path.join(
            output_dir,
            f"confusion_matrix_{os.path.basename(folder_path)}.png"
        )
        title = f"Matrice de confusion - {os.path.basename(folder_path)}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_conf)
        plt.show()

        conf_msg = (
            f"{Fore.GREEN}✅ Matrice de confusion sauvegardée : {output_conf}"
        )
        print(conf_msg)


def show_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Matrice de confusion")
    plt.tight_layout()

    # ➡️ Sauvegarder l'image dans le dossier output
    output_path = os.path.join(OUTPUT_PREDICTIONS, "confusion_matrix.png")
    plt.savefig(output_path)
    print(f"{Fore.GREEN}✅ Matrice de confusion sauvegardée dans {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Prédiction sur une image ou un dossier")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Image ou dossier à prédire")
    parser.add_argument("--output", type=str, default=OUTPUT_PREDICTIONS,
                        help="Dossier de sortie")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Modèle H5 entraîné")
    parser.add_argument("--confusion", action='store_true',
                        help="Générer la matrice de confusion")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"{Fore.RED}❌ Erreur : le chemin {args.input} n'existe pas")
        return

    print(f"{Fore.BLUE}➡️ Chargement du modèle : {args.model}")
    model = load_model(args.model)
    class_labels = load_class_labels(CLASSES_DIR)

    if os.path.isfile(args.input):
        predict_on_single_image(model, class_labels, args.input)
    elif os.path.isdir(args.input):
        predict_on_folder(model, class_labels, args.input, args.output,
                          confusion=args.confusion)
    else:
        print(f"{Fore.RED}❌ Chemin invalide : {args.input}")


if __name__ == "__main__":
    main()

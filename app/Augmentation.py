#!/usr/bin/env python3
"""
Augmentation.py
Génère des augmentations d'images pour le projet Leafflection.
Sauvegarde des augmentations et des visualisations dans le dossier de sortie.
"""

import os
import sys
import argparse
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorama import Fore, init

# Initialisation de colorama pour l'affichage coloré en console
init(autoreset=True)

# Extensions d'images valides
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def is_image_file(filename):
    """
    Vérifie si le fichier est une image valide
    """
    return filename.lower().endswith(VALID_EXTENSIONS)


class ImageAugmentation:
    """
    Classe gérant le chargement, sauvegarde et augmentations d'images
    """

    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        """
        Charge une image depuis un chemin
        """
        return cv2.imread(image_path)

    def save_image(self, image, method_name, output_dir, image_name):
        """
        Sauvegarde une image après transformation
        """
        base_name = os.path.splitext(image_name)[0]
        save_folder = os.path.join(output_dir)
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, f"{base_name}_{method_name}.JPG")
        cv2.imwrite(save_path, image)
        print(f"{Fore.GREEN}Image sauvegardée ➜ {save_path}")

    @staticmethod
    def rotate(image, angle=90):
        """
        Rotation de l'image
        """
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def flip(image):
        """
        Flip horizontal (symétrie gauche-droite)
        """
        return cv2.flip(image, 1)

    @staticmethod
    def skew(image):
        """
        Déformation par skewing (affine)
        """
        rows, cols, _ = image.shape
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([
            [0, 0],
            [int(0.8 * (cols - 1)), int(0.2 * (rows - 1))],
            [int(0.2 * (cols - 1)), int(0.8 * (rows - 1))]
        ])
        matrix = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(image, matrix, (cols, rows))

    @staticmethod
    def shear(image):
        """
        Cisaillement horizontal
        """
        rows, cols, _ = image.shape
        matrix = np.float32([[1, 0.5, 0], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (cols + int(cols * 0.5), rows))

    @staticmethod
    def crop(image, scale=0.8):
        """
        Crop central de l'image
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        start_x = w // 2 - new_w // 2
        start_y = h // 2 - new_h // 2
        cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def distort(image):
        """
        Flou gaussien
        """
        return cv2.GaussianBlur(image, (5, 5), 0)


def apply_augmentations(image, augmenter):
    """
    Applique les différentes augmentations sur une image
    """
    return {
        "Flip": augmenter.flip(image),
        "Rotate": augmenter.rotate(image),
        "Skew": augmenter.skew(image),
        "Shear": augmenter.shear(image),
        "Crop": augmenter.crop(image),
        "Distortion": augmenter.distort(image)
    }


def process_single_image(image_path, output_dir, graph_dir=None):
    """
    Traite une seule image (augmentation + visualisation)
    """
    augmenter = ImageAugmentation()
    image_name = os.path.basename(image_path)
    image = augmenter.load_image(image_path)

    if image is None:
        print(f"{Fore.RED}Erreur : Impossible de charger l'image {image_path}")
        return

    augmentations = apply_augmentations(image, augmenter)

    # Sauvegarde des images augmentées
    for method_name, aug_img in augmentations.items():
        augmenter.save_image(aug_img, method_name, output_dir, image_name)

    # Génération et sauvegarde du graphique de visualisation
    if graph_dir:
        save_visualization(image, augmentations, image_name, graph_dir)


def process_directory(directory_path, output_dir, graph_dir=None):
    """
    Traite toutes les images d'un dossier
    """
    print(f"{Fore.YELLOW}Traitement du dossier ➜ {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if is_image_file(file):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, directory_path)
                target_output_dir = os.path.join(output_dir, relative_path)
                target_graph_dir = (os.path.join(graph_dir, relative_path)
                                    if graph_dir else None)
                process_single_image(
                    img_path, target_output_dir, target_graph_dir)


def save_visualization(original_image, augmentations, image_title, output_dir):
    """
    Génère et sauvegarde un graphe matplotlib montrant toutes les augmentations
    """
    methods = list(augmentations.keys())
    images = list(augmentations.values())

    fig, axs = plt.subplots(1, len(images) + 1, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis("off")

    for i, (method, img) in enumerate(zip(methods, images)):
        axs[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(method)
        axs[i + 1].axis("off")

    plt.tight_layout()

    base_name = os.path.splitext(image_title)[0]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"visualization_{base_name}.png")
    plt.savefig(output_file)
    plt.close()

    print(f"{Fore.CYAN}Graphe sauvegardé ➜ {output_file}")


def main():
    """
    Point d'entrée principal du script
    """
    parser = argparse.ArgumentParser(
        description="Data Augmentation Leafflection")
    parser.add_argument("path", type=str,
                        help="Chemin vers l'image ou le dossier d'images")
    parser.add_argument("--output", type=str, default="./output/augmented",
                        help="Dossier de sauvegarde des images augmentées")
    parser.add_argument("--graph_output", type=str, default=None,
                        help="Dossier de sauvegarde des visualisations")
    args = parser.parse_args()

    input_path = args.path
    output_dir = args.output
    graph_dir = args.graph_output

    if not os.path.exists(input_path):
        print(f"{Fore.RED}Erreur : Le chemin {input_path} n'existe pas")
        sys.exit(1)

    # Nettoyage des dossiers output si besoin
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    if graph_dir and os.path.isdir(graph_dir):
        shutil.rmtree(graph_dir)

    if os.path.isfile(input_path) and is_image_file(input_path):
        process_single_image(input_path, output_dir, graph_dir)
    elif os.path.isdir(input_path):
        process_directory(input_path, output_dir, graph_dir)
    else:
        print(f"{Fore.RED}Erreur : Chemin non valide ou "
              "type de fichier incorrect")


if __name__ == "__main__":
    main()

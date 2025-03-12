#!/usr/bin/env python3
"""
Transformation.py
Applique des transformations d'images pour le projet Leafflection.
Peut fonctionner sur une seule image ou un dossier complet.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from colorama import Fore, init

init(autoreset=True)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def is_image_file(filename):
    """
    Vérifie si le fichier est une image valide
    """
    return filename.lower().endswith(VALID_EXTENSIONS)


class ImageTransformation:
    """
    Classe regroupant les méthodes de transformation d'image
    """

    @staticmethod
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (9, 9), 0)

    @staticmethod
    def edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def roi(image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        top_left = (int(w * 0.25), int(h * 0.25))
        bottom_right = (int(w * 0.75), int(h * 0.75))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        roi_img = cv2.bitwise_and(image, image, mask=mask)
        return roi_img

    @staticmethod
    def color_histogram(image):
        chans = cv2.split(image)
        colors = ("b", "g", "r")
        hist_img = np.zeros((300, 256, 3), dtype=np.uint8)

        for chan, color, idx in zip(chans, colors, range(3)):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
            for x, y in enumerate(hist):
                cv2.line(hist_img,
                         (x, 300),
                         (x, 300 - int(y)),
                         (255 if color == 'b' else 0,
                          255 if color == 'g' else 0,
                          255 if color == 'r' else 0),
                         1)
        return hist_img

    @staticmethod
    def pseudolandmarks(image):
        h, w = image.shape[:2]
        points = [(int(w * 0.3), int(h * 0.3)),
                  (int(w * 0.7), int(h * 0.3)),
                  (int(w * 0.5), int(h * 0.7))]
        img_copy = image.copy()
        for pt in points:
            cv2.circle(img_copy, pt, 10, (0, 255, 255), -1)
        return img_copy

    @staticmethod
    def binary_mask(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_transformations(image):
    """
    Applique les différentes transformations sur une image
    """
    transformer = ImageTransformation()

    return {
        "GaussianBlur": transformer.gaussian_blur(image),
        "EdgeDetection": transformer.edge_detection(image),
        "ROI": transformer.roi(image),
        "ColorHistogram": transformer.color_histogram(image),
        "Pseudolandmarks": transformer.pseudolandmarks(image),
        "BinaryMask": transformer.binary_mask(image)
    }


def process_single_image(image_path):
    """
    Affiche les transformations sur une image unique
    """
    print(f"{Fore.YELLOW}Traitement de l'image ➜ {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"{Fore.RED}Erreur : Impossible de charger l'image {image_path}")
        return

    transformations = apply_transformations(image)

    fig, axs = plt.subplots(1, len(transformations) + 1, figsize=(20, 5))

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis("off")

    for i, (name, transformed_image) in enumerate(transformations.items()):
        axs[i + 1].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(name)
        axs[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def process_directory(src_dir, dst_dir):
    """
    Applique les transformations
    sur un dossier complet et sauvegarde les images
    """
    print(f"{Fore.YELLOW}Traitement du dossier ➜ {src_dir}")

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for root, _, files in os.walk(src_dir):
        for file in files:
            if not is_image_file(file):
                continue

            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"{Fore.RED}Erreur : Impossible de charger {img_path}")
                continue

            relative_path = os.path.relpath(root, src_dir)
            out_dir = os.path.join(dst_dir, relative_path)
            os.makedirs(out_dir, exist_ok=True)

            base_name = os.path.splitext(file)[0]

            transformations = apply_transformations(image)

            # Sauvegarde des images transformées
            for name, transformed_image in transformations.items():
                output_filename = f"{base_name}_{name}.JPG"
                output_path = os.path.join(out_dir, output_filename)
                cv2.imwrite(output_path, transformed_image)
                print(f"{Fore.GREEN}Image sauvegardée ➜ {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transformations d'images Leafflection")
    parser.add_argument("src",
                        type=str,
                        help="Image unique ou dossier d'images à traiter")
    parser.add_argument("-dst",
                        type=str,
                        default="./output/transformations",
                        help="Dossier de sortie pour les transformations")
    args = parser.parse_args()

    src_path = args.src
    dst_path = args.dst

    if not os.path.exists(src_path):
        print(f"{Fore.RED}Erreur : Le chemin {src_path} n'existe pas")
        sys.exit(1)

    if os.path.isfile(src_path) and is_image_file(src_path):
        process_single_image(src_path)
    elif os.path.isdir(src_path):
        process_directory(src_path, dst_path)
    else:
        print(f"{Fore.RED}Erreur : chemin non valide ou "
              f"type de fichier incorrect")
        sys.exit(1)


if __name__ == "__main__":
    main()

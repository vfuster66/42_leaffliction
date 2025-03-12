#!/usr/bin/env python3
"""
BalanceDataset.py
Équilibre un dataset en forçant le même nombre d'images par classe.
Utile après l'augmentation pour préparer l'entraînement du modèle.
"""

import os
import sys
import shutil
import random
import argparse
from tqdm import tqdm
from colorama import Fore, init

init(autoreset=True)


VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def is_image_file(filename):
    """
    Vérifie si le fichier est une image valide
    """
    return filename.lower().endswith(VALID_EXTENSIONS)


def balance_dataset(input_dir, output_dir, target_size):
    """
    Copie ou duplique les images pour équilibrer chaque classe à target_size
    """
    if not os.path.exists(input_dir):
        print(f"{Fore.RED}Erreur : Le dossier source {input_dir} n'existe pas")
        sys.exit(1)

    # Nettoyage du dossier output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    classes = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    print(f"{Fore.YELLOW}➡️ {len(classes)} classes détectées : {classes}\n")

    for cls in tqdm(classes, desc="Balancing classes"):
        src_class_dir = os.path.join(input_dir, cls)
        dst_class_dir = os.path.join(output_dir, cls)
        os.makedirs(dst_class_dir, exist_ok=True)

        images = [
            img for img in os.listdir(src_class_dir) if is_image_file(img)
        ]

        if not images:
            print(f"{Fore.RED}⚠️ Classe {cls} vide, ignorée.")
            continue

        # Si la classe a assez d'images ➜ on prend un échantillon
        if len(images) >= target_size:
            selected_images = random.sample(images, target_size)
        else:
            # Sinon ➜ on duplique aléatoirement jusqu'à atteindre target_size
            selected_images = images.copy()
            while len(selected_images) < target_size:
                selected_images.append(random.choice(images))

        # Copie vers le dossier de sortie
        for img in selected_images:
            src_path = os.path.join(src_class_dir, img)
            dst_path = os.path.join(dst_class_dir, img)
            shutil.copy(src_path, dst_path)

    print(f"\n{Fore.GREEN}✅ Dataset équilibré ➜ {output_dir}")
    print(f"{Fore.CYAN}   ➜ {target_size} images par classe")
    print(f"{Fore.CYAN}   ➜ Total images : {len(classes) * target_size}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Balance un dataset d'images en forçant un nombre d'images égal "
            "par classe"
        )
    )
    parser.add_argument("src", type=str,
                        help="Chemin du dossier source (augmented)")
    parser.add_argument("dst", type=str,
                        help=(
                            "Chemin du dossier équilibré à créer "
                            "(train_dataset)"
                        ))
    parser.add_argument("--size", type=int, default=500,
                        help=(
                            "Nombre d'images à conserver par classe "
                            "(par défaut : 500)"
                        ))

    args = parser.parse_args()

    balance_dataset(args.src, args.dst, args.size)


if __name__ == "__main__":
    main()

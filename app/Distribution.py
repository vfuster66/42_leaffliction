import os
import sys
import matplotlib.pyplot as plt
import random
import argparse


def load_images_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"⚠️ Skipping non-directory: {directory_path}")
        return []

    images = os.listdir(directory_path)
    images = [
        image for image in images
        if image.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return images


def main_images_distribution(directory_path):
    sub_directories = os.listdir(directory_path)
    valid_sub_directories = 0
    directory_images = {}
    total_images = 0

    for sub_directory in sub_directories:
        sub_directory_path = os.path.join(directory_path, sub_directory)

        # 🔧 Vérifie que c'est bien un dossier
        if not os.path.isdir(sub_directory_path):
            print(f"⚠️ Ignored: {sub_directory_path} (not a directory)")
            continue

        images = load_images_from_directory(sub_directory_path)

        if len(images) != 0:
            total_images += len(images)
            directory_images[sub_directory] = images
            valid_sub_directories += 1

    print(f"Found {valid_sub_directories} sub-directories in {directory_path} "
          f"with a total of {total_images} images.")
    return directory_images


def generate_random_hexa_color_codes(number_of_colors_to_generate):
    color_codes = []
    for _ in range(number_of_colors_to_generate):
        color_codes.append("#" + ("%06x" % random.randint(0, 0xFFFFFF)))
    return color_codes


def plot_image_distribution(
    base_directory_name, images_with_directory_names, output_dir
):
    labels = list(images_with_directory_names.keys())
    images_list = list(images_with_directory_names.values())
    total_images = [len(images) for images in images_list]
    colors = generate_random_hexa_color_codes(len(labels))

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Images distribution in {base_directory_name}")

    plt.subplot(1, 2, 1)
    plt.pie(total_images, labels=labels, autopct="%1.1f%%", colors=colors)
    plt.title("Pie chart")

    plt.subplot(1, 2, 2)
    plt.bar(labels, total_images, color=colors)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True)
    plt.title("Bar chart")

    output_file = os.path.join(
        output_dir, f"distribution_{os.path.basename(base_directory_name)}.png"
    )
    plt.savefig(output_file)
    print(f"Graphes sauvegardés dans ➜ {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="Path to the source dir.")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix to add to output filenames.")
    args = parser.parse_args()

    directory = args.src
    if not os.path.isdir(directory):
        sys.exit("Invalid directory path")

    directory_names_with_images = main_images_distribution(directory)
    plot_image_distribution(
        directory, directory_names_with_images, "./output/plots"
    )


if __name__ == "__main__":
    main()

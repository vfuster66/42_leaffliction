import os
import pytest
from app.Distribution import (
    load_images_from_directory,
    main_images_distribution,
    generate_random_hexa_color_codes,
    plot_image_distribution
)

# === FIXTURES ===


@pytest.fixture
def test_directory(tmp_path):
    """
    Crée un dossier temporaire avec des sous-dossiers et images
    """
    # Création du dossier racine temporaire
    root = tmp_path / "images"
    root.mkdir()

    # Sous-dossiers simulant des classes
    class1 = root / "Apple_healthy"
    class2 = root / "Apple_scab"
    class1.mkdir()
    class2.mkdir()

    # Fichiers images dans chaque classe
    (class1 / "image1.jpg").write_text("fake image data")
    (class1 / "image2.JPG").write_text("fake image data")
    (class2 / "image1.png").write_text("fake image data")
    (class2 / "note.txt").write_text("this is not an image")

    return root


@pytest.fixture
def output_dir(tmp_path):
    """
    Crée un dossier temporaire pour sauvegarder les graphes
    """
    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    return out_dir


def test_load_images_from_directory(test_directory):
    class1_path = test_directory / "Apple_healthy"
    images = load_images_from_directory(str(class1_path))

    assert len(images) == 2
    assert "image1.jpg" in images
    assert "image2.JPG" in images


def test_main_images_distribution(test_directory):
    result = main_images_distribution(str(test_directory))

    assert isinstance(result, dict)
    assert "Apple_healthy" in result
    assert "Apple_scab" in result
    assert len(result["Apple_healthy"]) == 2
    assert len(result["Apple_scab"]) == 1


def test_generate_random_hexa_color_codes():
    color_codes = generate_random_hexa_color_codes(5)

    assert isinstance(color_codes, list)
    assert len(color_codes) == 5
    for color in color_codes:
        assert color.startswith("#") and len(color) == 7


def test_plot_image_distribution(output_dir, test_directory):
    directory_name = str(test_directory)
    images_with_dirs = {
        "Apple_healthy": ["image1.jpg", "image2.JPG"],
        "Apple_scab": ["image1.png"]
    }

    plot_image_distribution(directory_name, images_with_dirs, str(output_dir))

    # Vérifie que le fichier plot est généré
    files = os.listdir(output_dir)
    assert any(file.endswith(".png") for file in files)

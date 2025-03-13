import os
import pytest
import numpy as np
import cv2

from app.Augmentation import (
    is_image_file,
    ImageAugmentation,
    apply_augmentations,
)

# === FIXTURES ===


@pytest.fixture
def sample_image(tmp_path):
    """
    Crée une image dummy 100x100 RGB
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 255, 0]  # Vert

    # Enregistre l'image dans un fichier temporaire
    img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def output_dir(tmp_path):
    """
    Crée un dossier temporaire pour l'output des images augmentées
    """
    out_dir = tmp_path / "augmented"
    out_dir.mkdir()
    return out_dir


def test_is_image_file():
    assert is_image_file("test.jpg") is True
    assert is_image_file("photo.jpeg") is True
    assert is_image_file("image.PNG") is True
    assert is_image_file("document.pdf") is False
    assert is_image_file("archive.zip") is False


def test_load_image(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))

    assert img is not None
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100, 3)


def test_rotate(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    rotated_img = augmenter.rotate(img)

    assert rotated_img.shape == img.shape


def test_flip(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    flipped_img = augmenter.flip(img)

    assert flipped_img.shape == img.shape


def test_skew(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    skewed_img = augmenter.skew(img)

    assert isinstance(skewed_img, np.ndarray)
    assert skewed_img.shape[:2] == img.shape[:2]


def test_shear(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    sheared_img = augmenter.shear(img)

    assert isinstance(sheared_img, np.ndarray)
    assert sheared_img.shape[0] == img.shape[0]
    assert sheared_img.shape[1] > img.shape[1]  # largeur augmente


def test_crop(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    cropped_img = augmenter.crop(img)

    assert cropped_img.shape == img.shape


def test_distort(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    distorted_img = augmenter.distort(img)

    assert distorted_img.shape == img.shape


def test_apply_augmentations(sample_image):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))
    augmentations = apply_augmentations(img, augmenter)

    assert isinstance(augmentations, dict)
    assert set(augmentations.keys()) == {
        "Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"
    }

    for name, aug_img in augmentations.items():
        assert isinstance(aug_img, np.ndarray)
        assert aug_img.shape[0] > 0
        assert aug_img.shape[1] > 0


def test_save_image(sample_image, output_dir):
    augmenter = ImageAugmentation()
    img = augmenter.load_image(str(sample_image))

    augmenter.save_image(img, "test_method", str(output_dir), "sample.jpg")

    files = os.listdir(output_dir)
    assert len(files) == 1
    assert files[0].startswith("sample_test_method")
    assert files[0].endswith(".JPG")

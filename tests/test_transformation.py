import pytest
import numpy as np
import cv2

from app.Transformation import (
    is_image_file,
    ImageTransformation,
    apply_transformations
)

# === FIXTURES ===


@pytest.fixture
def sample_image(tmp_path):
    """
    Crée une image dummy 100x100 RGB
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]  # Rouge

    # Enregistre l'image dans un fichier temporaire
    img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


def test_is_image_file():
    assert is_image_file("leaf.jpg") is True
    assert is_image_file("leaf.jpeg") is True
    assert is_image_file("leaf.png") is True
    assert is_image_file("leaf.bmp") is False
    assert is_image_file("doc.pdf") is False


def test_gaussian_blur(sample_image):
    transformer = ImageTransformation()
    img = cv2.imread(str(sample_image))
    blurred = transformer.gaussian_blur(img)

    assert isinstance(blurred, np.ndarray)
    assert blurred.shape == img.shape


def test_edge_detection(sample_image):
    transformer = ImageTransformation()
    img = cv2.imread(str(sample_image))
    edges = transformer.edge_detection(img)

    assert isinstance(edges, np.ndarray)
    assert edges.shape == img.shape


def test_roi(sample_image):
    transformer = ImageTransformation()
    img = cv2.imread(str(sample_image))
    roi_img = transformer.roi(img)

    assert isinstance(roi_img, np.ndarray)
    assert roi_img.shape == img.shape


def test_color_histogram(sample_image):
    transformer = ImageTransformation()
    img = cv2.imread(str(sample_image))
    hist_img = transformer.color_histogram(img)

    assert isinstance(hist_img, np.ndarray)
    assert hist_img.shape == (300, 256, 3)


def test_pseudolandmarks(sample_image):
    transformer = ImageTransformation()
    img = cv2.imread(str(sample_image))
    landmark_img = transformer.pseudolandmarks(img)

    assert isinstance(landmark_img, np.ndarray)
    assert landmark_img.shape == img.shape


def test_binary_mask(sample_image):
    transformer = ImageTransformation()
    img = cv2.imread(str(sample_image))
    mask_img = transformer.binary_mask(img)

    assert isinstance(mask_img, np.ndarray)
    assert mask_img.shape == img.shape


def test_apply_transformations(sample_image):
    img = cv2.imread(str(sample_image))
    transformations = apply_transformations(img)

    assert isinstance(transformations, dict)
    assert set(transformations.keys()) == {
        "GaussianBlur", "EdgeDetection", "ROI",
        "ColorHistogram", "Pseudolandmarks", "BinaryMask"
    }

    for name, img_trans in transformations.items():
        assert isinstance(img_trans, np.ndarray)
        assert img_trans.shape[0] > 0
        assert img_trans.shape[1] > 0

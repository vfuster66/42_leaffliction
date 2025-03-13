import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock

from app.predict import (
    load_and_prepare_image,
    predict_image,
    predict_image_top_k,
    predict_on_single_image,
    load_class_labels,
    is_image_file
)


@pytest.fixture
def sample_image(tmp_path):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 128]  # Gris uniforme
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def mock_model():
    model = MagicMock()
    # Simule predict() => 8 classes => retourne des probabilités aléatoires
    model.predict.return_value = [np.array([0.1, 0.2, 0.7] + [0]*5)]
    return model


@pytest.fixture
def class_labels():
    return ["Class1", "Class2", "Class3"] + [f"Class{i}" for i in range(4, 9)]


def test_is_image_file():
    assert is_image_file("leaf.jpg") is True
    assert is_image_file("leaf.png") is True
    assert is_image_file("leaf.docx") is False


def test_load_and_prepare_image(sample_image):
    img_tensor = load_and_prepare_image(str(sample_image))

    assert isinstance(img_tensor, np.ndarray)
    assert img_tensor.shape == (1, 224, 224, 3)
    assert (img_tensor <= 1.0).all() and (img_tensor >= 0.0).all()


def test_predict_image(mock_model):
    # Image tensor batch de 1 image
    img_tensor = np.random.rand(1, 224, 224, 3)

    idx, confidence = predict_image(mock_model, img_tensor)

    assert idx == 2
    assert abs(confidence - 0.7) < 1e-6


def test_predict_image_top_k(mock_model, class_labels):
    img_tensor = np.random.rand(1, 224, 224, 3)

    results = predict_image_top_k(mock_model, img_tensor, class_labels, k=2)

    assert len(results) == 2
    assert results[0][0] == "Class3"
    assert results[0][1] == 0.7
    assert results[1][0] == "Class2"
    assert results[1][1] == 0.2


def test_predict_on_single_image(mock_model, class_labels, sample_image):
    label, confidence = predict_on_single_image(
        mock_model, class_labels, str(sample_image)
    )

    assert label == "Class3"
    assert abs(confidence - 0.7) < 1e-6


def test_load_class_labels(tmp_path):
    (tmp_path / "ClassA").mkdir()
    (tmp_path / "ClassB").mkdir()
    (tmp_path / "file.txt").write_text("not a folder")

    labels = load_class_labels(tmp_path)

    assert "ClassA" in labels
    assert "ClassB" in labels
    assert "file.txt" not in labels

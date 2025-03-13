import os
import cv2
import pytest
import numpy as np
import zipfile

from app.train import (
    generate_sha1,
    create_zip_and_signature,
    plot_history,
    train_model
)


@pytest.fixture
def dummy_model_file(tmp_path):
    file = tmp_path / "model.h5"
    with open(file, "wb") as f:
        f.write(os.urandom(1024))  # 1KB fichier fake
    return file


@pytest.fixture
def dummy_output_dir(tmp_path):
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    return out_dir


def test_generate_sha1(dummy_model_file):
    sha1 = generate_sha1(dummy_model_file)
    assert isinstance(sha1, str)
    assert len(sha1) == 40


def test_create_zip_and_signature(dummy_model_file, tmp_path):
    zip_file = tmp_path / "model.zip"
    sig_file = tmp_path / "signature.txt"

    create_zip_and_signature(
        str(dummy_model_file), str(zip_file), str(sig_file)
    )

    # Le fichier zip doit exister et contenir le modèle
    assert os.path.exists(zip_file)
    with zipfile.ZipFile(zip_file, 'r') as z:
        assert "model.h5" in z.namelist()

    # Le fichier de signature doit exister
    assert os.path.exists(sig_file)
    with open(sig_file, 'r') as f:
        signature = f.read()
        assert "model.zip" in signature


class DummyHistory:
    history = {
        'accuracy': [0.5, 0.6, 0.7],
        'val_accuracy': [0.4, 0.5, 0.6],
        'loss': [1.0, 0.8, 0.6],
        'val_loss': [1.2, 1.0, 0.9]
    }


def test_plot_history(dummy_output_dir):
    history = DummyHistory()
    plot_history(history, dummy_output_dir)

    plot_file = dummy_output_dir / "training_history.png"
    assert plot_file.exists()


@pytest.fixture
def dummy_dataset(tmp_path):
    dataset_dir = tmp_path / "balanced"
    os.makedirs(dataset_dir / "ClassA")
    os.makedirs(dataset_dir / "ClassB")

    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(5):
        cv2.imwrite(str(dataset_dir / "ClassA" / f"imgA_{i}.jpg"), dummy_img)
        cv2.imwrite(str(dataset_dir / "ClassB" / f"imgB_{i}.jpg"), dummy_img)

    return dataset_dir


def test_train_model_runs(dummy_dataset, tmp_path):
    model_dir = tmp_path / "models"
    zip_file = tmp_path / "dataset.zip"
    sig_file = tmp_path / "signature.txt"

    train_model(
        balanced_dir=str(dummy_dataset),
        output_model_dir=str(model_dir),
        zip_output_file=str(zip_file),
        signature_file=str(sig_file),
        img_size=(64, 64),  # accélérer
        batch_size=2,
        epochs=1,
        validation_split=0.5
    )

    # Résultats attendus :
    assert os.path.exists(zip_file)
    assert os.path.exists(sig_file)
    assert any(file.endswith(".h5") for file in os.listdir(model_dir))

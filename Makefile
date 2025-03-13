# CONFIGURATION
IMAGE_NAME = leaffliction
WORKDIR = /app
PROJECT_DIR = $(PWD)
IMAGES_DIR = $(PROJECT_DIR)/images
OUTPUT_DIR = $(PROJECT_DIR)/output

DOCKER_RUN = @docker run --rm -t \
	-v $(PROJECT_DIR):$(WORKDIR) \
	-v $(IMAGES_DIR):/app/images \
	-v $(OUTPUT_DIR):/app/output \
	-e PYTHONPATH=/app \
	$(IMAGE_NAME)

# DOCKER ENVIRONNEMENT
build:
	@echo "🔨 Build de l'image Docker $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME) .

shell:
	@echo "🐚 Shell interactif dans Docker..."
	@docker run -it --rm \
		-v $(PROJECT_DIR):$(WORKDIR) \
		-v $(IMAGES_DIR):/app/images \
		-v $(OUTPUT_DIR):/app/output \
		$(IMAGE_NAME) bash

# EXPLORATION / ANALYSE
run-distribution:
	@echo "📊 Analyse Distribution sur images/..."
	$(DOCKER_RUN) python3 app/Distribution.py /app/images --suffix original

run-distribution-balanced:
	@echo "📊 Analyse Distribution sur output/balanced/..."
	$(DOCKER_RUN) python3 app/Distribution.py /app/output/balanced --suffix balanced

# DATA AUGMENTATION + BALANCE
run-augmentation:
	@echo "🧬 Data Augmentation sur images/ ➜ output/augmented..."
	$(DOCKER_RUN) python3 app/Augmentation.py /app/images --output /app/output/augmented

balance-augmented:
	@echo "⚖️  Équilibrage ➜ output/balanced..."
	$(DOCKER_RUN) python3 app/BalanceDataset.py /app/output/augmented /app/output/balanced --size 500

# TRANSFORMATIONS VISUELLES (OPTIONNEL)
run-transformation-balanced:
	@echo "🖌️  Transformation sur output/balanced ➜ output/transformations..."
	$(DOCKER_RUN) python3 app/Transformation.py /app/output/balanced -dst /app/output/transformations

# ENTRAINEMENT
train:
	@echo "🚀 Entraînement du modèle CNN ➜ output/models..."
	$(DOCKER_RUN) python3 app/train.py --balanced_dir /app/output/balanced --output_model_dir /app/output/models

# PRÉDICTIONS
predict-image:
	@echo "🔮 Prédiction sur une image (ex: Apple_healthy1.JPG)"
	$(DOCKER_RUN) python3 app/predict.py --input /app/Unit_test1/Apple_healthy1.JPG

predict-folder-unit1:
	@echo "🔮 Prédictions sur Unit_test1 (Apple)..."
	$(DOCKER_RUN) python3 app/predict.py --input /app/Unit_test1 --output /app/output/predictions --confusion

predict-folder-unit2:
	@echo "🔮 Prédictions sur Unit_test2 (Grape)..."
	$(DOCKER_RUN) python3 app/predict.py --input /app/Unit_test2 --output /app/output/predictions --confusion

# DIAGNOSTIC
diagnostic-model-unit1:
	@echo "🔍 Diagnostic du modèle sur Unit_test1..."
	$(DOCKER_RUN) python3 /app/model_diagnostic.py --model /app/output/models/leaffliction_model.h5 --test_dir /app/Unit_test1 --output /app/output/diagnostic

diagnostic-model-unit2:
	@echo "🔍 Diagnostic du modèle sur Unit_test2..."
	$(DOCKER_RUN) python3 /app/model_diagnostic.py --model /app/output/models/leaffliction_model.h5 --test_dir /app/Unit_test2 --output /app/output/diagnostic

diagnostic-single-image:
	@echo "🔍 Diagnostic sur une image spécifique..."
	$(DOCKER_RUN) python3 /app/model_diagnostic.py --model /app/output/models/leaffliction_model.h5 --test_dir /app/Unit_test1 --image $(IMAGE) --output /app/output/diagnostic

# CLEAN
clean:
	@echo "🧹 Nettoyage complet..."
	rm -rf output/models/* \
	       output/plots/* \
	       output/predictions/* \
	       output/reports/* \
	       output/augmented/* \
	       output/transformations/* \
	       output/balanced/* \
		   output/diagnostic/* \
		   train_dataset/ \
		   output/

# TESTS UNITAIRES
test:
	@echo "🧪 Lancement de tous les tests avec pytest..."
	$(DOCKER_RUN) pytest tests/ --disable-warnings -v

test-distribution:
	@echo "🧪 Test unitaire de Distribution.py..."
	$(DOCKER_RUN) pytest tests/test_distribution.py --disable-warnings -v

test-augmentation:
	@echo "🧪 Test unitaire de Augmentation.py..."
	$(DOCKER_RUN) pytest tests/test_augmentation.py --disable-warnings -v

test-transformation:
	@echo "🧪 Test unitaire de Transformation.py..."
	$(DOCKER_RUN) pytest tests/test_transformation.py --disable-warnings -v

test-train:
	@echo "🧪 Test unitaire de train.py..."
	$(DOCKER_RUN) pytest tests/test_train.py --disable-warnings -v

test-predict:
	@echo "🧪 Test unitaire de predict.py..."
	$(DOCKER_RUN) pytest tests/test_predict.py --disable-warnings -v

super-clean:
	@echo "🔥 Super Clean : Suppression des fichiers et des images Docker..."

	# Supprime tout sauf les dossiers et fichiers essentiels
	find . -mindepth 1 -maxdepth 1 \
		! -name 'Unit_test1' \
		! -name 'Unit_test2' \
		! -name 'app' \
		! -name 'images' \
		! -name 'sujet' \
		! -name 'tests' \
		! -name 'Dockerfile' \
		! -name 'Makefile' \
		! -name 'requirements.txt' \
		! -name '.' \
		-exec rm -rf {} +

	@echo "🧹 Suppression des containers Docker arrêtés..."
	docker container prune -f

	@echo "🧹 Suppression des images Docker inutilisées sauf 'leaffliction'..."
	docker images --format "{{.Repository}} {{.ID}}" | grep -v leaffliction | awk '{print $$2}' | xargs -r docker rmi -f

	@echo "✅ Super Clean terminé !"


# PIPELINE COMPLET
pipeline: clean build run-distribution run-augmentation balance-augmented run-transformation-balanced run-distribution-balanced train predict-folder-unit1 predict-folder-unit2
	@echo "🚀 Pipeline complet terminé avec succès !"

zip-dataset:
	@echo "📦 Création de dataset_leafflection.zip avec images/ Unit_test1/ Unit_test2/..."
	zip -r dataset_leafflection.zip images Unit_test1 Unit_test2

test-flake8:
	@echo "🧹 Lancement de flake8 sur tout le projet..."
	$(DOCKER_RUN) flake8 . --config=.flake8
